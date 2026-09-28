import asyncio
import logging
import os
import re
import uuid
import warnings
from pathlib import Path

import edge_tts
import nltk
import sounddevice as sd
import soundfile as sf
import torch
from kokoro import KPipeline
from tqdm import tqdm

from .utils import chunk_text, split_paragraphs, split_sentences

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

SAMPLE_RATE = 24000


class TTSProcessor:
    """Wrapper around Kokoro pipeline for text-to-speech with device
    management.

    * Automatically selects ``mps`` (Metal) or ``cuda`` backends when
      available, falling back to CPU.
    * Sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` to avoid occasional kernel
      failures on Apple silicon.
    * On MPS you can limit memory usage by setting
      ``TORCH_MPS_MEMORY_FRACTION`` (default ``0.8``) before launching the
      program; the constructor will call ``torch.mps.set_per_process_memory_fraction``
      if the API is available.
    * After every generation chunk the processor empties the PyTorch cache and
      runs ``gc.collect()`` to help avoid ``RuntimeError: out of memory``
      when working with large batches or long texts.
    """

    def __init__(self, lang_code="a", sentence_pause=0.25, paragraph_pause=0.6):
        # Pauses (seconds) stitched between synthesized chunks. Without them
        # each sentence is cut hard into the next, which sounds like weird
        # abrupt breaks. Set either to 0 to disable.
        self.sentence_pause = float(
            os.environ.get("TTS_SENTENCE_PAUSE", sentence_pause)
        )
        self.paragraph_pause = float(
            os.environ.get("TTS_PARAGRAPH_PAUSE", paragraph_pause)
        )

        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS (Metal) acceleration")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            # constrain memory usage on MPS to avoid OOM; the value is a
            # fraction of total device memory.  Users can override by setting
            # TORCH_MPS_LOW_MEMORY environment variable before launch.
            frac = float(os.environ.get("TORCH_MPS_MEMORY_FRACTION", "0.9"))
            try:
                torch.mps.set_per_process_memory_fraction(frac)
            except Exception:
                # older torch versions don't have this API
                pass
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA (NVIDIA GPU) acceleration")
        else:
            self.device = "cpu"
            print("MPS unavailable, falling back to CPU")

        self.pipeline = KPipeline(
            lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=self.device
        )

    def _clear_memory(self):
        """Release any cached GPU/MPS memory and run the garbage collector."""
        # PyTorch provides separate empty_cache functions for each backend
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps" and torch.backends.mps.is_available():
                # MPS added empty_cache() in newer releases, safe to call
                torch.mps.empty_cache()
        except Exception:  # pragma: no cover - safe fallback
            pass
        import gc

        gc.collect()

    def generate_audio(self, text, voice="af_heart", speed=1.0):
        """Full audio generator (yields gs, ps, audio).

        The underlying pipeline may allocate intermediate tensors on the
        selected device; once the iterator is exhausted we clear any cached
        memory to avoid OOMs on constrained backends like MPS.
        """
        # use no_grad to avoid keeping graph on device
        with torch.no_grad():
            for gs, ps, audio in self.pipeline(text, voice=voice, speed=speed):
                yield gs, ps, audio
        # after the caller finishes iterating, free cache
        self._clear_memory()

    def _silence(self, seconds):
        """Silence tensor of the given duration at the pipeline sample rate."""
        return torch.zeros(int(SAMPLE_RATE * seconds))

    def stream_generator(self, text, voice="af_heart", speed=1.0):
        """
        Yields (text_chunk, audio_tensor) for CLI streaming.

        Splits text into paragraphs and sentences, synthesizes each
        separately, and stitches in natural pauses between sentences
        (sentence_pause) and paragraphs (paragraph_pause) so the audio
        doesn't cut abruptly between chunks.
        """
        first_chunk = True
        for para in split_paragraphs(text):
            for sent in split_sentences(para):
                if not sent.strip():
                    continue
                if not first_chunk and self.sentence_pause > 0:
                    yield "", self._silence(self.sentence_pause)
                first_chunk = False
                for gs, ps, audio in self.generate_audio(sent.strip(), voice, speed):
                    if audio is not None:
                        yield sent, audio
                # clear cache between sentences to keep memory low
                self._clear_memory()
            if self.paragraph_pause > 0:
                yield "", self._silence(self.paragraph_pause)
                first_chunk = True

    def save(self, text, output_path, voice="af_heart", speed=1.0, chunk_size=None):
        """
        Save text to WAV file with natural pauses between sentences and
        paragraphs. (chunk_size is accepted for API compatibility; the text
        is always streamed sentence-by-sentence.)
        """
        all_audio = []
        for _, audio in self.stream_generator(text, voice, speed):
            all_audio.append(audio)

        if all_audio:
            combined = torch.cat(all_audio)
            sf.write(output_path, combined.numpy(), SAMPLE_RATE)


class EdgeTTSProcessor:
    """Wrapper around edge-tts for text-to-speech."""

    def __init__(self, voice="en-US-AvaMultilingualNeural"):
        self.voice = voice

    async def _generate_audio_async(self, text, output_path, rate=None):
        """Generate audio using edge-tts asynchronously."""
        communicate = edge_tts.Communicate(text, self.voice, rate=rate)
        await communicate.save(output_path)

    def generate_audio(self, text, output_path, speed=1.0):
        """Generate audio using edge-tts (synchronous wrapper)."""
        # edge-tts expresses rate as a percentage offset from normal pace
        rate = f"{round((speed - 1.0) * 100):+d}%"
        asyncio.run(self._generate_audio_async(text, output_path, rate=rate))

    def stream_generator(self, text, voice=None, speed=1.0):
        """
        Yields (text_chunk, audio_data) for CLI streaming.
        Note: edge-tts doesn't support streaming, so this processes whole text.
        """
        import tempfile

        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        self.generate_audio(text, tmp_path, speed=speed)

        data, samplerate = sf.read(tmp_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        yield text, torch.from_numpy(data)

        os.unlink(tmp_path)

    def save(self, text, output_path, voice=None, speed=1.0, chunk_size=None):
        """
        Save text to audio file.
        If chunk_size is provided, splits text into chunks and concatenates.
        """
        if chunk_size is not None:
            import tempfile

            temp_files = []
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i : i + chunk_size]
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_file = tmp.name
                self.generate_audio(chunk_text, tmp_file, speed=speed)
                temp_files.append(tmp_file)

            import numpy as np

            all_data = []
            samplerate = None
            for tmp_file in temp_files:
                data, sr = sf.read(tmp_file)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                all_data.append(data)
                if samplerate is None:
                    samplerate = sr
                os.unlink(tmp_file)

            if all_data:
                combined = np.concatenate(all_data)
                sf.write(output_path, combined, samplerate)
        else:
            self.generate_audio(text, output_path, speed=speed)


class BreezeTTSProcessor:
    """Wrapper around Breeze TTS 2 (MLX port) for text-to-speech on
    Apple Silicon.

    * Voice design: pass ``instruction`` (a natural-language voice
      description, e.g. ``"A warm, thoughtful young woman with a calm
      delivery"``). Use ``cfg_scale=4`` to strengthen instruction
      following.
    * Voice clone: pass ``ref_audio`` (path to clean reference audio)
      together with ``ref_text`` (its exact transcript).
    * Voice direction: pass all three to clone the reference voice while
      steering tone, emotion, and pace via the instruction.
    * Auto-anchoring: without ``ref_audio``, the first generation creates a
      short anchor utterance in the designed/default voice, which is then
      used as the reference for every subsequent chunk — keeping one stable
      voice across chapters instead of a freshly sampled voice each chunk.
      Reuse the processor across chapters/files to keep the same voice.
    * Inline vocal events are supported in the text, e.g. ``(laugh)``,
      ``(sigh)``, ``(cough)``, ``(clears throat)``.
    * The model weights (INT8, ~3.7 GB) are downloaded automatically from
    the Hugging Face Hub on first use, or pass a local checkpoint directory
    via ``model`` / the ``BREEZE_TTS_MODEL`` environment variable.
    * The runtime is heavy (3B params), so reuse one processor instance for
    many texts rather than creating a new one per chapter.
    """

    DEFAULT_REPO = "rishikksh20/Breeze-TTS-2-mlx"
    DEFAULT_CHUNK_SIZE = 600
    CHUNK_PAUSE = 0.25
    SILENCE_RMS = 0.02
    SILENCE_STOP_SECONDS = 4.0
    ALL_SILENT_STOP_SECONDS = 10.0
    ANCHOR_SENTENCE = (
        "This is my voice, recorded once so that every part of this "
        "reading sounds exactly like me."
    )

    def __init__(
        self,
        model=None,
        instruction=None,
        ref_audio=None,
        ref_text=None,
        cfg_scale=1.0,
        seed=42,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
    ):
        from breeze_tts_mlx.runtime import BreezeMLXRuntime, MLXRuntimeConfig
        from breeze_tts_mlx.sampling import SamplingConfig

        self.instruction = instruction
        self.ref_audio = str(ref_audio) if ref_audio else None
        self.ref_text = ref_text
        self.cfg_scale = float(cfg_scale)
        self.template_name = self._select_template()
        self._anchor_dir = None

        # The codec logs a "Residual tail decode" warning at the end of
        # every generation; it is expected (the runtime closes the request
        # right after) and floods the console, clobbering progress bars.
        logging.getLogger("breeze_tts_mlx").setLevel(logging.ERROR)

        model_dir = self._resolve_model(model)
        sampling = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
        self.runtime = BreezeMLXRuntime(
            model_dir,
            audio_device="auto",
            seed=seed,
            config=MLXRuntimeConfig(
                backbone_sampling=sampling,
                depth_sampling=sampling,
            ),
        )
        self.sample_rate = self.runtime.sample_rate
        self._patch_ref_encoding()

    def _patch_ref_encoding(self):
        """Cache reference-audio encoding. ``prepare_inputs`` re-encodes the
        ref audio through the audio tokenizer for every chunk (~5s each on
        MPS); the anchor/user ref is immutable, so encode it once."""
        import breeze_tts_mlx.templates as templates

        if getattr(templates.encode_prompt_audio, "_tts_studio_cached", False):
            return
        original = templates.encode_prompt_audio
        cache = {}

        def cached(audio_tokenizer, audio_path):
            key = str(audio_path)
            if key not in cache:
                cache[key] = original(audio_tokenizer, audio_path)
            return cache[key]

        cached._tts_studio_cached = True
        templates.encode_prompt_audio = cached

    def _select_template(self):
        """Pick the prompt template, mirroring the reference CLI's auto mode."""
        has_ref = self.ref_audio is not None or self.ref_text is not None
        if has_ref and (self.ref_audio is None or not self.ref_text):
            raise ValueError(
                "ref_audio and ref_text must be provided together for voice cloning"
            )
        if has_ref and self.instruction:
            name = "ref_edit_tata"
        elif has_ref:
            name = "ref_clone_tata"
        elif self.instruction:
            name = "tts_instruction"
        else:
            name = "tts_plain"
        if name in ("tts_plain", "ref_clone_tata") and self.cfg_scale != 1.0:
            raise ValueError(
                "cfg_scale > 1 requires an instruction (voice design/direction)"
            )
        return name

    def _resolve_model(self, model):
        """Return a local checkpoint directory, downloading from the Hub if
        a repo id is given."""
        candidate = model or os.environ.get("BREEZE_TTS_MODEL") or self.DEFAULT_REPO
        path = Path(candidate).expanduser()
        if (path / "mlx_config.json").is_file():
            return path
        from huggingface_hub import snapshot_download

        tqdm.write(
            f"Downloading Breeze TTS 2 checkpoint '{candidate}' "
            "(~3.7 GB, first run only)..."
        )
        return Path(snapshot_download(repo_id=candidate))

    def stream_generator(self, text, voice=None, speed=1.0, chunk_size=None):
        """
        Yields (text_chunk, audio_data) with text split at sentence
        boundaries, since generation length is capped (~60s per chunk).
        """
        import numpy as np

        if self.ref_audio is None:
            self._create_anchor()

        chunks = chunk_text(
            text, initial_chunk_size=chunk_size or self.DEFAULT_CHUNK_SIZE
        )
        show_progress = len(chunks) > 1
        first = True
        for index, text_chunk in enumerate(chunks, start=1):
            if not first and self.CHUNK_PAUSE > 0:
                yield "", np.zeros(
                    int(self.sample_rate * self.CHUNK_PAUSE), dtype=np.float32
                )
            first = False
            pieces = list(self._generate_chunk_audio(text_chunk))
            if pieces:
                audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                audio = self._trim_silence(audio)
                if show_progress:
                    tqdm.write(
                        f"    breeze: chunk {index}/{len(chunks)} done "
                        f"({audio.size / self.sample_rate:.1f}s audio)"
                    )
                yield text_chunk, audio

    def _create_anchor(self):
        """Generate one short anchor utterance and adopt it as the voice
        reference for all subsequent chunks, so the voice stays stable
        across chapters. Skipped (with a warning) if generation fails."""
        import numpy as np

        text = self.ANCHOR_SENTENCE
        tqdm.write(f"  🎙  Creating voice anchor for Breeze ({len(text)} chars)...")
        try:
            pieces = list(self._generate_chunk_audio(text))
            audio = np.concatenate(pieces) if pieces else None
            if audio is None or audio.size == 0:
                raise ValueError("empty anchor generation")
            audio = self._trim_silence(audio)
            if float(np.sqrt((audio.astype(np.float64) ** 2).mean())) < 0.005:
                raise ValueError("anchor generation is silent")
        except Exception as e:
            tqdm.write(
                f"warning: voice anchoring failed ({e}); falling back to "
                "per-chunk generation with an unstable voice"
            )
            self.ref_audio = ""
            self.ref_text = None
            return

        import tempfile

        self._anchor_dir = tempfile.TemporaryDirectory(prefix="breeze-anchor-")
        anchor_path = str(Path(self._anchor_dir.name) / "anchor.wav")
        sf.write(anchor_path, audio, self.sample_rate, subtype="PCM_16")

        self.ref_audio = anchor_path
        self.ref_text = text
        self.template_name = "ref_edit_tata" if self.instruction else "ref_clone_tata"

    def _trim_silence(self, audio, frame_rms_threshold=0.02, pad_seconds=0.15):
        """Trim leading/trailing near-silence from a generated chunk. The
        model occasionally emits long low-level dither frames before its EOS
        token, which would otherwise append tens of seconds of silence."""
        import numpy as np

        if audio.size == 0:
            return audio
        frame = max(1, int(self.sample_rate * 0.03))
        count = audio.size // frame
        if count == 0:
            return audio
        rms = np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1))
        loud = np.flatnonzero(rms > frame_rms_threshold)
        if loud.size == 0:
            return audio[: int(self.sample_rate * pad_seconds)]
        pad = int(self.sample_rate * pad_seconds)
        start = max(0, int(loud[0]) * frame - pad)
        end = min(audio.size, (int(loud[-1]) + 1) * frame + pad)
        return audio[start:end]

    def _generate_chunk_audio(self, text, request_id=None):
        """Yield numpy audio arrays for one text chunk."""
        from breeze_tts_mlx.templates import get_template, prepare_inputs

        request = {
            "id": request_id or f"tts-studio-{uuid.uuid4().hex}",
            "text": text,
            "speaker": "S0",
        }
        if self.instruction:
            request["instruction"] = self.instruction
        if self.ref_audio:
            request["ref_audio_path"] = self.ref_audio
            request["ref_text"] = self.ref_text.strip()

        inputs = prepare_inputs(
            self.runtime.tokenizer,
            self.runtime.audio_tokenizer,
            self.runtime,
            [request],
            get_template(self.template_name),
            guidance_scale=self.cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        request_id = request["id"]
        import numpy as np

        generated = 0.0
        silent = 0.0
        loud_seen = False
        for chunk in self.runtime.iter_audio_chunks(inputs, request_id=request_id):
            audio = chunk.audio
            duration = audio.size / self.sample_rate
            generated += duration
            rms = float(np.sqrt(np.square(audio.astype(np.float32)).mean()))
            if rms >= self.SILENCE_RMS:
                loud_seen = True
                silent = 0.0
            else:
                silent += duration
            # The model can ramble long low-level dither after the speech
            # before emitting EOS; stopping early saves that dead time (the
            # trailing silence is trimmed from the output anyway).
            if (loud_seen and silent >= self.SILENCE_STOP_SECONDS) or (
                not loud_seen and generated >= self.ALL_SILENT_STOP_SECONDS
            ):
                if loud_seen:
                    tqdm.write(
                        f"    breeze: stopped after {silent:.0f}s of trailing "
                        "silence"
                    )
                break
            if (
                chunk.timing.get("total_frames")
                >= self.runtime.runtime_config.max_new_tokens
            ):
                tqdm.write(
                    "warning: text chunk hit the generation length limit; audio "
                    "may end abruptly — try a smaller chunk size"
                )
            yield audio

    def generate_audio(self, text, output_path, speed=1.0):
        """Generate audio using Breeze TTS 2 (speed is not supported; steer
        pace through the instruction instead)."""
        self.save(text, output_path, speed=speed)

    def save(self, text, output_path, voice=None, speed=1.0, chunk_size=None):
        """
        Save text to a WAV file, streaming audio chunks to disk. If
        chunk_size is provided, text is split at sentence boundaries into
        chunks of roughly that many characters (default 600, keeping each
        generation under the model's length cap).
        """
        with sf.SoundFile(
            output_path, "w", samplerate=self.sample_rate, channels=1, subtype="PCM_16"
        ) as f:
            for _, audio in self.stream_generator(
                text, speed=speed, chunk_size=chunk_size
            ):
                f.write(audio)
