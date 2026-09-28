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

from .breeze_fast import FastDepth
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
    """Wrapper around Breeze TTS 2 (mlx-audio + FastDepth) for text-to-speech
    on Apple Silicon.

    * Runs the mlx-audio Breeze implementation with the FastDepth patch
      (vendored in ``breeze_fast.py``), which fixes the depth decoder's
      missing intra-frame KV reuse — roughly a 2-4x speedup — and uses 4bit
      weights by default (~3 GB).
    * Voice design: pass ``instruction`` (a natural-language voice
      description). Use ``cfg_scale=4`` to strengthen instruction following
      (at extra compute cost per frame).
    * Voice clone: pass ``ref_audio`` (path to clean reference audio)
      together with ``ref_text`` (its exact transcript).
    * Voice direction: pass all three to clone the reference voice while
      steering tone, emotion, and pace via the instruction.
    * Auto-anchoring: without ``ref_audio``, the first generation creates a
      short anchor utterance in the designed/default voice, which is then
      used as the reference for every subsequent chunk — keeping one stable
      voice across chapters. Reuse the processor across chapters/files to
      keep the same voice.
    * Inline vocal events are supported in the text, e.g. ``(laugh)``,
      ``(sigh)``, ``(cough)``, ``(clears throat)``.
    * The model is downloaded automatically from the Hugging Face Hub on
      first use, or pass a local directory via ``model`` / the
      ``BREEZE_TTS_MODEL`` environment variable.
    """

    DEFAULT_MODEL = "mlx-community/Breeze-TTS-2-mlx-4bit"
    DEFAULT_CHUNK_SIZE = 600
    CHUNK_PAUSE = 0.25
    SILENCE_RMS = 0.02
    SILENCE_STOP_SECONDS = 4.0
    ALL_SILENT_STOP_SECONDS = 10.0
    MAX_TOKENS_CEILING = 2000
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
        depth_mode="cached",
    ):
        import mlx.core as mx
        from mlx_audio.tts.models.breeze_tts.breeze_tts import Model
        from mlx_audio.tts.utils import load_model

        self.instruction = instruction
        self.ref_audio = str(ref_audio) if ref_audio else None
        self.ref_text = ref_text
        self.cfg_scale = float(cfg_scale or 1.0)
        self.seed = seed
        self.sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p)

        if (self.ref_audio is None) != (self.ref_text is None):
            raise ValueError(
                "ref_audio and ref_text must be provided together for voice cloning"
            )

        model_dir = self._resolve_model(model)
        self.model = load_model(model_dir)
        if not isinstance(self.model, Model):
            raise ValueError(
                f"'{model_dir}' is not a Breeze TTS 2 checkpoint "
                f"(got {type(self.model).__name__})"
            )
        self.sample_rate = int(self.model.sample_rate)
        self._anchor_dir = None
        self._max_frames_seen = 0

        decode_rate = getattr(self.model.audio_tokenizer, "decode_upsample_rate", None)
        if decode_rate is None:
            decode_rate = getattr(
                self.model.audio_tokenizer.decoder, "decode_upsample_rate", None
            )
        if not decode_rate or decode_rate <= 0:
            raise ValueError("Breeze audio tokenizer has no valid decode rate")
        # Stream PCM in ~2-codec-frame slices for responsive silence checks.
        self.streaming_interval = (2 + 1e-6) * decode_rate / self.sample_rate

        self._fast_depth = FastDepth(self.model, depth_mode)
        self._fast_depth.install()
        mx.eval(self.model.parameters())
        if self.ref_audio:
            self._cache_ref_encoding()

    def _resolve_model(self, model):
        """Return a local checkpoint directory, downloading from the Hub if
        a repo id is given."""
        candidate = model or os.environ.get("BREEZE_TTS_MODEL") or self.DEFAULT_MODEL
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
        from mlx_audio.utils import get_model_path

        tqdm.write(
            f"Downloading Breeze TTS 2 model '{candidate}' "
            "(~3 GB, first run only)..."
        )
        return get_model_path(candidate)

    def _cache_ref_encoding(self):
        """Encode the reference audio once and pin it on the model instance,
        so per-chunk generation skips the (~5s) re-encode."""
        import mlx.core as mx
        from mlx_audio.tts.models.breeze_tts.breeze_tts import Model

        codes = Model._encode_reference(self.model, self.ref_audio)
        mx.eval(codes)
        self.model._encode_reference = lambda ref: codes

    def stream_generator(self, text, voice=None, speed=1.0, chunk_size=None):
        """
        Yields (text_chunk, audio_data) with text split at sentence
        boundaries, since generation length is capped per chunk.
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
        self._cache_ref_encoding()

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

    def _max_tokens_for(self, text):
        """Generation cap sized from the text: ~15 chars/s of speech at 25
        codec frames/s, with headroom for pauses."""
        return min(self.MAX_TOKENS_CEILING, max(750, int(len(text) * 1.8) + 100))

    def _generate_chunk_audio(self, text):
        """Yield numpy audio arrays for one text chunk."""
        import mlx.core as mx
        import numpy as np

        ref_audio = self.ref_audio or None
        ref_text = self.ref_text if ref_audio else None
        max_tokens = self._max_tokens_for(text)
        results = self.model.generate(
            text=text,
            instruct=self.instruction or None,
            ref_audio=ref_audio,
            ref_text=ref_text,
            cfg_scale=self.cfg_scale,
            max_tokens=max_tokens,
            seed=self.seed,
            stream=True,
            streaming_interval=self.streaming_interval,
            **self.sampling,
        )
        generated = 0.0
        silent = 0.0
        loud_seen = False
        total_frames = 0
        try:
            for result in results:
                audio = np.array(result.audio.astype(mx.float32), copy=True)
                duration = audio.size / self.sample_rate
                generated += duration
                total_frames += int(result.token_count)
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
                yield audio
            if total_frames >= max_tokens:
                tqdm.write(
                    "warning: text chunk hit the generation length limit; audio "
                    "may end abruptly — try a smaller chunk size"
                )
        finally:
            try:
                results.close()
            finally:
                self.model.audio_tokenizer.decoder.reset_streaming_state()

    def generate_audio(self, text, output_path, speed=1.0):
        """Generate audio using Breeze TTS 2 (speed is not supported; steer
        pace through the instruction instead)."""
        self.save(text, output_path, speed=speed)

    def save(self, text, output_path, voice=None, speed=1.0, chunk_size=None):
        """
        Save text to a WAV file, streaming audio chunks to disk. If
        chunk_size is provided, text is split at sentence boundaries into
        chunks of roughly that many characters (default 600).
        """
        with sf.SoundFile(
            output_path, "w", samplerate=self.sample_rate, channels=1, subtype="PCM_16"
        ) as f:
            for _, audio in self.stream_generator(
                text, speed=speed, chunk_size=chunk_size
            ):
                f.write(audio)
