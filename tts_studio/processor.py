import asyncio
import logging
import os
import re
import warnings

import edge_tts
import nltk
import sounddevice as sd
import soundfile as sf
import torch
from kokoro import KPipeline

from .utils import split_paragraphs, split_sentences

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
        self.sentence_pause = float(os.environ.get("TTS_SENTENCE_PAUSE", sentence_pause))
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
