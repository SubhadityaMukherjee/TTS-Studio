import logging
import os
import re
import warnings

import nltk
import sounddevice as sd
import soundfile as sf
import torch
from kokoro import KPipeline

nltk.download("punkt", quiet=True)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


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
    def __init__(self, lang_code="a"):
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

    def stream_generator(self, text, voice="af_heart", speed=1.0):
        """
        Yields (text_chunk, audio_tensor) for CLI streaming.
        Splits text into chunks internally to allow progress tracking.
        """
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            if sent.strip():
                for gs, ps, audio in self.generate_audio(sent.strip(), voice, speed):
                    if audio is not None:
                        yield sent, audio
                # clear cache between sentences to keep memory low
                self._clear_memory()

    def save(self, text, output_path, voice="af_heart", speed=1.0, chunk_size=None):
        """
        Save text to WAV file.
        If chunk_size is provided, splits text into chunks and appends audio.
        """
        all_audio = []

        if chunk_size is not None:
            # Chunked saving
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i : i + chunk_size]
                generator = self.generate_audio(chunk_text, voice, speed)
                for _, _, audio in generator:
                    if audio is not None:
                        all_audio.append(audio)
                self._clear_memory()
        else:
            # Single-shot saving
            generator = self.generate_audio(text, voice, speed)
            for _, _, audio in generator:
                if audio is not None:
                    all_audio.append(audio)
            self._clear_memory()

        if all_audio:
            combined = torch.cat(all_audio)
            sf.write(output_path, combined.numpy(), 24000)
