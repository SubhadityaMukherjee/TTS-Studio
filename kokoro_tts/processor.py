import warnings
import logging
import torch
import sounddevice as sd
import soundfile as sf

from kokoro import KPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


class TTSProcessor:
    def __init__(self, lang_code="a"):
        self.pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")

    def generate_audio(self, text, voice="af_heart", speed=1.0):
        """Full audio generator (yields gs, ps, audio)"""
        return self.pipeline(text, voice=voice, speed=speed)

    def stream_generator(self, text, voice="af_heart", speed=1.0, chunk_size=500):
        """
        Yields (text_chunk, audio_tensor) for CLI streaming.
        Splits text into chunks internally to allow progress tracking.
        """
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            generator = self.generate_audio(chunk_text, voice, speed)
            for gs, ps, audio in generator:
                if audio is not None:
                    yield chunk_text, audio

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
        else:
            # Single-shot saving
            generator = self.generate_audio(text, voice, speed)
            for _, _, audio in generator:
                if audio is not None:
                    all_audio.append(audio)

        if all_audio:
            combined = torch.cat(all_audio)
            sf.write(output_path, combined.numpy(), 24000)
