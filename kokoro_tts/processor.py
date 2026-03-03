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
        return self.pipeline(text, voice=voice, speed=speed)

    def stream_generator(self, text, voice="af_heart", speed=1.0):
        """Yields (text_segment, audio_tensor) for the CLI to handle."""
        generator = self.generate_audio(text, voice, speed)
        for gs, ps, audio in generator:
            if audio is not None:
                yield gs, audio

    def save(self, text, output_path, voice="af_heart", speed=1.0):
        generator = self.generate_audio(text, voice, speed)
        all_audio = [audio for _, _, audio in generator if audio is not None]
        if all_audio:
            combined = torch.cat(all_audio)
            sf.write(output_path, combined.numpy(), 24000)
