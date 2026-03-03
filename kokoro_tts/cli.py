from logging import disable
import click
import os
import sys
from pathlib import Path
import threading
import re
from tqdm import tqdm
from .processor import TTSProcessor
from .parsers import EpubParser, PdfParser
from .utils import spinning_wheel, dynamic_print
import sounddevice as sd
import numpy as np
import soundfile as sf


@click.group()
def main():
    """Kokoro TTS: A high-quality text-to-speech CLI."""
    pass


@main.command()
@click.argument("input_file")
@click.argument("output_file", required=False)
@click.option("--voice", default="af_heart", help="Voice ID")
@click.option("--speed", default=1.0, type=float, help="Speech speed")
@click.option("--lang", default="a", help="Language code")
@click.option("--stream", is_flag=True, help="Stream audio in real-time")
@click.option(
    "--split-output", type=click.Path(), help="Directory to save chapter files"
)
def convert(input_file, output_file, voice, speed, lang, stream, split_output):
    """Convert text, EPUB, or PDF to audio with real-time saving."""

    lang_map = {"en": "a", "en-us": "a", "en-gb": "b"}
    final_lang = lang_map.get(lang.lower(), lang)
    processor = TTSProcessor(lang_code=final_lang)

    # --- Input Handling ---
    if input_file == "-":
        content = sys.stdin.read()
        chapters = [{"title": "STDIN", "content": content, "order": 1}]
    elif input_file.endswith(".epub"):
        chapters = EpubParser.extract_chapters(input_file)
    elif input_file.endswith(".pdf"):
        chapters = PdfParser(input_file).get_chapters()
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            chapters = [{"title": "File", "content": f.read(), "order": 1}]

    for chapter in chapters:
        if not chapter["content"].strip():
            continue

        # --- Filename Cleaning ---
        safe_title = chapter["title"] or f"Chapter_{chapter['order']:02d}"
        safe_title = re.sub(
            r"^(?:\d+_)?(?:xhtml_\d+_)?", "", safe_title, flags=re.IGNORECASE
        )
        safe_title = re.sub(r"[^\w\d_-]", "_", safe_title)
        safe_title = re.sub(r"_+", "_", safe_title).strip("_")

        if split_output:
            os.makedirs(split_output, exist_ok=True)
            out_file = os.path.join(
                split_output, f"{chapter['order']:02d}_{safe_title}.wav"
            )
        else:
            out_file = output_file or "output.wav"

        # --- Processing & Saving on the fly ---
        if not Path(out_file).is_file() or stream:
            click.secho(f"\n# Processing: {chapter['title']}", fg="cyan", bold=True)

            chunk_size = 500
            total_chunks = (len(chapter["content"]) + chunk_size - 1) // chunk_size

            chunks = processor.stream_generator(
                chapter["content"], voice=voice, speed=speed, chunk_size=chunk_size
            )

            # Open file for writing immediately
            with sf.SoundFile(out_file, mode="w", samplerate=24000, channels=1) as f:
                for _, audio in tqdm(
                    chunks,
                    desc=f"Chapter {chapter['order']:02d}",
                    unit="chunk",
                    total=total_chunks,
                ):
                    # Write to disk as we go
                    f.write(audio)

                    if stream:
                        sd.play(audio, 24000)
                        sd.wait()

            click.secho(f"✔ Completed: {out_file}", fg="green")
        else:
            click.secho(f"Skipping (exists): {out_file}", fg="yellow")


if __name__ == "__main__":
    main()
