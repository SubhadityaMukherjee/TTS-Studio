from logging import disable
import click
import os
import sys
import threading
import sounddevice as sd
from .processor import TTSProcessor
from .parsers import EpubParser, PdfParser
from .utils import spinning_wheel, dynamic_print


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
@click.option("--stream", is_flag=True, help="Stream audio with typewriter effect")
@click.option(
    "--display/--no-display", default=True, help="Enable/Disable typewriter effect"
)
@click.option("--split-output", type=click.Path(), help="Directory to save chunks")
def convert(input_file, output_file, voice, speed, lang, stream, display, split_output):
    lang_map = {"en": "a", "en-us": "a", "en-gb": "b"}
    final_lang = lang_map.get(lang.lower(), lang)
    processor = TTSProcessor(lang_code=final_lang)

    # Input Handling
    if input_file == "-":
        content = sys.stdin.read()
        chapters = [{"title": "STDIN", "content": content, "order": 1}]
    elif input_file.endswith(".epub"):
        chapters = EpubParser.extract_chapters(input_file)
    elif input_file.endswith(".pdf"):
        chapters = PdfParser(input_file).get_chapters()
    else:
        with open(input_file, "r") as f:
            chapters = [{"title": "File", "content": f.read(), "order": 1}]

    for chapter in chapters:
        if not chapter["content"].strip():
            continue

        if stream:
            click.secho(f"\n# Reading: {chapter['title']}", fg="cyan", bold=True)
            for gs, audio in processor.stream_generator(
                chapter["content"], voice=voice, speed=speed
            ):
                # Calculate duration: samples / sample_rate
                duration = len(audio) / 24000

                # Start audio (non-blocking)
                sd.play(audio, 24000)

                if display:
                    # Typewriter print while audio plays
                    dynamic_print(gs, duration)

                # Wait for audio to finish before next chunk
                sd.wait()
            print("\n")  # New line at the end

        else:
            # Saving logic remains the same...
            out = output_file or "output.wav"
            processor.save(chapter["content"], out, voice=voice, speed=speed)
            click.secho(f"✔ Saved to {out}", fg="green")


if __name__ == "__main__":
    main()
