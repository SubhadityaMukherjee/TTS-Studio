import click
import os
import sys
from pathlib import Path
import re
from tqdm import tqdm
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .processor import TTSProcessor
from .parsers import EpubParser, PdfParser


@click.group()
def main():
    """Kokoro TTS: A high-quality text-to-speech CLI."""
    pass


def process_chapter(chapter, voice, speed, lang, split_output):
    """Run in a separate process for TTS conversion."""

    # --- Safe filename formatting ---
    safe_title = chapter.get("title", f"Chapter_{chapter['order']:02d}")
    safe_title = re.sub(
        r"^(?:\d+_)?(?:xhtml_\d+_)?", "", safe_title, flags=re.IGNORECASE
    )
    safe_title = re.sub(r"[^\w\d_-]+", "_", safe_title)
    safe_title = re.sub(r"_+", "_", safe_title).strip("_")

    out_file = os.path.join(split_output, f"{chapter['order']:02d}_{safe_title}.wav")

    # --- Skip if file already exists ---
    if os.path.exists(out_file):
        return out_file, "skipped"

    processor = TTSProcessor(lang_code=lang)

    # Process with progress-aware streaming
    generator = processor.stream_generator(chapter["content"], voice=voice, speed=speed)
    with sf.SoundFile(out_file, "w", samplerate=24000, channels=1) as f:
        for _, audio in generator:
            f.write(audio)

    return out_file, "completed"


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
    """Convert text, EPUB, or PDF to audio, with progress bars and file skipping."""

    # --- Language mapping ---
    lang_map = {"en": "a", "en-us": "a", "en-gb": "b"}
    final_lang = lang_map.get(lang.lower(), lang)
    processor = TTSProcessor(lang_code=final_lang)

    # --- Input handling ---
    if input_file == "-":
        content = sys.stdin.read()
        chapters = [{"title": "STDIN", "content": content, "order": 1}]
    elif input_file.endswith(".epub"):
        chapters = EpubParser.extract_chapters(input_file)
    elif input_file.endswith(".pdf"):
        chapters = PdfParser(input_file).get_chapters()
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            chapters = [
                {"title": Path(input_file).stem, "content": f.read(), "order": 1}
            ]

    if not chapters:
        click.secho("❌ No chapters found to process.", fg="red")
        sys.exit(1)

    os.makedirs(split_output or ".", exist_ok=True)
    max_workers = min(4, multiprocessing.cpu_count() // 2)

    click.secho(
        f"🔄 Starting conversion of {len(chapters)} chapters "
        f"(max {max_workers} workers)...\n",
        fg="cyan",
    )

    results = []
    skipped_count = 0

    # --- Use a progress bar for better UX ---
    with tqdm(total=len(chapters), desc="Processing chapters", ncols=80) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_chapter, ch, voice, speed, final_lang, split_output
                ): ch
                for ch in chapters
                if ch["content"].strip()
            }

            for future in as_completed(futures):
                chapter = futures[future]
                try:
                    out_path, status = future.result()
                    if status == "skipped":
                        skipped_count += 1
                        click.secho(f"⏩ Skipped (exists): {out_path}", fg="yellow")
                    else:
                        click.secho(f"✔ Completed: {out_path}", fg="green")
                    results.append(out_path)
                except Exception as e:
                    click.secho(
                        f"❌ Error processing chapter {chapter['title']}: {e}", fg="red"
                    )
                finally:
                    pbar.update(1)

    click.secho(f"\n🏁 All chapters processed! ({skipped_count} skipped)", fg="cyan")


if __name__ == "__main__":
    main()
