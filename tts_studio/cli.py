import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import soundfile as sf
from tqdm import tqdm

from .parsers import EpubParser, PdfParser
from .processor import EdgeTTSProcessor, TTSProcessor


@click.group()
def main():
    """Kokoro TTS: A high-quality text-to-speech CLI."""
    pass


def process_chapter(chapter, voice, speed, lang, split_output, abstract_only, engine="kokoro"):
    """Run in a separate process for TTS conversion."""

    # --- Safe filename formatting ---
    safe_title = chapter.get("title", f"Chapter_{chapter['order']:02d}")
    if abstract_only and "abstract" not in safe_title.lower():
        return f"{safe_title}.wav", "skipped"
    safe_title = re.sub(
        r"^(?:\d+_)?(?:xhtml_\d+_)?", "", safe_title, flags=re.IGNORECASE
    )
    safe_title = re.sub(r"[^\w\d_-]+", "_", safe_title)
    safe_title = re.sub(r"_+", "_", safe_title).strip("_")

    out_ext = ".mp3" if engine == "edge" else ".wav"
    out_file = os.path.join(split_output, f"{chapter['order']:02d}_{safe_title}{out_ext}")

    # --- Skip if file already exists ---
    if os.path.exists(out_file):
        return out_file, "skipped"

    if engine == "edge":
        processor = EdgeTTSProcessor(voice=voice)
        processor.save(chapter["content"], out_file, voice=voice, speed=speed)
    else:
        processor = TTSProcessor(lang_code=lang)

        # Process with progress-aware streaming
        generator = processor.stream_generator(chapter["content"], voice=voice, speed=speed)
        with sf.SoundFile(out_file, "w", samplerate=24000, channels=1) as f:
            for _, audio in generator:
                f.write(audio)

    return out_file, "completed"


def load_chapters(input_file):
    """Load chapters from a single input file."""
    if input_file.endswith(".epub"):
        return EpubParser.extract_chapters(input_file)
    elif input_file.endswith(".pdf"):
        return PdfParser(input_file).get_chapters()
    else:
        if Path(input_file).is_file():
            with open(input_file, "r", encoding="utf-8") as f:
                return [
                    {"title": Path(input_file).stem, "content": f.read(), "order": 1}
                ]
        else:
            return [{"title": "Content", "content": str(input_file), "order": 1}]


def process_single_file(
    input_file,
    voice,
    speed,
    final_lang,
    split_output,
    abstract_only,
    file_index,
    total_files,
    engine="kokoro",
):
    """Process all chapters of a single input file."""
    chapters = load_chapters(input_file)
    file_stem = Path(input_file).stem

    # If multiple files, use a subdirectory per file to avoid collisions
    if total_files > 1:
        file_out_dir = os.path.join(split_output, re.sub(r"[^\w\d_-]+", "_", file_stem))
    else:
        file_out_dir = split_output

    os.makedirs(file_out_dir, exist_ok=True)

    max_workers = min(4, multiprocessing.cpu_count() // 2)

    click.secho(
        f"\n📄 [{file_index}/{total_files}] Processing: {input_file} "
        f"({len(chapters)} chapters, max {max_workers} workers)",
        fg="cyan",
    )

    skipped_count = 0

    with tqdm(total=len(chapters), desc=f"  Chapters", ncols=80, leave=True) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_chapter,
                    ch,
                    voice,
                    speed,
                    final_lang,
                    file_out_dir,
                    abstract_only,
                    engine,
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
                        tqdm.write(
                            click.style(
                                f"  ⏩ Skipped (exists): {out_path}", fg="yellow"
                            )
                        )
                    else:
                        tqdm.write(
                            click.style(f"  ✔ Completed: {out_path}", fg="green")
                        )
                except Exception as e:
                    tqdm.write(
                        click.style(
                            f"  ❌ Error processing chapter {chapter['title']}: {e}",
                            fg="red",
                        )
                    )
                finally:
                    pbar.update(1)

    return skipped_count


@main.command()
@click.argument("input_files", nargs=-1, required=True)
@click.option("--voice", default="af_heart", help="Voice ID")
@click.option("--speed", default=1.0, type=float, help="Speech speed")
@click.option("--lang", default="a", help="Language code")
@click.option("--stream", is_flag=True, help="Stream audio in real-time")
@click.option(
    "--split-output", type=click.Path(), help="Directory to save chapter files"
)
@click.option(
    "--abstract-only",
    is_flag=True,
    help="If you only want to convert the abstract of a paper (PDF input)",
)
@click.option(
    "--engine",
    default="kokoro",
    type=click.Choice(["kokoro", "edge"]),
    help="TTS engine to use (kokoro or edge)",
)
def convert(input_files, voice, speed, lang, stream, split_output, abstract_only, engine):
    """Convert one or more text, EPUB, or PDF files to audio.

    Pass multiple INPUT_FILES to queue them for sequential processing.
    Each file's chapters are processed in parallel; files are queued one after another.

    Examples:

      tts-studio convert book.epub --split-output ./out

      tts-studio convert paper1.pdf paper2.pdf paper3.pdf --split-output ./out --abstract-only

      tts-studio convert text.txt --engine edge
    """

    # --- Language mapping ---
    lang_map = {"en": "a", "en-us": "a", "en-gb": "b"}
    final_lang = lang_map.get(lang.lower(), lang)

    base_output = split_output or "."
    os.makedirs(base_output, exist_ok=True)

    total_files = len(input_files)
    total_skipped = 0

    # --- Queue progress bar across all files ---
    click.secho(
        f"🗂  Queued {total_files} file(s) for conversion.\n",
        fg="cyan",
    )

    with tqdm(
        total=total_files, desc="Queue progress", ncols=80, position=0
    ) as queue_bar:
        for idx, input_file in enumerate(input_files, start=1):
            if not Path(input_file).exists() and not input_file.endswith(
                (".epub", ".pdf")
            ):
                # Treat as raw text content (original behaviour)
                pass
            elif not Path(input_file).exists():
                tqdm.write(
                    click.style(f"⚠  File not found, skipping: {input_file}", fg="red")
                )
                queue_bar.update(1)
                continue

            skipped = process_single_file(
                input_file,
                voice,
                speed,
                final_lang,
                base_output,
                abstract_only,
                idx,
                total_files,
                engine,
            )
            total_skipped += skipped
            queue_bar.update(1)

    click.secho(
        f"\n🏁 All {total_files} file(s) processed! ({total_skipped} chapter(s) skipped)",
        fg="cyan",
    )


if __name__ == "__main__":
    main()
