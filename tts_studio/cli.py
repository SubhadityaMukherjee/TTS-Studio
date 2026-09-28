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
from .processor import BreezeTTSProcessor, EdgeTTSProcessor, TTSProcessor
from .utils import ChunkCache


@click.group()
def main():
    """Kokoro TTS: A high-quality text-to-speech CLI."""
    pass


def process_chapter(
    chapter,
    voice,
    speed,
    lang,
    split_output,
    abstract_only,
    engine="kokoro",
    breeze_processor=None,
):
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
    out_file = os.path.join(
        split_output, f"{chapter['order']:02d}_{safe_title}{out_ext}"
    )

    # --- Skip if file already exists ---
    if os.path.exists(out_file):
        return out_file, "skipped"

    if engine == "edge":
        processor = EdgeTTSProcessor(voice=voice)
        processor.save(chapter["content"], out_file, voice=voice, speed=speed)
    elif engine == "breeze":
        # The 3B MLX runtime is too heavy to load per chapter and per worker
        # process; callers pass a shared processor and run chapters serially.
        # save() itself resumes: finished chunks are cached under
        # <out_file>.chunks/ and the final write is atomic.
        processor = breeze_processor or BreezeTTSProcessor()
        processor.save(chapter["content"], out_file, speed=speed)
    else:
        processor = TTSProcessor(lang_code=lang)

        # Process with progress-aware streaming; cache finished sentences
        # so a crashed run resumes instead of restarting the chapter.
        cache = ChunkCache(
            out_file + ".chunks",
            namespace=f"kokoro|{lang}|{voice}|{speed}|24000",
        )
        generator = processor.stream_generator(
            chapter["content"], voice=voice, speed=speed, cache=cache
        )
        part_file = out_file + ".part"
        with sf.SoundFile(
            part_file, "w", samplerate=24000, channels=1, format="WAV", subtype="PCM_16"
        ) as f:
            for _, audio in generator:
                f.write(audio)
        os.replace(part_file, out_file)
        cache.clear()

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


def _log_chapter_result(result, title):
    """Log a chapter conversion result; returns True if it was skipped."""
    out_path, status = result
    if status == "skipped":
        tqdm.write(click.style(f"  ⏩ Skipped (exists): {out_path}", fg="yellow"))
        return True
    tqdm.write(click.style(f"  ✔ Completed: {out_path}", fg="green"))
    return False


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
    breeze_opts=None,
    breeze_processor=None,
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

    chapters = [ch for ch in chapters if ch["content"].strip()]
    max_workers = min(4, multiprocessing.cpu_count() // 2)

    click.secho(
        f"\n📄 [{file_index}/{total_files}] Processing: {input_file} "
        f"({len(chapters)} chapters, max {max_workers} workers)",
        fg="cyan",
    )

    skipped_count = 0

    with tqdm(total=len(chapters), desc=f"  Chapters", ncols=80, leave=True) as pbar:
        if engine == "breeze":
            # One shared runtime (3B params) across all chapters, serially;
            # if the caller didn't provide one, load it now (per file).
            processor = breeze_processor or BreezeTTSProcessor(**(breeze_opts or {}))
            for ch in chapters:
                try:
                    result = process_chapter(
                        ch,
                        voice,
                        speed,
                        final_lang,
                        file_out_dir,
                        abstract_only,
                        engine,
                        breeze_processor=processor,
                    )
                    if _log_chapter_result(result, ch["title"]):
                        skipped_count += 1
                except Exception as e:
                    tqdm.write(
                        click.style(
                            f"  ❌ Error processing chapter {ch['title']}: {e}",
                            fg="red",
                        )
                    )
                finally:
                    pbar.update(1)
        else:
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
                }

                for future in as_completed(futures):
                    chapter = futures[future]
                    try:
                        result = future.result()
                        if _log_chapter_result(result, chapter["title"]):
                            skipped_count += 1
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
    type=click.Choice(["kokoro", "edge", "breeze"]),
    help="TTS engine to use (kokoro, edge, or breeze)",
)
@click.option(
    "--instruction",
    help="Breeze only: natural-language voice description (e.g. 'A warm, "
    "thoughtful narrator'), OR a path to a sample audio file to clone that "
    "voice",
)
@click.option(
    "--cfg-scale",
    default=1.0,
    type=float,
    help="Breeze only: CFG guidance scale for text instructions (try 4)",
)
@click.option(
    "--ref-audio",
    type=click.Path(exists=True),
    help="Breeze only: path to clean reference audio for voice cloning "
    "(alternative to passing a sample path to --instruction)",
)
@click.option(
    "--ref-text",
    help="Breeze only: exact transcript of the reference audio "
    "(auto-transcribed with whisper when omitted)",
)
@click.option(
    "--breeze-model",
    help="Breeze only: local checkpoint directory or Hugging Face repo id "
    "(default: BREEZE_TTS_MODEL env var or mlx-community/Breeze-TTS-2-mlx-4bit)",
)
@click.option("--seed", default=42, type=int, help="Breeze only: sampling seed")
@click.option(
    "--breeze-workers",
    default=2,
    type=click.IntRange(1, 8),
    help="Breeze only: parallel chunk-generation processes; each loads its "
    "own model copy (~3 GB RAM each). 1 = single process",
)
@click.option(
    "--breeze-depth-mode",
    default="cached",
    type=click.Choice(["cached", "compiled"]),
    help="Breeze only: depth-decoder mode (compiled may be faster, with a "
    "one-time compile warm-up)",
)
def convert(
    input_files,
    voice,
    speed,
    lang,
    stream,
    split_output,
    abstract_only,
    engine,
    instruction,
    cfg_scale,
    ref_audio,
    ref_text,
    breeze_model,
    seed,
    breeze_workers,
    breeze_depth_mode,
):
    """Convert one or more text, EPUB, or PDF files to audio.

    Pass multiple INPUT_FILES to queue them for sequential processing.
    Each file's chapters are processed in parallel; files are queued one after another.

    Examples:

      tts-studio convert book.epub --split-output ./out

      tts-studio convert paper1.pdf paper2.pdf paper3.pdf --split-output ./out --abstract-only

      tts-studio convert text.txt --engine edge

      tts-studio convert text.txt --engine breeze --instruction "A warm, thoughtful young woman with a calm delivery" --cfg-scale 4

      tts-studio convert text.txt --engine breeze --instruction sample.wav
    """
    breeze_opts = None
    if engine == "breeze":
        # --instruction is polymorphic: a text voice description, or a path
        # to a sample audio file to clone.
        sample = Path(instruction).expanduser() if instruction else None
        if sample and sample.is_file():
            if ref_audio:
                raise click.BadParameter(
                    "pass a sample via --instruction or --ref-audio, not both"
                )
            ref_audio = str(sample)
            instruction = None
            if not ref_text:
                click.secho("🎙  Transcribing voice sample with whisper...", fg="cyan")
                from .processor import transcribe_sample

                try:
                    ref_text = transcribe_sample(ref_audio)
                except Exception as e:
                    raise click.ClickException(
                        f"could not transcribe {ref_audio} ({e}); "
                        "pass --ref-text with its exact transcript instead"
                    )
                click.secho(f"   Transcript: {ref_text!r}", fg="cyan")
            duration = sf.info(ref_audio).duration
            if duration > 30:
                click.secho(
                    f"⚠  Voice sample is {duration:.0f}s long; cloning works "
                    "best with 5-15s of clean speech",
                    fg="yellow",
                )
        if (ref_audio is None) != (ref_text is None):
            raise click.BadParameter(
                "--ref-audio and --ref-text must be provided together"
            )
        breeze_opts = dict(
            model=breeze_model,
            instruction=instruction,
            ref_audio=ref_audio,
            ref_text=ref_text,
            cfg_scale=cfg_scale,
            seed=seed,
            depth_mode=breeze_depth_mode,
            workers=breeze_workers,
        )
        # Load the runtime once for the whole queue so the auto-created
        # voice anchor carries across every file and chapter.
        breeze_processor = BreezeTTSProcessor(**breeze_opts)
    else:
        breeze_processor = None

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
                breeze_opts=breeze_opts,
                breeze_processor=breeze_processor,
            )
            total_skipped += skipped
            queue_bar.update(1)

    click.secho(
        f"\n🏁 All {total_files} file(s) processed! ({total_skipped} chapter(s) skipped)",
        fg="cyan",
    )


if __name__ == "__main__":
    main()
