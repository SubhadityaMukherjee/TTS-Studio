# TTS-Studio

!!! warning "Copyright notice — read this before using TTS-Studio"

    This tool converts text (including copyrighted books, articles, and papers) into audio **for your own personal use only**. The author of TTS-Studio holds **no responsibility** for how it is used, and specifically for its use on copyrighted works.

    You are **not legally allowed to distribute** any audio you generate with TTS-Studio unless you hold the rights or a license to the underlying text (and, where applicable, to the cloned voice). This includes uploading, sharing, selling, or publishing generated audio. Respect the licenses of the works you convert, and the licenses of the models you run (e.g. Breeze TTS 2 is research / non-commercial only).

A modular, multi-engine text-to-speech CLI. Turn text, EPUB, and PDF files (or stdin) into audiobooks using one of three interchangeable engines:

| Engine | Backend | Best for | Requirements |
|---|---|---|---|
| `kokoro` (default) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | High-quality local synthesis, real-time streaming | Any machine (MPS/CUDA/CPU) |
| `edge` | [edge-tts](https://github.com/rany2/edge-tts) | Fast, lightweight, many voices/languages — no GPU or model download | Internet connection |
| `breeze` | [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) | Voice design from a text description, voice cloning from a sample | Apple-Silicon Mac |

This is HEAVILY inspired by [nazdridoy/kokoro-tts](https://github.com/nazdridoy/kokoro-tts), but my vision for it differed too greatly for me to fork it — hence a standalone, engine-agnostic tool.

## Features

- **Three engines, one CLI**: switch between kokoro, edge, and breeze with `--engine`
- **Multi-format support**: Convert text, EPUB, and PDF files to audio (multiple files per run)
- **Multiple voices**: Voice IDs per engine (`af_heart`, etc.)
- **Voice design & cloning** (breeze): describe a voice in natural language, or clone one from a 5–15s sample
- **Adjustable speed**: Customize speech speed with configurable options
- **Multi-language support**: English variants (en, en-us, en-gb) and language codes
- **Real-time streaming**: Stream audio output in real-time
- **Parallel processing**: Multi-worker chapter processing for fast conversion
- **Resumable runs**: Finished chunks are cached to disk; crashed or interrupted conversions pick up where they left off
- **Progress tracking**: Real-time progress bars and status updates
- **Smart file handling**: Automatic chapter extraction, output skipping for existing files, and abstract-only extraction for papers
- **Safe naming**: Automatic filename sanitization for chapter outputs

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# core CLI + edge engine
uv sync

# ...with the kokoro engine (torch, kokoro, spacy model)
uv sync --extra kokoro

# ...with the breeze engine (mlx, mlx-audio, mlx-whisper — Apple Silicon only)
uv sync --extra breeze

# everything
uv sync --all-extras
```

## Usage

### Basic Usage

Convert text to audio:

```bash
uv run tts-studio convert input.txt
```

Convert EPUB file:

```bash
uv run tts-studio convert book.epub
```

Convert PDF file:

```bash
uv run tts-studio convert document.pdf
```

Stream from stdin:

```bash
echo "Hello world" | tts-studio convert -
```

Use a different engine:

```bash
uv run tts-studio convert input.txt --engine edge
```

### Command Options

```bash
uv run tts-studio convert INPUT_FILE... [OUTPUT_FILE] [OPTIONS]
```

**Arguments:**

- `INPUT_FILE`: Path to input file(s) (text, EPUB, PDF) or `-` for stdin. You can pass in multiple files

**Common options:**

- `--engine`: TTS engine: `kokoro`, `edge`, or `breeze` (default: `kokoro`)
- `--voice`: Voice ID to use (default: `af_heart`; kokoro/edge only)
- `--speed`: Speech speed multiplier (default: `1.0`)
- `--lang`: Language code (`a` = en/en-us, `b` = en-gb, default: `a`)
- `--stream`: Enable real-time audio streaming
- `--split-output`: Directory to save individual chapter files instead of one file
- `--abstract-only`: For PDF files, if you just want the audio files for the Abstract, set this flag

**Breeze-only options:**

- `--instruction`: natural-language voice description, **or** a path to a sample audio file whose voice gets cloned (transcript auto-transcribed)
- `--cfg-scale`: CFG guidance scale for text `--instruction` (try 4)
- `--ref-audio` / `--ref-text`: alternative way to pass reference audio + its exact transcript
- `--breeze-model`: local checkpoint dir or HF repo id (default: `BREEZE_TTS_MODEL` env var or the 4bit MLX build)
- `--seed`: sampling seed (default: 42)
- `--breeze-workers`: parallel chunk-generation processes (each loads its own model copy, ~3 GB RAM; default 2)
- `--breeze-depth-mode`: depth-decoder mode, `cached` or `compiled` (compiled may be faster after a one-time warm-up)

### Examples

Convert with custom voice and speed:

```bash
uv run tts-studio convert input.txt --voice af_heart --speed 1.2
```

Convert with [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) on Apple Silicon (voice design, no reference audio needed):

```bash
uv run tts-studio convert input.txt --engine breeze --instruction "A warm, thoughtful young woman with a calm, reflective delivery" --cfg-scale 4
```

Clone a voice from a sample recording — `--instruction` also accepts an audio file path; its transcript is auto-transcribed (or pass `--ref-text` with the exact words):

```bash
uv run tts-studio convert input.txt --engine breeze --instruction sample.wav
```

Convert multiple research papers but just save their abstracts to a folder:

```bash
uv run tts-studio convert paper1.pdf paper2.pdf paper3.pdf --split-output ~/Downloads/books/papers/ --abstract-only --voice af_heart
```

Split EPUB chapters into separate files:

```bash
uv run tts-studio convert book.epub --split-output ./audio_chapters/
```

Convert PDF with British English:

```bash
uv run tts-studio convert document.pdf --lang en-gb
```

Stream audio in real-time:

```bash
uv run tts-studio convert input.txt --stream
```

## Input Formats

- **Text files** (.txt): Plain text files are processed as a single chapter
- **EPUB files** (.epub): Automatically extracted into chapters with intelligent sentence parsing
- **PDF files** (.pdf): Converted to chapters with layout-aware text extraction
- **Stdin**: Pipe text directly using `-` as the input file

## Processing Details

- **Chapter handling**: EPUB and PDF files are automatically split into chapters
- **Progress display**: Real-time progress bar shows processing status
- **Parallel processing**: Uses up to 4 workers (adjusted based on CPU count); the breeze engine processes chapters serially with one shared runtime
- **Resumability**: kokoro and breeze cache finished audio chunks under `<output>.chunks/`; interrupted runs resume instead of restarting, and the final file write is atomic
- **File skipping**: Existing output files are automatically skipped
- **Audio format**: Generated audio is saved at 24kHz, mono WAV (edge outputs MP3)

## Breeze TTS 2 notes

- Requires an Apple-Silicon Mac; the official upstream code is CUDA-only
- Runs via [mlx-audio](https://github.com/Blaizzy/mlx-audio) with a [FastDepth](https://github.com/xzf-thu/BreezeTTS2_Mac_Streaming) depth-decoder optimization (intra-frame KV reuse, ~2-4x faster); 4bit weights (~3 GB) download automatically on first use
- **Voice auto-anchoring**: with a text `--instruction` (or none at all), the first generation creates a short anchor utterance that is then cloned for every subsequent chunk, chapter, and file in the run — giving one stable voice. Passing a sample audio skips the anchor and clones that voice directly
- Supports English and Chinese; inline vocal events like `(laugh)`, `(sigh)`, `(cough)` can appear in the text
- Model weights and self-hosted outputs are licensed for research and non-commercial use only (see the [BreezeBlue license](https://huggingface.co/BreezeBlue/Breeze-TTS-2))

## Development

```bash
uv sync --all-extras --all-groups   # everything: engines, tests, docs
uv run pytest                       # run the test suite
uv run zensical serve               # preview the docs (built with Zensical)
```

The docs are built with [Zensical](https://zensical.org) and deployed to GitHub Pages by CI; tests run on every push and pull request.
