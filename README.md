# TTS-Studio

A high-quality text-to-speech CLI using the Kokoro API. This is HEAVILY inspired by [nazdridoy/kokoro-tts](https://github.com/nazdridoy/kokoro-tts) but my vision for it differed too greatly for me to fork it.

## Features

- **Multi-format support**: Convert text, EPUB, and PDF files to audio
- **Multiple voices**: Support for various voice options (af_heart, etc.)
- **Adjustable speed**: Customize speech speed with configurable options
- **Multi-language support**: English variants (en, en-us, en-gb) and language codes
- **Real-time streaming**: Stream audio output in real-time
- **Parallel processing**: Multi-worker chapter processing for fast conversion
- **Progress tracking**: Real-time progress bars and status updates
- **Smart file handling**: Automatic chapter extraction and file skipping for existing outputs
- **Safe naming**: Automatic filename sanitization for chapter outputs

## Installation

```bash
uv sync
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

### Command Options

```bash
uv run tts-studio convert INPUT_FILE [OUTPUT_FILE] [OPTIONS]
```

**Arguments:**

- `INPUT_FILE`: Path to input file(s) (text, EPUB, PDF) or `-` for stdin. You can pass in multiple files

**Options:**

- `--engine`: TTS engine: `kokoro`, `edge`, or `breeze` (default: `kokoro`)
- `--voice`: Voice ID to use (default: `af_heart`; kokoro/edge only)
- `--speed`: Speech speed multiplier (default: `1.0`)
- `--lang`: Language code (`a` = en/en-us, `b` = en-gb, default: `a`)
- `--stream`: Enable real-time audio streaming
- `--split-output`: Directory to save individual chapter files instead of one file
- `--abstract-only`: For PDF files, if you just want to have the audio files for the Abstract, set this flag
- `--instruction`: (breeze) natural-language voice description for voice design
- `--cfg-scale`: (breeze) CFG guidance scale for `--instruction` (try 4)
- `--ref-audio` / `--ref-text`: (breeze) reference audio + its exact transcript for voice cloning
- `--breeze-model`: (breeze) local checkpoint dir or HF repo id (default: `BREEZE_TTS_MODEL` env var or `rishikksh20/Breeze-TTS-2-mlx`)
- `--seed`: (breeze) sampling seed (default: 42)

### Examples

Convert with custom voice and speed:

```bash
uv run tts-studio convert input.txt --voice af_heart --speed 1.2
```

Convert with [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) on Apple Silicon (voice design, no reference audio needed):

```bash
uv run tts-studio convert input.txt --engine breeze --instruction "A warm, thoughtful young woman with a calm, reflective delivery" --cfg-scale 4
```

Clone a voice from reference audio:

```bash
uv run tts-studio convert input.txt --engine breeze --ref-audio reference.wav --ref-text "Exact transcript of the reference audio."
```

Convet multiple research papers but just save their abstracts to a folder

```bash
uv run tts-studio convert /Users/smukherjee/Downloads/2203.02395v1.pdf /Users/smukherjee/Downloads/2025.emnlp-main.794.pdf /Users/smukherjee/Downloads/2025.ijcnlp-demo.3.pdf --split-output ~/Downloads/books/papers/ --abstract-only --voice af_heart
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
- **File skipping**: Existing output files are automatically skipped
- **Audio format**: Generated audio is saved at 24kHz, mono WAV (edge outputs MP3)

## Breeze TTS 2 notes

- Requires an Apple-Silicon Mac and [SoX](https://sox.sourceforge.io/) (`brew install sox`); the official upstream code is CUDA-only
- The INT8 MLX checkpoint (~3.7 GB) is downloaded automatically on first use
- **Voice auto-anchoring**: without `--ref-audio`, the first generation creates a short anchor utterance that is then cloned for every subsequent chunk, chapter, and file in the run — giving one stable voice. Use `--instruction` (+ `--cfg-scale 4`) to design the anchor's voice, or `--ref-audio`/`--ref-text` to anchor to a real recording
- Supports English and Chinese; inline vocal events like `(laugh)`, `(sigh)`, `(cough)` can appear in the text
- Model weights and self-hosted outputs are licensed for research and non-commercial use only (see the [BreezeBlue license](https://huggingface.co/BreezeBlue/Breeze-TTS-2))
