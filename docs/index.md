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

- `INPUT_FILE`: Path to input file (text, EPUB, PDF) or `-` for stdin
- `OUTPUT_FILE`: (Optional) Output audio file path

**Options:**

- `--voice`: Voice ID to use (default: `af_heart`)
- `--speed`: Speech speed multiplier (default: `1.0`)
- `--lang`: Language code (`a` = en/en-us, `b` = en-gb, default: `a`)
- `--stream`: Enable real-time audio streaming
- `--split-output`: Directory to save individual chapter files instead of one file

### Examples

Convert with custom voice and speed:

```bash
uv run tts-studio convert input.txt --voice af_heart --speed 1.2
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
- **Parallel processing**: Uses up to 4 workers (adjusted based on CPU count)
- **File skipping**: Existing output files are automatically skipped
- **Audio format**: Generated audio is saved at 24kHz, mono WAV format
