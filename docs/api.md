# API Reference

TTS-Studio is a plain Python package — every part of the CLI is usable as a library. The public surface is small and stable:

| Module | Provides |
|---|---|
| `tts_studio.cli` | Click command group, file/chapter orchestration |
| `tts_studio.parsers` | `EpubParser`, `PdfParser` — input extraction |
| `tts_studio.processor` | The three engine processors + `transcribe_sample` |
| `tts_studio.utils` | Text normalization/splitting, `ChunkCache` |
| `tts_studio.breeze_fast` | `FastDepth` — MLX depth-decoder optimization for Breeze |

Full auto-generated docs (from docstrings) are linked at the bottom of this page. The engine processors are only importable when the matching extra is installed (`uv sync --extra kokoro` / `--extra breeze`); `cli`, `parsers`, and `utils` work with the core install.

## `tts_studio.cli`

### `main()`

Click group exposed as the `tts-studio` console script. Subcommand: `convert`.

### `convert` command

```text
tts-studio convert INPUT_FILES... [OPTIONS]
```

Accepts one or more input files (`.txt`, `.epub`, `.pdf`, or `-` for stdin). Key options: `--engine {kokoro,edge,breeze}`, `--voice`, `--speed`, `--lang`, `--stream`, `--split-output DIR`, `--abstract-only`, plus the breeze-only options (`--instruction`, `--cfg-scale`, `--ref-audio`/`--ref-text`, `--breeze-model`, `--seed`, `--breeze-workers`, `--breeze-depth-mode`).

### `load_chapters(input_file)`

Load a single input file into chapter dicts. Dispatches by extension: `.epub` → `EpubParser.extract_chapters`, `.pdf` → `PdfParser.get_chapters`, anything else is read as text (a non-existent path is treated as raw text content). Returns a list of `{"title": str, "content": str, "order": int}`.

### `process_chapter(chapter, voice, speed, lang, split_output, abstract_only, engine="kokoro", breeze_processor=None)`

Convert one chapter dict to `NN_Safe_Title.wav` (`.mp3` for edge) inside `split_output`. Returns `(out_file, status)` where status is `"completed"` or `"skipped"` (output exists, or `abstract_only` set and the title has no "abstract"). With `--abstract-only`, chapters whose titles don't contain "abstract" are skipped.

### `process_single_file(input_file, ...)`

Processes all chapters of one file: parallel `ProcessPoolExecutor` (up to 4 workers) for kokoro/edge; serial with one shared `breeze_processor` for breeze. Returns the number of skipped chapters.

## `tts_studio.parsers`

### `EpubParser.extract_chapters(epub_file)`

Static method. Reads an EPUB in spine order and extracts chapters with layered fallback parsing: `<p>` tags → block-level containers → newline splitting. Words split across inline tags stay joined (`well-<span>known</span>` → `well-known`). If a single large chapter is found, it retries splitting by headings (`h1`–`h3`). Returns chapter dicts with `title`, `content`, `sentences`, `order`.

### `PdfParser(pdf_path).get_chapters()`

Layout-aware extraction via `pymupdf4llm` (page by page, falling back to plain text for malformed pages), split into chapters on markdown headers, with normalized paragraphs and sentences. Falls back to raw page text if section splitting yields nothing. Returns chapter dicts with `title` (truncated to 100 chars), `content`, `paragraphs`, `sentences`, `order`.

## `tts_studio.processor`

### `TTSProcessor(lang_code="a", sentence_pause=0.25, paragraph_pause=0.6)`

Kokoro-82M wrapper with device management (MPS → CUDA → CPU, plus MPS memory fraction via `TORCH_MPS_MEMORY_FRACTION`) and cache clearing after each chunk.

- `generate_audio(text, voice="af_heart", speed=1.0)` — generator yielding `(gs, ps, audio)` from the Kokoro pipeline.
- `stream_generator(text, voice="af_heart", speed=1.0, cache=None)` — generator yielding `(sentence, audio)`, stitching silence between sentences/paragraphs. With a `ChunkCache`, finished sentences are stored and reused on resume.
- `save(text, output_path, voice="af_heart", speed=1.0, chunk_size=None)` — writes 24 kHz mono WAV via a `.part` atomic rename; the sentence cache lives under `<output>.chunks/`.

### `EdgeTTSProcessor(voice="en-US-AvaMultilingualNeural")`

- `generate_audio(text, output_path, speed=1.0)` — synthesizes to file (rate is derived from `speed`).
- `stream_generator(text, voice=None, speed=1.0)` — yields the whole text at once (edge-tts has no streaming).
- `save(text, output_path, voice=None, speed=1.0, chunk_size=None)` — writes MP3 atomically; with `chunk_size`, splits and concatenates.

### `BreezeTTSProcessor(model=None, instruction=None, ref_audio=None, ref_text=None, cfg_scale=1.0, seed=42, temperature=0.9, top_k=50, top_p=1.0, depth_mode="cached", workers=1)`

Breeze TTS 2 on Apple Silicon (MLX). `model` accepts a local checkpoint dir, an HF repo id, or the `BREEZE_TTS_MODEL` env var (default `mlx-community/Breeze-TTS-2-mlx-4bit`, ~3 GB, resolved offline-first from the HF cache). Voice modes: `instruction` only (voice design, try `cfg_scale=4`), `ref_audio` + `ref_text` (clone), or all three (clone + steer). Without a reference, the first generation creates a persisted voice anchor under `~/.cache/tts-studio/anchors/` that stabilizes the voice across chunks, chapters, and runs.

- `stream_generator(text, voice=None, speed=1.0, chunk_size=None, cache=None)` — yields `(chunk, audio)` split at sentence boundaries (~600 chars per chunk by default), with trailing-silence trimming and resumable chunks.
- `save(text, output_path, voice=None, speed=1.0, chunk_size=None)` — writes WAV atomically; with `workers > 1`, chunks are generated in parallel spawn-processes (one model copy each) and assembled in order.
- `generate_audio(text, output_path, speed=1.0)` — alias of `save` (speed is not supported; steer pace via the instruction).
- `close()` — shut down the worker pool.

### `transcribe_sample(audio_path, model="mlx-community/whisper-base-mlx")`

Transcribes a voice sample with mlx-whisper (used to auto-fill `ref_text` for cloning).

## `tts_studio.utils`

### `normalize_text(text, strip_markdown=False)`

Normalizes text for TTS: removes soft hyphens, rejoins hyphen-broken words across line breaks and markup boundaries, collapses whitespace while keeping paragraph breaks, and optionally strips markdown (links, bold, italic, code).

### `split_paragraphs(text)`

Splits into paragraphs on blank lines, folding soft line-wraps inside each paragraph.

### `split_sentences(paragraph)`

Sentence-splits with nltk punkt, then merges fragments that punkt mis-splits on abbreviations ("Mr.", "etc.") and unterminated segments so the TTS doesn't pause mid-thought.

### `chunk_text(text, initial_chunk_size=1000)`

Splits text into chunks at sentence boundaries of roughly `initial_chunk_size` characters.

### `ChunkCache(directory, namespace="")`

Disk cache of per-chunk audio (`.npy` files keyed by namespace, index, and text hash) enabling resume after a crash. Writes are atomic; I/O failures are silently ignored.

- `get(index, text)` — cached audio array or `None`.
- `put(index, text, audio)` — atomically cache audio (best-effort).
- `clear()` — remove the cache directory once the final output exists.

## `tts_studio.breeze_fast`

### `FastDepth(model, mode="cached")`

Patches the Breeze TTS 2 depth decoder's attention with intra-frame KV reuse (`FrameKVCache`), deferring host reads to the end of each frame — a ~2-4x speedup over the upstream mlx-audio implementation. `mode` is `"cached"` or `"compiled"`; call `install()` to activate on a loaded model (done automatically by `BreezeTTSProcessor`).

## Auto-generated reference

- [CLI](tts_studio/cli.md) — `tts_studio.cli`
- [Parsers](tts_studio/parsers.md) — `tts_studio.parsers`
- [Processor](tts_studio/processor.md) — `tts_studio.processor`
- [Utils](tts_studio/utils.md) — `tts_studio.utils`
- [Breeze backend](tts_studio/breeze_fast.md) — `tts_studio.breeze_fast`
