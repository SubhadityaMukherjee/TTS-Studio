# Engines

TTS-Studio ships with three interchangeable engines. All of them share the same CLI surface, chapter splitting, resumability, and output handling — pick one with `--engine`.

| | `kokoro` | `edge` | `breeze` |
|---|---|---|---|
| Default | ✅ | | |
| Backend | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Microsoft Edge TTS | [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) |
| Hardware | MPS / CUDA / CPU | CPU only | Apple Silicon |
| Network | model download on first run | every run (online service) | model download on first run |
| Output | 24 kHz WAV | MP3 | 24 kHz WAV |
| Voice selection | `--voice` (e.g. `af_heart`) | `--voice` | `--instruction` (design), sample/ref audio (clone) |
| Extra install | `--extra kokoro` | included in core | `--extra breeze` |

## kokoro

The default engine: a small, high-quality local model that runs anywhere (Metal on Apple Silicon, CUDA, or CPU). It supports real-time streaming with `--stream` and per-chapter resumability.

```bash
uv sync --extra kokoro
uv run tts-studio convert book.epub --voice af_heart --speed 1.1
```

- `--voice`: Kokoro voice ID (default `af_heart`)
- `--lang`: `a` (en, en-us) or `b` (en-gb)

## edge

A thin wrapper around [edge-tts](https://github.com/rany2/edge-tts): fast, no model downloads, and a large catalogue of voices and languages. Handy when you don't want to pull in torch or MLX — it works with a bare `uv sync`.

```bash
uv run tts-studio convert input.txt --engine edge --voice en-US-AriaNeural
```

- `--voice`: Edge TTS voice name (default `af_heart`)
- Output is MP3 rather than WAV

## breeze

[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) on Apple Silicon, with voice **design** and **cloning**:

```bash
uv sync --extra breeze

# design a voice from a description
uv run tts-studio convert input.txt --engine breeze \
    --instruction "A warm, thoughtful young woman with a calm, reflective delivery" \
    --cfg-scale 4

# clone a voice from a 5-15s sample (transcript auto-transcribed with whisper)
uv run tts-studio convert input.txt --engine breeze --instruction sample.wav
```

Key options:

- `--instruction`: text voice description **or** a path to sample audio
- `--ref-audio` / `--ref-text`: reference audio plus its exact transcript
- `--cfg-scale`: CFG guidance scale for text instructions (try 4)
- `--breeze-model`: checkpoint dir or HF repo id (`BREEZE_TTS_MODEL` env var overrides)
- `--seed`: sampling seed (default 42)
- `--breeze-workers`: parallel chunk workers, each with its own model copy (~3 GB RAM)
- `--breeze-depth-mode`: `cached` or `compiled` depth-decoder mode

!!! note "Licensing"

    Breeze TTS 2 model weights and self-hosted outputs are licensed for research and non-commercial use only — see the [BreezeBlue license](https://huggingface.co/BreezeBlue/Breeze-TTS-2).

See [Breeze TTS 2 notes](index.md#breeze-tts-2-notes) on the home page for performance and voice-anchoring details.
