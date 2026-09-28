import hashlib
import itertools
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from nltk import sent_tokenize

# Titles/abbreviations that punkt often misreads as sentence ends, causing
# the TTS to pause mid-thought ("Mr. Smith arrived" -> "Mr." | "Smith arrived")
_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|esp|approx|Fig|No|Vol|Ch|Rev|"
    r"Hon|Gen|Col|Capt|Lt|Sgt|Inc|Ltd|Co|Ph\.D|MD|DD|BS|BA|MA|MS)\.$",
    re.IGNORECASE,
)
_TERMINALS = (".", "!", "?", "…", "。", "！", "？")
_CLOSERS = ('"', "'", "”", "’", ")", "]", "»")


def _ends_sentence(sentence):
    stripped = sentence.rstrip()
    while stripped and stripped[-1] in _CLOSERS:
        stripped = stripped[:-1]
    return stripped.endswith(_TERMINALS)


def normalize_text(text, strip_markdown=False):
    """Normalize text for TTS: rejoin hyphen-broken words and clean spacing.

    Fixes the two classic causes of weird mid-word breaks:
    - EPUB words split across inline tags ("well- known" -> "well-known")
    - PDF line-wrap hyphenation ("inter-\\nesting" -> "interesting")
    """
    # Soft hyphens are invisible on screen but wreck pronunciation
    text = text.replace("\u00ad", "")

    # Rejoin words broken across a line break at their hyphen
    # ("mother-\nin-law" -> "mother-in-law"). The hyphen is kept: without a
    # dictionary we cannot distinguish typesetter hyphenation from a real
    # compound, and a spoken hyphen pause is far less jarring than
    # "motherinlaw".
    text = re.sub(r"(\w+)-[ \t]*\n[ \t\n]*(\w+)", r"\1-\2", text)

    # Rejoin words split across markup boundaries ("well- known" -> "well-known")
    text = re.sub(r"(\w)- (\w)", r"\1-\2", text)

    if strip_markdown:
        # pymupdf4llm emits markdown that the TTS would read aloud
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links
        text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)  # bold
        text = re.sub(r"(\*|_)(.+?)\1", r"\2", text)  # italic
        text = re.sub(r"`([^`]*)`", r"\1", text)  # code

    # Collapse whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def split_paragraphs(text):
    """Split text into paragraphs on blank lines, normalized for TTS."""
    text = normalize_text(text)
    paragraphs = []
    for p in re.split(r"\n{2,}", text):
        p = re.sub(r"\s+", " ", p).strip()  # fold soft line-wraps inside para
        if p:
            paragraphs.append(p)
    return paragraphs


def split_sentences(paragraph):
    """Sentence-split a paragraph, merging punkt mistakes on abbreviations
    and quotes so the TTS doesn't pause mid-thought."""
    raw = [s.strip() for s in sent_tokenize(paragraph) if s.strip()]
    sentences = []
    for sent in raw:
        if sentences:
            prev = sentences[-1]
            if (
                _ABBREV_RE.search(prev)
                or not _ends_sentence(prev)
                or (sent[0].islower() and not _ends_sentence(prev))
            ):
                sentences[-1] = prev + " " + sent
                continue
        sentences.append(sent)
    return sentences


def spinning_wheel(message="Processing...", progress=None, stop_event=None):
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not stop_event.is_set():
        spin = next(spinner)
        msg = f"\r{message} {progress if progress else ''} {spin}"
        sys.stdout.write(msg)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 60) + "\r")
    sys.stdout.flush()


def dynamic_print(text, audio_length_sec):
    """Prints text character-by-character synced to audio duration."""
    if not text:
        return
    # Calculate delay per character
    delay = audio_length_sec / len(text)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(" ")  # Space between chunks
    sys.stdout.flush()


def chunk_text(text, initial_chunk_size=1000):
    """Split text into chunks at sentence boundaries."""
    sentences = split_sentences(normalize_text(text.replace("\n", " ")))
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        if current_size + len(sentence) > initial_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_size = [], 0
        current_chunk.append(sentence)
        current_size += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


class ChunkCache:
    """Disk cache of per-chunk audio enabling resume after a crash or quit.

    Chunks are stored as ``.npy`` files keyed by (namespace, index, text
    hash) so that cached audio is only reused for the identical chunk of
    the identical text, voice, and generation settings. Writes are atomic
    (temp file + ``os.replace``) so a killed process never leaves a
    half-written chunk that looks complete. The cache is best-effort: any
    I/O failure is silently ignored, and :meth:`clear` removes it once the
    final output has been assembled.
    """

    def __init__(self, directory, namespace=""):
        self.dir = Path(directory)
        self.namespace = namespace
        self._ns = hashlib.sha1(namespace.encode("utf-8")).hexdigest()[:12]

    def _path(self, index, text):
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return self.dir / f"{index:05d}_{digest}_{self._ns}.npy"

    def get(self, index, text):
        """Return cached audio for a chunk, or None if not cached."""
        import numpy as np

        path = self._path(index, text)
        if not path.is_file():
            return None
        try:
            audio = np.load(path)
            if audio.ndim == 1 and audio.size:
                return audio
        except Exception:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        return None

    def put(self, index, text, audio):
        """Atomically cache audio for a chunk (best-effort)."""
        import numpy as np

        path = self._path(index, text)
        tmp = path.with_name(path.name + ".tmp")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                np.save(f, np.asarray(audio, dtype=np.float32))
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass

    def clear(self):
        """Remove the cache directory (call after the output is complete)."""
        shutil.rmtree(self.dir, ignore_errors=True)
