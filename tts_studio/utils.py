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
    """Normalize text for TTS: rejoin hyphen-broken words, drop URLs, and
    clean spacing.

    Fixes the three classic causes of weird mid-word breaks:
    - EPUB words split across inline tags ("well- known" -> "well-known")
    - PDF line-wrap hyphenation ("inter-\\nesting" -> "interesting")
    - URLs, which no listener wants read aloud
    """
    if strip_markdown:
        # pymupdf4llm emits markdown that the TTS would read aloud
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links
        text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)  # bold
        text = re.sub(r"(\*|_)(.+?)\1", r"\2", text)  # italic
        text = re.sub(r"`([^`]*)`", r"\1", text)  # code

    # URLs are never worth narrating ("available at https://x.co/y, 2024");
    # runs after the markdown strip so "[text](http://url)" keeps its text
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)

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


# --- speakable-text normalization (rule-based, applied before synthesis) ---#

_ABBREV_EXPANSIONS = {
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "etc.": "et cetera",
    "vs.": "versus",
    "e.g.": "for example",
    "i.e.": "that is",
}

_ROMAN_KEYS = (
    "chapter",
    "section",
    "part",
    "book",
    "volume",
    "act",
    "scene",
)
_ROMAN_RE = re.compile(
    r"\b((" + "|".join(_ROMAN_KEYS) + r")\s+)([IVXLCDM]+)\b", re.IGNORECASE
)
_ROMAN_VALUES = {
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
}
_ROMAN_INVALID = re.compile(r"(IIII|VV|XXXX|LL|CCCC|DD)")


def _roman_to_int(text):
    """Value of a well-formed roman numeral (1-3999), else None."""
    s = text.upper()
    if _ROMAN_INVALID.search(s):
        return None
    total = 0
    values = [_ROMAN_VALUES[c] for c in s]
    for current, nxt in zip(values, values[1:]):
        total += -current if current < nxt else current
    if values:
        total += values[-1]
    # reject non-canonical forms like "IL" by round-tripping
    remainder = total
    canonical = ""
    for numeral, value in (
        ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100),
        ("XC", 90), ("L", 50), ("XL", 40), ("X", 10), ("IX", 9),
        ("V", 5), ("IV", 4), ("I", 1),
    ):
        while remainder >= value:
            canonical += numeral
            remainder -= value
    return total if canonical == s else None


def speakable_text(text):
    """Rewrite text into its spoken form: expand abbreviations, number
    ranges, chapter numerals, and symbols that TTS models read literally
    or with weird stress. Conservative on purpose - only unambiguous,
    high-confidence rules."""
    for abbrev, spoken in _ABBREV_EXPANSIONS.items():
        # keep any trailing punctuation (comma/semicolon/colon) - only the
        # abbreviation itself is replaced
        pattern = r"\b" + re.escape(abbrev) + r"(?=[\s,;:\"'(\[]|$)"
        text = re.sub(pattern, spoken, text)
    # "&" is read "and" ("R&D" -> "R and D", "A & B" -> "A and B")
    text = re.sub(r"\s*&\s*", " and ", text)
    # unicode ellipsis -> the ASCII form models know
    text = text.replace("\u2026", "...")
    # number ranges: "1968-1972" / "2001-2003" -> "1968 to 1972"
    text = re.sub(r"(\d)\s*[\u2013\u2014]\s*(\d)", r"\1 to \2", text)
    # "Chapter IV" / "PART II" -> spoken numbers (only after a heading word)
    def _expand_roman(match):
        value = _roman_to_int(match.group(3))
        if value is None or value > 100:
            return match.group(0)
        return f"{match.group(1)}{value}"

    text = _ROMAN_RE.sub(_expand_roman, text)
    return text


# Strong clause boundaries worth a real breathing point when a sentence
# runs very long: semicolons and em/en dashes, spaced or not (NOT commas,
# colons, or the ASCII hyphen).
_CLAUSE_SEP_RE = re.compile(r"\s*;\s*|\s*[\u2013\u2014]\s*")


def split_long_sentence(sentence, max_chars=250):
    """Break an over-long sentence at strong clause boundaries (semicolons,
    em-dashes) so each generation unit ends at a natural breathing point.

    Pieces are re-joined with their original punctuation while they still
    fit; only pieces that must stand alone get a period appended. Sentences
    without usable boundaries are returned untouched.
    """
    if len(sentence) <= max_chars:
        return [sentence]
    pieces = [p.strip() for p in _CLAUSE_SEP_RE.split(sentence) if p.strip()]
    if len(pieces) < 2:
        return [sentence]
    seps = [s.strip() for s in _CLAUSE_SEP_RE.findall(sentence)]
    units, buf = [], pieces[0]
    for piece, sep in zip(pieces[1:], seps):
        if len(buf) + len(sep) + 1 + len(piece) <= max_chars:
            buf = f"{buf}{sep} {piece}"
        else:
            if not buf.endswith((";", ",", ":", ".", "!", "?")):
                buf += "."
            units.append(buf)
            buf = piece
    if not buf.endswith((".", "!", "?")):
        buf += "."
    units.append(buf)
    return units


def normalize_pauses(text):
    """Replace remaining prose dashes with commas so the model pauses
    naturally instead of stumbling on an em-dash. Run AFTER
    split_long_sentence, which uses dashes as split points."""
    return re.sub(r"\s*[\u2013\u2014]+\s*", ", ", text)


def chunk_paragraphs(text, max_chars=1000, max_sentence_chars=250):
    """Split text into TTS-sized chunks that respect paragraph structure.

    Returns a list of ``{"text": ..., "ends_paragraph": bool}`` dicts.
    Unlike :func:`chunk_text`, sentences are never packed across paragraph
    boundaries, and each chunk carries a flag marking the final chunk of a
    paragraph so the caller can insert a real pause between paragraphs.
    """
    text = normalize_text(text)
    chunks = []
    for para in re.split(r"\n{2,}", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        para = speakable_text(para)
        units = []
        for sent in split_sentences(para):
            units.extend(split_long_sentence(sent, max_chars=max_sentence_chars))
        units = [normalize_pauses(u) for u in units if u.strip()]
        current, size = [], 0
        for unit in units:
            if current and size + 1 + len(unit) > max_chars:
                chunks.append({"text": " ".join(current), "ends_paragraph": False})
                current, size = [], 0
            current.append(unit)
            size = size + (1 if len(current) > 1 else 0) + len(unit)
        if current:
            chunks.append({"text": " ".join(current), "ends_paragraph": True})
    return chunks


# Front/back matter that no audiobook listener wants narrated. Matched
# against a normalized chapter title (lowercase, punctuation as spaces),
# anchored at the START so a chapter genuinely titled "The Cover Up" or
# "Introduction to Statistics" is kept.
_SKIP_TITLE_RE = re.compile(
    r"^(?:\d+\s*)?"
    r"(cover|logo|adcard\w*|advert\w*|halftitle\w*|titlepage\w*|title ?page\w*|"
    r"title\d*|toc|table of contents|contents|colophon|copyright|index|"
    r"appendi\w*)\b"
)


def is_skippable_chapter(title):
    """True for front/back-matter chapters that should not be narrated:
    cover pages, logos, ad cards, (half)title pages, the table of
    contents, colophon, copyright page, indexes, and appendixes."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()
    return bool(_SKIP_TITLE_RE.match(normalized))


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
