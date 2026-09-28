import itertools
import re
import sys
import threading
import time

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
