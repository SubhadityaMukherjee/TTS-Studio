import sys
import time
import itertools
import threading


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
    sentences = text.replace("\n", " ").split(".")
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
        sentence = sentence.strip() + "."
        if current_size + len(sentence) > initial_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_size = [], 0
        current_chunk.append(sentence)
        current_size += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
