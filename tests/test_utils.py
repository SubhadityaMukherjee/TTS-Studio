import numpy as np
import pytest

from tts_studio.utils import (
    ChunkCache,
    chunk_text,
    normalize_text,
    split_paragraphs,
    split_sentences,
)


class TestNormalizeText:
    def test_soft_hyphens_removed(self):
        assert normalize_text("inter\u00adesting") == "interesting"

    def test_hyphen_linebreak_rejoined(self):
        assert normalize_text("mother-\nin-law") == "mother-in-law"

    def test_split_across_markup_rejoined(self):
        assert normalize_text("well- known") == "well-known"

    def test_whitespace_collapsed(self):
        assert normalize_text("a  b \t c") == "a b c"

    def test_paragraph_breaks_preserved(self):
        assert normalize_text("first para.\n\nsecond para.") == (
            "first para.\n\nsecond para."
        )

    def test_markdown_stripped(self):
        text = "[a link](http://x.com) and **bold** and *italic* and `code`"
        assert normalize_text(text, strip_markdown=True) == (
            "a link and bold and italic and code"
        )

    def test_markdown_kept_by_default(self):
        text = "**bold** text"
        assert normalize_text(text) == "**bold** text"


class TestSplitParagraphs:
    def test_splits_on_blank_lines(self):
        assert split_paragraphs("One.\n\nTwo.") == ["One.", "Two."]

    def test_folds_internal_line_wraps(self):
        assert split_paragraphs("One two\nthree.") == ["One two three."]

    def test_drops_empty_paragraphs(self):
        assert split_paragraphs("One.\n\n \n\nTwo.") == ["One.", "Two."]


class TestSplitSentences:
    def test_basic(self):
        assert split_sentences("First one. Second one? Third one!") == [
            "First one.",
            "Second one?",
            "Third one!",
        ]

    def test_abbreviation_not_split(self):
        sentences = split_sentences("Bring food, etc. and drinks.")
        assert sentences == ["Bring food, etc. and drinks."]

    def test_quote_closers_terminate(self):
        sentences = split_sentences('He said "Stop." Then he left.')
        assert len(sentences) == 2
        assert sentences[0] == 'He said "Stop."'


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("Hello world.") == ["Hello world."]

    def test_chunks_respect_size(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        chunks = chunk_text(text, initial_chunk_size=100)
        assert len(chunks) > 1
        # a chunk may overshoot by one sentence, but never wildly
        assert all(len(c) < 100 + 40 for c in chunks)
        # no text lost
        assert " ".join(chunks).replace("  ", " ") == " ".join(
            "Sentence number {} is here.".format(i) for i in range(50)
        )


class TestChunkCache:
    def test_miss_returns_none(self, tmp_path):
        cache = ChunkCache(tmp_path / "chunks", namespace="ns")
        assert cache.get(0, "some text") is None

    def test_put_get_roundtrip(self, tmp_path):
        cache = ChunkCache(tmp_path / "chunks", namespace="ns")
        audio = np.linspace(-1.0, 1.0, 240, dtype=np.float32)
        cache.put(0, "some text", audio)
        loaded = cache.get(0, "some text")
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, audio)

    def test_different_text_misses(self, tmp_path):
        cache = ChunkCache(tmp_path / "chunks", namespace="ns")
        cache.put(0, "some text", np.zeros(8, dtype=np.float32))
        assert cache.get(0, "other text") is None

    def test_different_namespace_misses(self, tmp_path):
        a = ChunkCache(tmp_path / "chunks", namespace="voiceA")
        b = ChunkCache(tmp_path / "chunks", namespace="voiceB")
        a.put(0, "some text", np.zeros(8, dtype=np.float32))
        assert b.get(0, "some text") is None

    def test_corrupt_file_is_evicted(self, tmp_path):
        cache = ChunkCache(tmp_path / "chunks", namespace="ns")
        cache.put(0, "some text", np.zeros(8, dtype=np.float32))
        path = cache._path(0, "some text")
        path.write_bytes(b"not an npy file")
        assert cache.get(0, "some text") is None
        assert not path.exists()

    def test_clear_removes_directory(self, tmp_path):
        cache = ChunkCache(tmp_path / "chunks", namespace="ns")
        cache.put(0, "some text", np.zeros(8, dtype=np.float32))
        assert cache.dir.exists()
        cache.clear()
        assert not cache.dir.exists()


def test_empty_cache_clear_is_safe(tmp_path):
    cache = ChunkCache(tmp_path / "missing", namespace="ns")
    cache.clear()
    assert not cache.dir.exists()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello world.", ["Hello world."]),
        ("", []),
    ],
)
def test_split_sentences_parametrized(text, expected):
    assert split_sentences(text) == expected
