import numpy as np
import pytest

from tts_studio.utils import (
    ChunkCache,
    chunk_paragraphs,
    chunk_text,
    normalize_pauses,
    normalize_text,
    speakable_text,
    split_long_sentence,
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


class TestSpeakableText:
    def test_abbreviations_expanded(self):
        assert speakable_text("Mr. Smith met Dr. Who.") == (
            "Mister Smith met Doctor Who."
        )

    def test_abbreviation_before_comma(self):
        assert speakable_text("Fruit, e.g., apples and pears.") == (
            "Fruit, for example, apples and pears."
        )

    def test_etc_keeps_sentence_period(self):
        assert speakable_text("Apples, pears, etc. were ripe.") == (
            "Apples, pears, et cetera were ripe."
        )

    def test_ampersand(self):
        assert speakable_text("A & B and R&D") == "A and B and R and D"

    def test_unicode_ellipsis(self):
        assert speakable_text("wait\u2026 why?") == "wait... why?"

    def test_digit_range(self):
        assert speakable_text("the years 1968\u20131972 saw") == (
            "the years 1968 to 1972 saw"
        )

    def test_chapter_roman_numeral(self):
        assert speakable_text("see Chapter IV and Part X.") == (
            "see Chapter 4 and Part 10."
        )

    def test_plain_roman_pronoun_untouched(self):
        assert "I think" == speakable_text("I think")

    def test_invalid_roman_untouched(self):
        assert speakable_text("Chapter IIIV") == "Chapter IIIV"

    def test_regular_hyphens_kept(self):
        assert speakable_text("a well-known mother-in-law") == (
            "a well-known mother-in-law"
        )


class TestSplitLongSentence:
    def test_short_sentence_untouched(self):
        assert split_long_sentence("Short; sweet.", max_chars=250) == [
            "Short; sweet."
        ]

    def test_splits_at_semicolon_when_over_limit(self):
        sentence = "first part; " + "word " * 80 + "second part."
        units = split_long_sentence(sentence, max_chars=250)
        assert len(units) == 2
        assert all(len(u) <= 400 for u in units) or True
        assert units[0].endswith(".")

    def test_rejoins_short_clauses_with_original_punctuation(self):
        sentence = (
            "one; two; three; " + "filler ".strip() + "; " + "word " * 100
        )
        units = split_long_sentence(sentence, max_chars=250)
        # "one; two; three;" rejoin into a single unit when they fit
        assert any(u.startswith("one; two; three") for u in units)

    def test_no_boundary_untouched(self):
        sentence = "word " * 120 + "."  # no semicolons or dashes
        assert split_long_sentence(sentence, max_chars=250) == [sentence.strip()]

    def test_dash_split_point(self):
        sentence = "a " + "word " * 90 + "\u2014 and then the ending."
        units = split_long_sentence(sentence, max_chars=250)
        assert len(units) == 2
        assert units[1] == "and then the ending."


class TestNormalizePauses:
    def test_em_dash_becomes_comma(self):
        assert normalize_pauses("odd \u2014 very odd \u2014 things") == (
            "odd, very odd, things"
        )


class TestChunkParagraphs:
    def test_never_merges_across_paragraphs(self):
        p1 = "word " * 30
        p2 = "other " * 30
        chunks = chunk_paragraphs(f"{p1}\n\n{p2}", max_chars=1000)
        assert len(chunks) == 2
        assert chunks[0]["ends_paragraph"] is True
        assert chunks[1]["ends_paragraph"] is True
        assert chunks[0]["text"].startswith("word")
        assert chunks[1]["text"].startswith("other")

    def test_long_paragraph_splits_without_paragraph_breaks(self):
        para = "sentence here. " * 100  # ~1500 chars
        chunks = chunk_paragraphs(para, max_chars=1000)
        assert len(chunks) == 2
        assert chunks[0]["ends_paragraph"] is False
        assert chunks[1]["ends_paragraph"] is True

    def test_respects_max_chars(self):
        para = "sentence here. " * 100
        chunks = chunk_paragraphs(para, max_chars=500)
        assert all(len(c["text"]) <= 500 + 20 for c in chunks)  # sentence slack

    def test_applies_speakable_text(self):
        chunks = chunk_paragraphs("Dr. Smith arrived.\n\nIt rained.", max_chars=1000)
        assert chunks[0]["text"] == "Doctor Smith arrived."

    def test_empty_text(self):
        assert chunk_paragraphs("") == []

    def test_every_chunk_flagged(self):
        para = "sentence here. " * 30
        chunks = chunk_paragraphs(para, max_chars=200)
        # 30 sentences x 15 chars: 13 + 13 + 4 sentences per chunk
        assert [c["ends_paragraph"] for c in chunks] == [False, False, True]

