import fitz
import pytest
from ebooklib import epub

from tts_studio.parsers import EpubParser, PdfParser


def _chapter(title, paragraphs, file_name):
    item = epub.EpubHtml(title=title, file_name=file_name, lang="en")
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    item.content = (
        f"<html><body><h1>{title}</h1>{body}</body></html>"
    ).encode("utf-8")
    return item


@pytest.fixture
def epub_file(tmp_path):
    book = epub.EpubBook()
    book.set_title("Test Book")
    book.set_language("en")

    chapters = [
        _chapter(
            "Chapter One",
            ["Hello world.", "This is the first chapter."],
            "chap1.xhtml",
        ),
        _chapter(
            "Chapter Two",
            ["Second chapter here.", "It has more text."],
            "chap2.xhtml",
        ),
    ]
    for chapter in chapters:
        book.add_item(chapter)
    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = chapters

    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)
    return str(path)


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    lines = ["A Study of Testing"]
    lines += [f"Body paragraph {i} with several words in it." for i in range(10)]
    page.insert_text((72, 72), "\n".join(lines))
    doc.save(str(path))
    doc.close()
    return str(path)


class TestEpubParser:
    def test_extracts_chapters_in_spine_order(self, epub_file):
        chapters = EpubParser.extract_chapters(epub_file)
        assert [c["title"] for c in chapters] == ["Chapter One", "Chapter Two"]
        assert [c["order"] for c in chapters] == [1, 2]
        assert "Hello world." in chapters[0]["content"]
        assert chapters[0]["sentences"]

    def test_words_split_across_tags_stay_joined(self, tmp_path):
        book = epub.EpubBook()
        book.set_title("Split Book")
        item = epub.EpubHtml(title="Only", file_name="only.xhtml", lang="en")
        item.content = (
            b"<html><body><h1>Only Chapter</h1>"
            b"<p>well-<span>known</span> fact.</p></body></html>"
        )
        book.add_item(item)
        book.toc = (item,)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = [item]
        path = tmp_path / "split.epub"
        epub.write_epub(str(path), book)

        chapters = EpubParser.extract_chapters(str(path))
        assert len(chapters) == 1
        assert "well-known" in chapters[0]["content"]

    def test_single_chapter_split_by_headings(self, tmp_path):
        book = epub.EpubBook()
        book.set_title("Heading Book")
        item = epub.EpubHtml(title="All", file_name="all.xhtml", lang="en")
        item.content = (
            b"<html><body>"
            b"<h2>Part One</h2><p>First part text.</p>"
            b"<h2>Part Two</h2><p>Second part text.</p>"
            b"</body></html>"
        )
        book.add_item(item)
        book.toc = (item,)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = [item]
        path = tmp_path / "headings.epub"
        epub.write_epub(str(path), book)

        chapters = EpubParser.extract_chapters(str(path))
        titles = [c["title"] for c in chapters]
        assert "Part One" in titles and "Part Two" in titles


class TestPdfParser:
    def test_extracts_text(self, pdf_file):
        chapters = PdfParser(pdf_file).get_chapters()
        assert chapters
        combined = "\n\n".join(c["content"] for c in chapters)
        assert "Body paragraph 3" in combined
        first = chapters[0]
        assert first["order"] == 1
        assert first["title"]
        assert first["sentences"]

    def test_titles_truncated(self, pdf_file):
        chapters = PdfParser(pdf_file).get_chapters()
        assert all(len(c["title"]) <= 100 for c in chapters)
