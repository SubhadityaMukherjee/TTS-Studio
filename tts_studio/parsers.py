import os
import re

import fitz  # PyMuPDF
import pymupdf4llm
import pymupdf.layout
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from .utils import normalize_text, split_sentences


class EpubParser:
    @staticmethod
    def _extract_paragraphs(soup):
        """
        Layered paragraph extraction:
        1. <p> tags
        2. block-level containers
        3. newline split fallback

        Uses an empty separator so that words split across inline tags
        (e.g. ``well-<span>known</span>``) stay joined as ``well-known``
        instead of becoming two words with a spurious TTS break.
        """

        # 1️⃣ Standard <p> tags
        paragraphs = []
        for p in soup.find_all("p"):
            text = re.sub(r"\s+", " ", p.get_text("", strip=False)).strip()
            if text:
                paragraphs.append(text)
        if paragraphs:
            return paragraphs

        # 2️⃣ Block-level fallback
        block_tags = soup.find_all(["div", "section", "article", "li"])
        paragraphs = [
            re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
            for tag in block_tags
        ]
        paragraphs = [p for p in paragraphs if p]
        if paragraphs:
            return paragraphs

        # 3️⃣ <br> / newline fallback
        raw_text = soup.get_text("\n", strip=True)
        return [p.strip() for p in raw_text.split("\n") if p.strip()]

    @staticmethod
    def _extract_title(soup, item):
        """
        More flexible title detection.
        """
        title_tag = soup.find(["h1", "h2", "h3", "title"])
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)

        return os.path.basename(item.get_name())

    @staticmethod
    def extract_chapters(epub_file):
        """
        Extract chapters from an EPUB file with layered fallback parsing.

        Keeps original behavior but improves robustness across
        non-standard EPUB structures.
        """

        book = epub.read_epub(epub_file)
        chapters = []

        # ✅ Use spine order (correct reading order)
        spine_items = []
        for idref, _ in book.spine:
            item = book.get_item_with_id(idref)
            if item and item.get_type() == ITEM_DOCUMENT:
                spine_items.append(item)

        for idx, item in enumerate(spine_items):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")

            title = EpubParser._extract_title(soup, item)
            paragraphs = EpubParser._extract_paragraphs(soup)

            full_text = "\n\n".join(paragraphs).strip()

            if not full_text:
                continue

            sentences = []
            for para in paragraphs:
                sentences.extend(split_sentences(para))

            chapters.append(
                {
                    "title": title,
                    "content": full_text,
                    "sentences": sentences,
                    "order": idx + 1,
                }
            )

        # 🔎 Heuristic fallback:
        # If only one large chapter detected, try splitting by headings
        if len(chapters) == 1:
            item = spine_items[0] if spine_items else None
            if item:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                headings = soup.find_all(["h1", "h2", "h3"])

                if len(headings) > 1:
                    split_chapters = []
                    for idx, header in enumerate(headings):
                        content = []
                        for sib in header.find_next_siblings():
                            if sib.name in ["h1", "h2", "h3"]:
                                break
                            text = sib.get_text(" ", strip=True)
                            if text:
                                content.append(text)

                        text = "\n\n".join(content).strip()
                        if text:
                            split_chapters.append(
                                {
                                    "title": header.get_text(strip=True),
                                    "content": text,
                                    "sentences": split_sentences(text),
                                    "order": idx + 1,
                                }
                            )

                    if split_chapters:
                        chapters = split_chapters

        if not chapters:
            all_text_chunks = []
            for item in book.get_items_of_type(ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                text = soup.get_text(" ", strip=True)
                if text:
                    all_text_chunks.append(text)

            if all_text_chunks:
                combined = "\n\n".join(all_text_chunks)
                sentences = split_sentences(combined)
                book_title = book.get_metadata("DC", "title")
                title = book_title[0][0] if book_title else os.path.basename(epub_file)
                chapters.append(
                    {
                        "title": title,
                        "content": combined,
                        "sentences": sentences,
                        "order": 1,
                    }
                )

        return chapters


class PdfParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def get_chapters(self):
        """
        Extract layout-aware text using pymupdf4llm + layout activation.
        """
        # Open document - layout analysis is now active
        doc = fitz.open(self.pdf_path)

        # Use layout-aware markdown extraction (handles columns, reading order)
        md_text = pymupdf4llm.to_markdown(
            doc,
            write_images=False,  # Skip images
            header=False,  # Skip headers
            footer=False,  # Skip footers
        )
        doc.close()

        # Split by markdown headers
        sections = re.split(r"(?m)^#{1,6}\s+", md_text)

        chapters = []
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) < 50:  # Skip tiny fragments
                continue

            lines = section.split("\n", 1)
            title = lines[0].strip("# ").strip() if lines else f"Section {i+1}"
            content = lines[1].strip() if len(lines) > 1 else section

            # Paragraphs and sentences (strip markdown so it isn't read aloud)
            paragraphs = [
                normalize_text(p, strip_markdown=True)
                for p in re.split(r"\n{2,}", content)
                if p.strip()
            ]
            paragraphs = [p for p in paragraphs if p]
            sentences = []
            for para in paragraphs:
                sentences.extend(split_sentences(para))

            chapters.append(
                {
                    "title": title[:100],
                    "content": "\n\n".join(paragraphs),
                    "paragraphs": paragraphs,
                    "sentences": sentences,
                    "order": i + 1,
                }
            )

        # fallback: if markdown-based section splitting yielded nothing,
        # fall back to raw page text.
        if not chapters:
            doc = fitz.open(self.pdf_path)
            full_text = []
            for page in doc:
                txt = page.get_text("text")
                if txt:
                    full_text.append(txt)
            doc.close()
            combined = "\n\n".join(full_text).strip()
            if combined:
                paras = [
                    normalize_text(p)
                    for p in re.split(r"\n{2,}", combined)
                    if p.strip()
                ]
                paras = [p for p in paras if p]
                sentences = []
                for para in paras:
                    sentences.extend(split_sentences(para))
                chapters.append(
                    {
                        "title": os.path.basename(self.pdf_path),
                        "content": combined,
                        "paragraphs": paras,
                        "sentences": sentences,
                        "order": 1,
                    }
                )

        return chapters


if __name__ == "__main__":
    pdfpars = PdfParser(pdf_path="/Users/smukherjee/Downloads/2203.02395v1.pdf")
    chaps = pdfpars.get_chapters()
    print(chaps)
