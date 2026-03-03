import os
import re
import fitz  # PyMuPDF
import pymupdf.layout
import pymupdf4llm

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from nltk import sent_tokenize


class EpubParser:
    @staticmethod
    def extract_chapters(epub_file):
        """
        Extract chapters from an EPUB file with improved sentence and paragraph parsing.
        """
        book = epub.read_epub(epub_file)
        chapters = []

        for idx, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")

            # Extract title from <h1> or <title>, fallback to item filename
            title_tag = soup.find(["h1", "title"])
            title = (
                title_tag.get_text(strip=True)
                if title_tag
                else os.path.basename(item.get_name())
            )

            # Clean up text content
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if p.get_text(strip=True)
            ]
            full_text = "\n\n".join(paragraphs)

            # Sentence-level parsing for better TTS or downstream use
            sentences = []
            for para in paragraphs:
                sentences.extend(sent_tokenize(para))

            if full_text:
                chapters.append(
                    {
                        "title": title,
                        "content": full_text,
                        "sentences": sentences,
                        "order": idx + 1,
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

            # Paragraphs and sentences
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
            sentences = []
            for para in paragraphs:
                sentences.extend(sent_tokenize(para))

            chapters.append(
                {
                    "title": title[:100],
                    "content": content,
                    "paragraphs": paragraphs,
                    "sentences": sentences,
                    "order": i + 1,
                }
            )

        return chapters
