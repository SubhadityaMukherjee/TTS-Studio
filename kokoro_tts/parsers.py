import os
import re
import fitz  # PyMuPDF
import pymupdf4llm
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup


class EpubParser:
    @staticmethod
    def extract_chapters(epub_file):
        book = epub.read_epub(epub_file)
        chapters = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text().strip()
            if text:
                chapters.append(
                    {
                        "title": item.get_name(),
                        "content": text,
                        "order": len(chapters) + 1,
                    }
                )
        return chapters


class PdfParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def get_chapters(self):
        # Using the markdown approach as it's more reliable for TTS flow
        md_text = pymupdf4llm.to_markdown(self.pdf_path)
        # Simple split by headers
        sections = re.split(r"#+\s+", md_text)
        chapters = []
        for i, content in enumerate(sections):
            if not content.strip():
                continue
            lines = content.split("\n")
            chapters.append(
                {
                    "title": lines[0][:50] if lines else f"Section {i}",
                    "content": content.strip(),
                    "order": i,
                }
            )
        return chapters
