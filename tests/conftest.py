import nltk
import pytest


@pytest.fixture(scope="session", autouse=True)
def _nltk_data():
    """Ensure punkt data is available for sentence tokenization."""
    for package in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            nltk.download(package, quiet=True)
