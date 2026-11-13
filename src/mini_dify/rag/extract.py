import io
import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class UnsupportedFileType(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Get plain text out of an uploaded file."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise UnsupportedFileType(f"Unsupported file type: {suffix or 'none'}")


def clean_text(text: str) -> str:
    """Normalise extracted text before splitting it."""
    text = text.replace("\x00", "")  # Postgres text columns cannot store NUL bytes
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)  # runs of spaces/tabs -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines -> one blank line
    return text.strip()
