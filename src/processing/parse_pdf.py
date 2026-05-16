from pathlib import Path
from pypdf import PdfReader


def parse_pdf(file_path: str) -> list[dict]:
    
    path = Path(file_path)
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        pages.append({
            "page": i + 1,
            "text": text,
            "source": path.name,
        })

    return pages