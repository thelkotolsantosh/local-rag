"""Read raw documents (.txt, .md, .pdf) from a folder."""
from pathlib import Path
from typing import Dict, List

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(raw_dir) -> List[Dict]:
    """Return [{'source': relative_path, 'text': full_text}, ...]."""
    raw_dir = Path(raw_dir)
    docs = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        text = text.strip()
        if text:
            docs.append({"source": path.relative_to(raw_dir).as_posix(), "text": text})
    return docs
