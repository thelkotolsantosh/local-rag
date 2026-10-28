"""Split documents into overlapping word-based chunks."""
from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """Split text into chunks of `chunk_size` words, sharing `overlap` words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """Turn documents into [{'source', 'chunk_id', 'text'}, ...]."""
    out = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            out.append({"source": doc["source"], "chunk_id": i, "text": chunk})
    return out
