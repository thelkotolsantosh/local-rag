"""Build the FAISS index from the documents in data/raw."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import build_index  # noqa: E402


def main() -> None:
    try:
        stats = build_index()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
    print(f"Indexed {stats['documents']} document(s) into {stats['chunks']} chunk(s).")
    print("Index saved. Now run: python scripts/query.py")


if __name__ == "__main__":
    main()
