"""FAISS vector store with JSON metadata, saved to disk."""
import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

INDEX_FILE = "index.faiss"
RECORDS_FILE = "records.json"


class VectorStore:
    def __init__(self):
        self.index = None
        self.records: List[Dict] = []

    def add(self, embeddings: np.ndarray, records: List[Dict]) -> None:
        if len(embeddings) != len(records):
            raise ValueError("embeddings and records must have the same length")
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.records.extend(records)

    def search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype="float32")
        scores, ids = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            record = dict(self.records[idx])
            record["score"] = float(score)
            results.append(record)
        return results

    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        with open(directory / RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory) -> "VectorStore":
        directory = Path(directory)
        index_path = directory / INDEX_FILE
        records_path = directory / RECORDS_FILE
        if not index_path.exists() or not records_path.exists():
            raise FileNotFoundError(
                f"No index found in {directory}. Run: python scripts/ingest.py"
            )
        store = cls()
        store.index = faiss.read_index(str(index_path))
        with open(records_path, "r", encoding="utf-8") as f:
            store.records = json.load(f)
        return store
