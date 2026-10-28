import numpy as np
import pytest

pytest.importorskip("faiss")

from rag.vectorstore import VectorStore  # noqa: E402


def make_store():
    embeddings = np.eye(3, dtype="float32")
    records = [
        {"source": "a", "chunk_id": 0, "text": "x"},
        {"source": "b", "chunk_id": 0, "text": "y"},
        {"source": "c", "chunk_id": 0, "text": "z"},
    ]
    store = VectorStore()
    store.add(embeddings, records)
    return store


def test_search_returns_closest_first():
    store = make_store()
    results = store.search(np.array([0, 1, 0], dtype="float32"), 2)
    assert results[0]["source"] == "b"
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_save_and_load_roundtrip(tmp_path):
    make_store().save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    results = loaded.search(np.array([0, 0, 1], dtype="float32"), 1)
    assert results[0]["source"] == "c"


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path)
