import pytest

from rag.chunker import chunk_documents, chunk_text


def make_text(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text(make_text(5), chunk_size=10, overlap=2) == [make_text(5)]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", chunk_size=10, overlap=2) == []


def test_chunks_overlap_and_cover_everything():
    chunks = chunk_text(make_text(25), chunk_size=10, overlap=2)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 10 for c in chunks)
    assert chunks[1].split()[:2] == chunks[0].split()[-2:]
    assert chunks[-1].split()[-1] == "w24"


def test_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=5, overlap=5)


def test_chunk_documents_keeps_source_and_ids():
    docs = [{"source": "a.md", "text": make_text(25)}, {"source": "b.md", "text": "hello"}]
    chunks = chunk_documents(docs, chunk_size=10, overlap=2)
    assert [c["chunk_id"] for c in chunks if c["source"] == "a.md"] == [0, 1, 2]
    assert [c["source"] for c in chunks].count("b.md") == 1
