"""End-to-end pipeline: build the index, then answer questions."""
from typing import Dict, Optional

from .chunker import chunk_documents
from .config import load_config
from .embedder import Embedder
from .loader import load_documents
from .retriever import Retriever
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided "
    "context. If the answer is not in the context, say you don't know. "
    "Keep the answer concise."
)
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def build_index(cfg: Optional[dict] = None) -> Dict[str, int]:
    """Load -> chunk -> embed -> index -> save. Returns simple stats."""
    cfg = cfg or load_config()
    docs = load_documents(cfg["paths"]["raw_dir"])
    if not docs:
        raise RuntimeError(
            f"No .txt/.md/.pdf files found in {cfg['paths']['raw_dir']}. "
            "Add some documents and try again."
        )
    chunks = chunk_documents(docs, cfg["chunk_size"], cfg["chunk_overlap"])
    embedder = Embedder(cfg["embedding_model"])
    embeddings = embedder.encode([c["text"] for c in chunks])

    store = VectorStore()
    store.add(embeddings, chunks)
    store.save(cfg["paths"]["index_dir"])
    return {"documents": len(docs), "chunks": len(chunks)}


def build_context(hits) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )


class RAGPipeline:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or load_config()
        self.embedder = Embedder(self.cfg["embedding_model"])
        self.store = VectorStore.load(self.cfg["paths"]["index_dir"])
        self.retriever = Retriever(self.embedder, self.store, self.cfg["top_k"])
        self._llm = None  # loaded lazily so retrieval-only use stays fast

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LocalLLM

            self._llm = LocalLLM(
                self.cfg["llm_model"],
                max_new_tokens=self.cfg["max_new_tokens"],
                temperature=self.cfg["temperature"],
            )
        return self._llm

    def ask(self, question: str) -> Dict:
        hits = self.retriever.retrieve(question)
        if not hits:
            return {"answer": "I don't have any documents to answer from.", "sources": []}
        prompt = USER_TEMPLATE.format(context=build_context(hits), question=question)
        answer = self.llm.generate(SYSTEM_PROMPT, prompt)
        sources = [
            {"source": h["source"], "chunk_id": h["chunk_id"], "score": h["score"]}
            for h in hits
        ]
        return {"answer": answer, "sources": sources}
