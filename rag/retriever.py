"""Turn a question into the top-k most similar chunks."""
from typing import Dict, List, Optional


class Retriever:
    def __init__(self, embedder, store, top_k: int = 4):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        query_embedding = self.embedder.encode([query])[0]
        return self.store.search(query_embedding, top_k or self.top_k)
