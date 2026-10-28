# Retrieval-Augmented Generation (RAG)

RAG stands for Retrieval-Augmented Generation. It combines a search step with a text-generation step. Instead of asking a language model to answer from memory alone, a RAG system first retrieves passages that are relevant to the question and then hands those passages to the model as context.

## Why use RAG

- Answers are grounded in your own documents, so the model can talk about private or very recent information.
- Grounding reduces hallucination, because the model is told to answer only from the supplied context.
- Knowledge is updated by re-indexing documents. No model retraining is needed.

## Pipeline stages

1. Loading: read raw files such as text, markdown and PDF.
2. Chunking: split long documents into smaller pieces. Chunks overlap a little so that a sentence cut at a boundary still appears whole in at least one chunk. This is why we split documents into overlapping chunks.
3. Embedding: convert each chunk into a vector with an embedding model.
4. Indexing: store the vectors in a vector index.
5. Retrieval: embed the question and fetch the top-k most similar chunks.
6. Generation: build a prompt from the retrieved chunks and the question, then let a language model write the answer and cite its sources.
