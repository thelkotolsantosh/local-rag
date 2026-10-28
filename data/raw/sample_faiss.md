# FAISS and embeddings

FAISS is an open-source library for fast similarity search over dense vectors. In a RAG system, FAISS is used to store the embedding of every document chunk and to find the chunks closest to a question embedding in a few milliseconds, even when there are millions of vectors.

## Flat index

The simplest FAISS index is a flat index. It compares the query against every stored vector, so results are exact. A flat index is a good default for small and medium collections, and it needs no training step.

## Cosine similarity and normalization

Cosine similarity measures the angle between two vectors and ignores their length. If every embedding is normalized to unit length, the inner product of two vectors equals their cosine similarity. That is why embeddings are normalized before being added to an inner-product index: the fast inner-product search then returns the same ranking as cosine similarity.

## Embedding models

A sentence-embedding model such as all-MiniLM-L6-v2 maps text to a 384-dimensional vector. Texts with similar meaning end up close together in that space, which is what makes semantic search work even when the question and the document share few exact words.
