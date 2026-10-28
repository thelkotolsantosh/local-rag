# local-rag

A small, fully local Retrieval-Augmented Generation (RAG) project.
Documents -> chunks -> embeddings -> FAISS index -> retrieve top-k -> answer with a ~1B local LLM.

No API keys needed. Everything runs on your own machine (CPU works, GPU is faster).

## How it works

```
data/raw/*.txt|md|pdf
        |  rag/loader.py
        v
   raw documents
        |  rag/chunker.py   (overlapping word chunks)
        v
      chunks
        |  rag/embedder.py  (sentence-transformers)
        v
    embeddings --> rag/vectorstore.py (FAISS, saved to index/)
                                |
 question --> embed --> top-k search (rag/retriever.py)
                                |
                  prompt = context + question
                                |  rag/llm.py (Hugging Face transformers)
                                v
                             answer + sources
```

## Project layout

```
local-rag/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── config.yaml            # all settings: models, chunk size, top_k, paths
├── pytest.ini
├── Makefile
├── Dockerfile
├── api.py                 # FastAPI server: POST /ask
├── rag/
│   ├── __init__.py
│   ├── config.py          # loads config.yaml
│   ├── loader.py          # read .txt / .md / .pdf
│   ├── chunker.py         # split text into overlapping chunks
│   ├── embedder.py        # sentence-transformers wrapper
│   ├── vectorstore.py     # FAISS index + metadata persistence
│   ├── retriever.py       # query -> top-k chunks
│   ├── llm.py             # local LLM wrapper
│   └── pipeline.py        # ties everything together
├── scripts/
│   ├── ingest.py          # build the index from data/raw
│   ├── query.py           # ask questions from the terminal
│   └── evaluate.py        # retrieval hit-rate check
├── data/
│   ├── raw/               # put your documents here (sample_*.md included)
│   ├── processed/
│   └── eval/questions.jsonl
├── index/                 # generated FAISS index (git-ignored)
├── tests/
│   ├── test_chunker.py
│   ├── test_loader.py
│   └── test_vectorstore.py
└── .github/workflows/ci.yml
```

## Quick start

```bash
git clone <your-repo-url> local-rag
cd local-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Put your `.txt`, `.md` or `.pdf` files in `data/raw/`.
2. Build the index:
   ```bash
   python scripts/ingest.py
   ```
3. Ask questions:
   ```bash
   python scripts/query.py "What is RAG?"
   python scripts/query.py            # interactive mode
   ```
4. Optional API server:
   ```bash
   uvicorn api:app --reload
   curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
        -d '{"question": "What is RAG?"}'
   ```
5. Check retrieval quality:
   ```bash
   python scripts/evaluate.py
   ```

## Configuration

Edit `config.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `embedding_model` | sentence-transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `llm_model` | Hugging Face chat model (~1B) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `chunk_size` / `chunk_overlap` | words per chunk / overlap | 200 / 40 |
| `top_k` | chunks passed to the LLM | 4 |
| `max_new_tokens` | answer length cap | 256 |

Swap in any other small instruct model (e.g. `meta-llama/Llama-3.2-1B-Instruct`, which needs a Hugging Face login) by changing `llm_model`.
If a model is gated, copy `.env.example` to `.env` and set `HF_TOKEN`.

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t local-rag .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/index:/app/index local-rag
```
(Run `python scripts/ingest.py` first, or exec it inside the container.)

## Ideas to extend

- Hybrid search (BM25 + vectors) and a re-ranker
- Metadata filtering, per-document citations with page numbers
- Swap FAISS for Chroma or Qdrant
- Quantized GGUF models via llama-cpp-python for faster CPU inference
- Better evaluation (RAGAS, LLM-as-judge)

## License

MIT
