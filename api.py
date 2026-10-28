"""FastAPI server. Run with: uvicorn api:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.pipeline import RAGPipeline

app = FastAPI(title="local-rag", version="0.1.0")
_pipeline = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return pipeline.ask(request.question)
