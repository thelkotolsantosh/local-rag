"""Retrieval hit-rate check (no LLM needed).

Each line of data/eval/questions.jsonl looks like:
  {"question": "...", "expected_source": "file_name.md"}
A question is a "hit" if any of the top-k retrieved chunks comes from expected_source.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    cfg = load_config()
    eval_path = Path(cfg["paths"]["eval_file"])
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}")
        sys.exit(1)

    try:
        pipeline = RAGPipeline(cfg)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hits = 0
    for row in rows:
        retrieved = pipeline.retriever.retrieve(row["question"])
        found = any(r["source"] == row["expected_source"] for r in retrieved)
        hits += int(found)
        print(f"[{'HIT ' if found else 'MISS'}] {row['question']}")

    rate = hits / len(rows) if rows else 0.0
    print(f"\nhit@{cfg['top_k']}: {hits}/{len(rows)} = {rate:.0%}")


if __name__ == "__main__":
    main()
