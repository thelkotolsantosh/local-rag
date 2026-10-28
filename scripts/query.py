"""Ask questions from the terminal (one-shot or interactive)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import RAGPipeline  # noqa: E402


def show(result: dict) -> None:
    print("\nAnswer:\n" + result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_id']}, score {s['score']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local RAG system a question.")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    args = parser.parse_args()

    try:
        pipeline = RAGPipeline()
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    if args.question:
        show(pipeline.ask(args.question))
        return

    print("Interactive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        show(pipeline.ask(question))


if __name__ == "__main__":
    main()
