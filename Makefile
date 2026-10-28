.PHONY: install ingest query eval test api

install:
	pip install -r requirements.txt

ingest:
	python scripts/ingest.py

query:
	python scripts/query.py

eval:
	python scripts/evaluate.py

test:
	pytest

api:
	uvicorn api:app --reload
