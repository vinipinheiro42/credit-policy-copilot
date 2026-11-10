# Credit Policy Copilot

Retrieval-augmented assistant that lets credit analysts at **Meridian Bank** (a fictional bank) ask questions
about internal credit policies and regulatory documents and get answers **with citations to the exact policy
section**, or an explicit refusal when the documents do not cover the question.

> Status: **Milestone 1** (ingestion, hybrid retrieval, API, offline extractive answers, tests).
> Evals (100+ question golden set), guardrails, LLM generation, CI and Cloud Run deploy come next.
> All Meridian Bank documents are synthetic. See `data/corpus/MANIFEST.yaml` for the source and reuse basis
> of every document.

## The problem (client framing)

Meridian Bank's retail credit analysts work from dozens of policy documents (underwriting, limits,
collateral, collections, exceptions, KYC/AML, privacy) plus external regulation. Questions like *"Who must
approve a DTI exception at USD 40,000 total exposure?"* take minutes of searching and are answered
inconsistently. The bank wants an assistant that answers from approved documents only, always shows its
source, and refuses rather than guesses, because a wrong threshold in a credit decision is a compliance
incident.

## Architecture (M1)

```
MANIFEST.yaml ─► parse (md / pdf outline / txt) ─► section-aware chunks (stable IDs)
                                                        │ embed only new chunks (e5-small, CPU)
                                                        ▼
                                  Postgres + pgvector (HNSW)  ◄─ SQLite fallback
                                                        │
question ─► BM25 ─┐                                     │
          └► dense ┴─► RRF fusion ─► [cross-encoder rerank] ─► abstention gate ─► answer + citations
                                                                         (extractive | Claude | OpenAI)
```

Key choices and trade-offs: [docs/DECISIONS.md](docs/DECISIONS.md).

## Quickstart (no Docker, no API key)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[ml,dev]"

export COPILOT_STORE=pgvector
export COPILOT_DATABASE_URL=$(python scripts/dev_postgres.py)   # embedded Postgres 16 + pgvector
python -m copilot.ingest.cli                                    # idempotent; re-run is a no-op
uvicorn copilot.api.app:app --port 8000

python scripts/ask.py "What is the maximum loan-to-value for a used vehicle?"
```

Without Postgres, set `COPILOT_STORE=sqlite` (file under `.data/`).
With Docker: `docker compose up -d db && docker compose run --rm ingest && docker compose up api`.

## API

`POST /ask` `{"question": "...", "top_k": 5, "mode": "hybrid|bm25|vector", "rerank": false}` returns
`answer`, `abstained`, `support_score`, `citations[]` (doc, section, chunk id, snippet), `contexts[]`
(retrieved text with per-retriever ranks), and latency. `GET /health` reports index size and configuration.
Example responses: [docs/samples/](docs/samples/).

## Evaluation (M2)

163 hand-verified questions (`evals/golden_set.yaml`, 42 held out), checked against the corpus by a test.
Held-out results (extractive generator, no LLM key), full method and failure analysis in
[docs/EVALS.md](docs/EVALS.md):

| | Recall@1 | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.786 | 0.929 | 0.881 | 0.886 |
| dense (e5-small) | 0.714 | 0.843 | 0.787 | 0.795 |
| hybrid RRF | 0.743 | 0.886 | 0.837 | 0.854 |
| hybrid + multilingual cross-encoder (default) | 0.829 | 0.986 | 0.919 | 0.930 |

End-to-end (held-out): citation hit 0.97, false refusal 2.9%, abstention F1 0.55. Questions about a
policy topic whose specific fact is missing are the main open problem. LLM-judged faithfulness and
correctness are implemented but **not run** (no API key).

```bash
python -m copilot.evals verify     # golden set vs corpus
python -m copilot.evals calibrate  # abstention rule, tune split only
python -m copilot.evals run        # -> evals/results/<timestamp>.{json,md}
```

## Tests

```bash
pytest            # unit + API contract + pgvector integration (auto-skips without Postgres)
pytest -m slow    # loads the real embedding model
```

## License

Code: MIT. Corpus documents keep their own terms (see the manifest).
