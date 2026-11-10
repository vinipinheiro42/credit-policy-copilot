PY ?= .venv/bin/python
export COPILOT_STORE ?= sqlite

.PHONY: install install-ml lint format test test-unit test-integration fetch-docs manifest ingest rebuild serve pg-up pg-down probe docker-up eval eval-verify eval-heldout calibrate guardrail-eval eval-gate ci-local

install:            ## create venv + install package with dev extras (no ML models)
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

install-ml: install ## add CPU torch + sentence-transformers
	uv pip install --python $(PY) torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python $(PY) -e ".[ml,dev]"

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

test:               ## all tests (integration tests auto-skip without Postgres)
	$(PY) -m pytest

test-unit:
	$(PY) -m pytest -m "not integration"

test-integration:
	$(PY) -m pytest -m integration -v

fetch-docs:         ## re-download public regulatory documents
	$(PY) scripts/fetch_public_docs.py

manifest:           ## recompute sha256 in data/corpus/MANIFEST.yaml
	$(PY) scripts/manifest_hashes.py --write

ingest:
	$(PY) scripts/manifest_hashes.py --check
	$(PY) -m copilot.ingest.cli

rebuild:
	$(PY) -m copilot.ingest.cli --rebuild

serve:
	$(PY) -m uvicorn copilot.api.app:app --port 8000

probe:              ## print abstention scores for in/out-of-scope probe questions
	$(PY) scripts/probe_abstention.py

pg-up:              ## local Postgres+pgvector without Docker; prints COPILOT_DATABASE_URL
	@$(PY) scripts/dev_postgres.py

pg-down:
	$(PY) scripts/dev_postgres.py --stop

docker-up:
	docker compose up -d db && docker compose run --rm ingest && docker compose up -d api

eval-verify:        ## check every golden-set answer/section against the corpus (no models needed)
	$(PY) -m copilot.evals verify

calibrate:          ## choose + fit the abstention rule by 5x5 CV on the TUNING split only -> evals/abstention_rule.json
	$(PY) -m copilot.evals calibrate

eval:               ## full eval (all retrievers + end-to-end) -> evals/results/<timestamp>.{json,md}
	$(PY) -m copilot.evals run --rerank-models cross-encoder/ms-marco-MiniLM-L-6-v2

eval-heldout:       ## end-to-end numbers on the held-out split only
	$(PY) -m copilot.evals run --split heldout

guardrail-eval:     ## PII / injection / output-check / poisoned-doc metrics (offline, seconds)
	$(PY) -c "import json; from copilot.evals.guardrail_eval import run_guardrail_eval, summary; print(json.dumps(summary(run_guardrail_eval(golden_path='evals/golden_set.yaml')), indent=1))"

eval-gate:          ## fast-config eval regression gate vs evals/baselines/fast.json (what CI runs)
	$(PY) scripts/eval_gate.py

ci-local:           ## run the steps of .github/workflows/ci.yml locally in a fresh .venv-ci
	scripts/ci_local.sh
