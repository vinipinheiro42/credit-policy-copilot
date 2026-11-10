#!/usr/bin/env bash
# Simulates .github/workflows/ci.yml locally (no Docker / `act` needed): fresh venv with dev extras only
# (no torch / sentence-transformers), then lint -> unit tests -> golden verify -> eval gate.
# The integration job needs Postgres+pgvector; it runs here via embedded `pgserver` if installed in .venv-ci.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${CI_VENV:-.venv-ci}
export COPILOT_LOG_JSON=false COPILOT_OTEL_EXPORTER=none
unset ANTHROPIC_API_KEY OPENAI_API_KEY COPILOT_DATABASE_URL COPILOT_TEST_DATABASE_URL || true
step() { printf '\n=== %s ===\n' "$*"; }
start=$(date +%s)
step "install (fresh $VENV, dev extras)"
[ -d "$VENV" ] || uv venv -q -p 3.12 "$VENV"
VIRTUAL_ENV=$VENV uv pip install -q -e ".[dev]"
PY=$VENV/bin/python
step "lint";            $PY -m ruff check . && $PY -m ruff format --check .
step "unit tests";      $PY -m pytest -q -m "not integration"
step "golden verify";   $PY -m copilot.evals verify
step "eval gate";       $PY scripts/eval_gate.py --tolerance "${GATE_TOL:-0.02}"
step "integration";     $PY -m pytest -q -m integration -rs || { echo "integration failed"; exit 1; }
echo "CI (local) passed in $(( $(date +%s) - start ))s"
