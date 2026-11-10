# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch first (the default PyPI wheel pulls CUDA and is ~10x larger).
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[ml]"

# Bake the embedding and reranker models into the image so containers start offline.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ARG RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANK_MODEL}')"

COPY data ./data
COPY evals/abstention_rule.json ./evals/abstention_rule.json

RUN useradd --create-home --uid 10001 app && chown -R app /app /opt/hf
USER app

ENV COPILOT_STORE=pgvector \
    COPILOT_CORPUS_DIR=/app/data/corpus \
    COPILOT_MANIFEST_PATH=/app/data/corpus/MANIFEST.yaml \
    COPILOT_ABSTAIN_RULE_PATH=/app/evals/abstention_rule.json \
    PORT=8080

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn copilot.api.app:app --host 0.0.0.0 --port ${PORT}"]
