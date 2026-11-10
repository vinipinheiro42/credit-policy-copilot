from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Every field can be set with a COPILOT_* environment variable."""

    model_config = SettingsConfigDict(
        env_prefix="COPILOT_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    store: Literal["pgvector", "sqlite"] = "sqlite"
    database_url: str = "postgresql://copilot:copilot@localhost:5432/copilot"
    sqlite_path: str = str(REPO_ROOT / ".data" / "copilot.sqlite")

    corpus_dir: str = str(REPO_ROOT / "data" / "corpus")
    manifest_path: str = str(REPO_ROOT / "data" / "corpus" / "MANIFEST.yaml")

    embedder: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    chunk_target_chars: int = 1200
    chunk_max_chars: int = 1800

    retrieval_mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    top_k: int = 5
    candidate_k: int = 30
    rrf_k: int = 60
    # On by default since M2: the calibrated abstention rule uses the reranker score, and the multilingual
    # cross-encoder had the best top-1 on the tuning split (docs/EVALS.md). Costs ~1 s/query on CPU.
    rerank: bool = True
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Fallback dense-only gate, used when no calibrated rule file exists.
    abstain_threshold: float = 0.83
    # Calibrated rule (written by `python -m copilot.evals calibrate`); "none" disables it.
    abstain_rule_path: str = str(REPO_ROOT / "evals" / "abstention_rule.json")

    llm_provider: Literal["none", "anthropic", "openai"] = "none"
    llm_model: str = ""
    llm_max_tokens: int = 700
    llm_timeout_s: float = 60.0
    # USD per million tokens, for cost reporting only. Left unset on purpose: prices change, set them
    # for the model you use (COPILOT_LLM_PRICE_INPUT_PER_MTOK / COPILOT_LLM_PRICE_OUTPUT_PER_MTOK).
    llm_price_input_per_mtok: float | None = None
    llm_price_output_per_mtok: float | None = None

    # Guardrails (input PII masking + injection filter, document neutralization, output checks)
    guardrails: bool = True

    # Observability
    log_level: str = "INFO"
    log_json: bool = True
    otel_exporter: Literal["none", "console", "otlp"] = "none"


@lru_cache
def get_settings() -> Settings:
    return Settings()
