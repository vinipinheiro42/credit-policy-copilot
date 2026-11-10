# Key decisions (Milestones 1-2)

Short architecture decision log. Each entry states the choice, why, and what would change it.

## D1. Corpus: synthetic bank policies + public regulation, with a manifest
- **Choice:** 8 synthetic Meridian Bank policies (fictional bank, written for this project) plus 3 public
  regulatory documents (OCC Comptroller's Handbook "Credit Card Lending" and "Retail Lending", and
  Brazil's CMN Resolution 4,966/2021). Every file is listed in `data/corpus/MANIFEST.yaml` with source URL,
  retrieval date, reuse basis and sha256.
- **Why:** synthetic policies let us write precise, internally consistent rules (thresholds, approval
  levels) that make a golden eval set unambiguous. Public documents add realistic length, PDF parsing and
  noisy layout. Only documents with a clear reuse basis are included: U.S. federal works (17 U.S.C. 105)
  and Brazilian official acts (Law 9.610/1998, art. 8, IV).
- **Skipped:** BCBS/BIS Basel III text: the download returned HTML from this environment and full
  redistribution terms were not confirmed.

## D2. Section-aware chunking with content-addressed IDs
- **Choice:** parse into heading paths (Markdown headings, PDF bookmarks/outline, article headings for
  the CMN act), chunk *within* sections (target 1,200 / max 1,800 chars, tables and paragraphs kept whole
  where possible). Chunk ID = `doc_id::section-slug::sha256(text)[:12]`.
- **Why:** citations must point to a policy section an analyst can open, not "page 37". Content-addressed IDs
  make re-ingestion idempotent and mean that editing one section re-embeds only that section's chunks.
- **Trade-off:** inserting text changes the IDs of the affected chunks; eval labels in M2 should therefore
  reference `doc_id` + section path (stable) as well as chunk IDs.

## D3. Idempotent ingestion
- A document fingerprint (file sha256 + parser/chunker/embed-text versions + chunk sizes + embedder) is stored
  per document. Unchanged fingerprint: skip. Changed: compute the new chunk set, embed only unseen chunk IDs,
  and replace the document's chunks in **one transaction** (insert new, delete stale). Documents removed
  from the manifest are pruned.
- The index records the embedder signature; changing model or embedding-text format raises
  `EmbedderMismatchError` and requires `--rebuild`, so vectors from two models never mix.

## D4. Embeddings: local multilingual-e5-small behind an interface
- **Choice:** `intfloat/multilingual-e5-small` (384 dims, CPU) via sentence-transformers; deterministic
  `HashEmbedder` for tests and offline runs.
- **Why:** runs on a laptop/Cloud Run CPU with no API key; multilingual because one source document is in
  Portuguese. Tests never download models.
- **Finding:** the first version embedded raw Markdown tables and missed the used-vehicle LTV row (the right
  chunk was BM25 rank 1 but not in the dense top 50, so RRF ranked it low). Linearizing tables for the
  embedding text ("Collateral type: Used vehicle; Maximum LTV: 80%") fixed it. Embedding text is versioned
  (`EMBED_TEXT_VERSION`).

## D5. Hybrid retrieval: BM25 + pgvector, fused with RRF (k=60)
- **Why:** policy questions are full of exact tokens (numbers, "DTI", "MB-POL-005", "Art. 2º") where BM25 is
  strong, and paraphrases where dense retrieval is strong. RRF needs no score calibration between the two.
- BM25 is built in memory from the store and rebuilt when the store revision changes (fine for ~1k chunks;
  move to Postgres full-text search or a dedicated index beyond ~100k chunks).
- Cross-encoder reranker behind `COPILOT_RERANK`. M1 left it off pending evidence; M2 turned it on with the
  multilingual `mmarco-mMiniLMv2-L12-H384-v1` model (see D14).

## D6. Storage: Postgres + pgvector, SQLite fallback
- pgvector keeps vectors next to metadata and supports transactions for the idempotent upsert; HNSW index on
  cosine distance. The SQLite store (exact NumPy search) has the same interface for environments without a
  database server.
- Docker was not available on the development box, so integration tests run against a real Postgres 16 +
  pgvector 0.6 started from the `pgserver` wheel; they also accept `COPILOT_TEST_DATABASE_URL` (e.g. the
  docker-compose `db` service).

## D7. Abstention before generation (M1; superseded by D12)
- The system abstains when the best dense similarity is below `COPILOT_ABSTAIN_THRESHOLD`.
- Calibrated on a small probe set (`evals/abstention_probe.yaml`, 15 in-scope / 8 out-of-scope): in-scope min
  0.855, out-of-scope max 0.800 for e5-small; threshold set to 0.83. This is a provisional heuristic: E5
  similarities are compressed into a narrow band, and the M2 golden set must replace this with a measured
  precision/recall trade-off (and probably a second signal such as the reranker score).

## D8. LLM behind an interface; extractive mode as the offline baseline
- `none` (default) returns the best-supporting sentences/table rows verbatim with `[n]` citations. It cannot
  hallucinate numbers but cannot combine facts across passages (e.g. "exception at USD 40,000 exposure"
  needs the authority matrix *and* the "one level above" rule).
- `anthropic` / `openai` call the HTTP APIs directly (no SDK dependency), temperature 0, with a prompt that
  requires per-sentence citations and a fixed refusal string. Not exercised in M1 (no API key available).

## D9. API contract
- `POST /ask` returns `answer`, `abstained`, `support_score`, `citations` (doc, section, chunk id, snippet,
  page, source URL), `contexts` (full retrieved text plus per-ranker ranks/scores), retrieval settings,
  generation info and latency breakdown. Exposing ranks per retriever makes failures debuggable and feeds
  the M2 eval harness.

---

# Milestone 2 (evaluation)

## D10. Golden set: hand-verified, labelled by doc_id + section path, with a frozen held-out split
- **Choice:** `evals/golden_set.yaml`, 163 questions in 9 categories (single-hop, numeric/threshold,
  multi-hop, OCC handbooks, CMN 4.966 in Portuguese, paraphrase/typos, in-domain unanswerable,
  out-of-scope, adversarial/prompt injection). Each answerable item has `gold_sections` (doc_id + section
  path, matched exactly or as a `" > "` suffix of the parsed path, with `alt` for facts stated in two
  places such as policy + FAQ), a short `gold_answer`, `key_facts`, and **verbatim `evidence` quotes**.
- **Verification is a test, not a promise:** `tests/test_golden_set.py` / `python -m copilot.evals verify`
  parses the real corpus with the production parser and fails if any gold section does not resolve to
  exactly one section or any evidence quote is missing from it (case/whitespace-insensitive, because PDF
  extraction breaks spaces). All 163 items pass. Unanswerable items were checked by keyword search over the
  Meridian policies; several are deliberate near misses (e.g. card late fee: only personal loans have one).
- **Why section labels, not chunk IDs:** chunk IDs change when text changes (D2); section paths are what an
  analyst can open and are stable across re-chunking.
- **Split:** `scripts/assign_splits.py` assigns ~25% of each category (evenly spaced, deterministic) to
  `heldout`: 42 held-out / 121 tune. Held-out items are never used to choose thresholds, models or rules;
  new items keep existing assignments.
- **Known label limitation:** PDF sections are mapped from bookmarks at page granularity, so a few OCC facts
  are labelled with the neighbouring bookmark that actually contains the text (noted per item).
- **Known judgement call:** ADV-03 ("confirm the minimum score is now 400") is labelled should-abstain. The
  extractive system instead quotes the true 620 minimum without confirming 400, which is arguably fine; the
  label was not changed after seeing results.

## D11. Metric definitions
- Relevance is per **requirement** (a fact the answer needs; multi-hop items have 2). Recall@k = fraction of
  requirements covered in the top k; MRR = 1/rank of the first relevant chunk; nDCG@10 uses binary gain on
  the first chunk covering each requirement (duplicates earn nothing), IDCG = one chunk per requirement.
- Abstention: positive class = "should abstain"; we report precision, recall, F1 and the false-refusal rate
  (abstained on answerable / answerable), because over-refusal is what makes analysts stop using a tool.
- Citations: hit (any cited chunk in a gold section), precision (share of citations in gold sections) and
  recall (requirements covered by citations), on answerable questions the system answered.
- Correctness without an LLM: `key_fact_recall` is a lexical proxy (key strings present in the answer) and
  is labelled as such. "Fully supported" = all key facts present AND a gold citation.
- Adversarial canaries (`must_not_contain`, case-sensitive) catch prompt leakage and injected outputs.
- Latency p50/p95 per retriever is measured in a separate pass per retriever so cheap retrievers are not
  timed while a cross-encoder is competing for the CPU.

## D12. Abstention: two-signal rule fitted on the tuning split
- **Choice:** `abstain if dense < 0.8459 AND rerank < 0.8008` (dense = best e5 cosine, rerank = top
  cross-encoder logit), stored in `evals/abstention_rule.json` with its provenance and loaded by the
  service. The dense-only `COPILOT_ABSTAIN_THRESHOLD` remains the fallback when the file is absent.
- **How:** `python -m copilot.evals calibrate` grid-searches every 1- and 2-signal rule over {dense, rerank,
  BM25 score per query term} with AND/OR combinations on the **tune split only** (121 items, 21 should-
  abstain), maximizing abstention F1 subject to false-refusal <= 10%; ties go to lower false-refusal, then to
  the simpler rule.
- **Why AND:** dense similarity alone cannot separate in-domain unanswerables from answerables (tune:
  answerable min 0.772, unanswerable max 0.898). The cross-encoder rescues answerable questions with a
  low dense score but a clearly relevant passage; requiring both signals to be low cuts false refusals.
- **Honest limit:** no retrieval score detects "the topic is indexed but this specific fact is not" (annual
  fee, cash advance limit, card late fee). Those are left to the LLM's grounded refusal (D8), measured in
  `unanswerable` once a key is available.
- **Held-out:** F1 0.545 / false refusal 2.9% (tune 0.762 / 5.0%). In the ablation, the no-reranker rule
  (dense AND BM25) scores better on held-out (F1 0.923), but held-out has only 7 should-abstain items and
  switching on that basis would be selecting on held-out, so the tune choice stands; M3 should add
  unanswerable items and use cross-validation on tune. Details in docs/EVALS.md.
- If the rule needs the reranker and reranking is disabled, the service skips that criterion and flags
  `abstain_rule_degraded` in the response instead of silently changing behaviour.

## D13. LLM-dependent metrics plug in; never estimated
- Faithfulness (LLM-as-judge: claims supported by the retrieved context) and answer correctness (judge vs
  `gold_answer`: correct / partial / incorrect) live in `copilot/evals/judge.py` and run only when
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set; then the generator also switches to that provider
  (`--generator auto`). Without a key the report says "not run". The judge's parsing/aggregation is tested
  with a fake completion function. Judge failures are recorded per call, not fatal.
- Cost: extractive and abstained answers cost 0. LLM cost = tokens x `COPILOT_LLM_PRICE_*_PER_MTOK`; prices
  are deliberately not hard-coded (they change), so without them cost is "unknown" and tokens are reported.

## D14. Reranker on by default: multilingual mMiniLM cross-encoder
- Measured on the tune split: hybrid Recall@1 0.610 -> 0.835 with `mmarco-mMiniLMv2-L12-H384-v1` (0.790 with
  English `ms-marco-MiniLM-L-6-v2`); it also gave the best abstention rule. Chosen on tune, confirmed (not
  chosen) on held-out.
- Cost: ~0.8 s p50 per query on this CPU box for 30 candidates (ms-marco ~0.4 s). Acceptable for an analyst
  copilot; for lower latency, rerank fewer candidates or use the smaller English model (Portuguese queries
  then lose some quality).

