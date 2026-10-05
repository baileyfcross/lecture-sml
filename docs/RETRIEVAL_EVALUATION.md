# Local Retrieval Evaluation

## Scope

This subsystem measures the existing SQLite FTS5 + local FastEmbed + RRF retrieval pipeline and its `SourceMaterial` context assembly. It is independent of SFT, teaching-candidate ingestion, preference collection, and generation evaluation. It does not call Ollama or use an LLM as a judge. The first real-vault pass is **query → retrieval → context assembly**, with no generation.

The implementation does not establish that retrieval quality is good. Only manually authored, human-reviewed cases run against the user's real index can do that. No real vault is opened by tests or by implementation-time validation.

## Explicit local workflow

First inspect the supplied vault only with the no-write dry-run:

```powershell
uv run python scripts/index_knowledge.py "D:\My Obsidian Vault" --dry-run
```

The report includes discovered non-ignored file count, supported and unsupported file counts by extension, ignored directory names, exact-byte duplicate candidate groups, supported-file count and byte workload, target index directory, configured embedding model, and whether the configured model cache contains files. It hashes supported files to recognize duplicates but performs no extraction, embedding, network access, cache creation, or vault writes. The cache field is an on-disk presence indicator, not a model-integrity check.

Only after reviewing the dry-run, run the explicit index and inspect its output/status:

```powershell
uv run python scripts/index_knowledge.py "D:\My Obsidian Vault"
uv run python scripts/index_knowledge.py --status
```

The indexing command reports scanned/new/changed/unchanged/duplicate/deleted/failed counts, embedding/chunk reuse and generation counts, extraction warnings and failures, and active index content statistics including format counts and approximate chunk-size average/minimum/median/maximum. No body text is printed in the index summary.

## Case file

Keep manually authored cases under ignored `data/knowledge/evaluation/`, for example `real-vault.jsonl`. JSONL means one JSON object per line. `RetrievalEvalCase` fields are validated; expected labels are optional. Do not populate expected sources or sections until a person has confirmed them in the actual index. Unlabeled cases are useful for human review, but Hit@K/MRR is `null` when no ground truth exists.

Example schema only:

```json
{"id":"case-001","query":"a query you selected from a real teaching need","category":"conceptual_topic","source_title":null,"source_id":null,"section":null,"course":null,"tags":[],"expected_source_ids":[],"expected_source_titles":[],"expected_sections":[],"expected_pages":[],"notes":"Add verified expected values after checking the index."}
```

Use a small pilot of roughly 20-30 queries, balanced across exact source, source-plus-section, note title, conceptual topic, course-scoped topic, ambiguous and missing sources, technical phrases, alternate terminology, and actual lecture-building needs. Do not generate this private test set with Qwen.

## Running controlled comparisons

```powershell
uv run python scripts/evaluate_retrieval.py data/knowledge/evaluation/real-vault.jsonl --mode all --compare-neighbors
```

Modes:

- `lexical`: SQLite FTS5/BM25 only; does not call the embedding provider.
- `semantic`: local dense search only.
- `hybrid`: current production hybrid/RRF mode.
- `all`: run lexical, semantic, and hybrid with the same case filters/candidate settings.

`--compare-neighbors` adds hybrid expansion 0 and 1 variants beside the configured hybrid baseline. This does not introduce a reranker. Small supported overrides are `--lexical-candidates`, `--semantic-candidates`, `--fused-candidates`, `--top-k`, `--rrf-constant`, and `--neighbor-expansion {0,1}`. Candidate and top-k constraints are validated by `RetrievalConfig`; production YAML need not be edited for an experiment.

Semantic/hybrid runs use the existing configured local `BAAI/bge-small-en-v1.5` model. Initial model construction/download time is recorded separately from per-query embedding and retrieval timing. No cloud embeddings are used.

Each private `results/<run-id>/` contains:

- `run.json`: timestamp, case version/count, modes, opaque vault ID, index schema/model/chunking metadata, index build timestamp, Git revision where available, config, initialization time, and run duration. It does not include the absolute vault root.
- `cases.jsonl`: exact evaluated case set.
- `results.jsonl`: per-case and per-mode resolution details, lexical/semantic/fused/final candidate diagnostics, boost flags, matches and passages, metrics, stage timings, and assembled `SourceMaterial`.
- `reviews.jsonl`: human annotations, updated by the review CLI.
- `summary.json`: per-mode metrics, mode deltas against hybrid where available, average retrieval-stage timings, and review counts.

Result bundles contain private passages and remain ignored under `data/knowledge/evaluation/`. Avoid copying or committing them.

## Metrics and diagnostics

Where verified expected data exists, the runner reports Hit@1, Hit@3, Hit@5, Hit@K, MRR, source-resolution accuracy, section-resolution accuracy, fail-closed accuracy for missing-source cases, and ambiguity-detection accuracy. Acceptable expected source IDs/titles are treated as alternatives. Neighbor additions do not count as ranked retrieval hits. Cases lacking expected labels are excluded from the corresponding denominator.

Source-resolution method is separately recorded as exact title, normalized title, alias, filename, fuzzy, ambiguous, missing, or unresolved. Section resolution tracks exact, prefix, heading match, and unresolved separately; a correct source does not imply a correct section. The retriever stores source/section warnings and the raw lexical, semantic and fused candidate lists. For each fused candidate it records rank, score, relative provenance and the source/section boost details; course/tag/folder are filters, not currently configured score boosts. No vector values are emitted.

Failure categories are available for human attribution: `source_not_found`, `source_ambiguous`, `section_not_found`, `relevant_source_not_retrieved`, `relevant_chunk_ranked_too_low`, `lexical_noise`, `semantic_noise`, `wrong_course`, `insufficient_context`, `chunk_boundary_problem`, `extraction_problem`, and `metadata_problem`. The system assigns only obvious source/section-resolution failures automatically; relevance and causal diagnoses must be reviewed by a person.

Context assembly diagnostics include candidates, selected chunk IDs/count, merged chunks and source-material count, exact repeated-overlap characters removed, approximate token count, configured Standard budget, unused budget, and excluded chunks. Assembled passages are kept in the private run for sufficiency review.

Standalone query and source inspection are also available:

```powershell
uv run python scripts/retrieve.py "rules of inference" --explain
uv run python scripts/inspect_knowledge_source.py --source-title "A title copied from index status"
uv run python scripts/inspect_knowledge_source.py --relative-path "notes/example.md"
```

The inspector is read-only, previews chunks by default, and accepts a source ID or relative path if a title is ambiguous. `--explain` includes source/section resolution, lexical and semantic candidates, RRF, boosts, neighbors, final results, and context selection; it never emits embedding vectors.

## Human review

Review each run locally:

```powershell
uv run python scripts/review_retrieval.py data/knowledge/evaluation/results/<run-id>
```

The terminal reviewer displays the query, filters, resolution, match rank, relative path, section/page/slide, retrieval ranks, and bounded passage previews. It asks for relevance (`highly_relevant`, `relevant`, `partially_relevant`, `irrelevant`), optional failure category/notes, and a separate assembled-context sufficiency label (`sufficient`, `partially_sufficient`, `insufficient`). It displays the assembled-context preview, does not call an LLM, and persists answers after each case/mode.

Use results to compare source resolution, section resolution, ranked relevance, human relevance, context sufficiency, and latency. Do not optimize latency if it causes a substantial quality loss. Do not tune special cases around one note. Investigate extraction/chunking first with the inspector when the relevant content is absent; distinguish extraction failures from ranking failures. Recommend a production setting only after reviewing the real pilot.

## Privacy and training separation

The vault path must be supplied explicitly to indexing; neither evaluation nor tests search for a vault. The indexer keeps generated state/cache outside the vault. Evaluation case files, queries, matches, passages, and reviews stay local and are ignored by Git. Repository teaching-source discovery rejects all `data/knowledge/` paths; evaluation types and code are under `lecture_slm.knowledge.evaluation`, not `lecture_slm.ingestion`. Never export retrieval evaluation data as SFT, preference, or teaching-candidate data.

No model changes, reranker, vector database, generation spot checks, or fine-tuning are part of the initial retrieval pilot. A limited generation comparison of at most three prompts is a later step, only after humans select a production retrieval configuration.
