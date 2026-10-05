# Local Knowledge Index and Retrieval

## Purpose and boundaries

The knowledge subsystem makes an explicitly configured filesystem directory (for example, an Obsidian vault) a local factual reference library for Lecture SLM. It is independent of the Obsidian application, plugins, APIs, local Obsidian ports, and any running vault process. Obsidian may remain the editor, but indexing and inference operate only on files.

Knowledge answers **what the instructor knows**. Fine-tuning teaches **how the instructor teaches**. Course profiles describe **who is being taught**, while `PreviousCourseContext` describes **where the class currently is**. Knowledge indexing never builds teaching candidates, approves content, exports SFT data, or changes model weights.

## Privacy and read-only behavior

- Only an explicitly supplied root, `vault_path` setting, or `LECTURE_SLM_VAULT_PATH` environment variable is used. No implicit directory scan occurs.
- Supported source files are read only. The indexer never writes metadata or generated state into the vault, follows external links, executes plugins, or contacts Obsidian.
- The default generated index and local model cache are under `data/knowledge/`, outside the configured vault. The indexer rejects a data directory nested inside the vault.
- No cloud embeddings, telemetry, uploads, Ollama calls, or remote vector database are used. FastEmbed downloads its ONNX model only when the user explicitly runs a non-dry indexing or retrieval command; the model is cached locally.
- Tests use synthetic files and deterministic fake embeddings. Do not commit the real vault, local database, vectors, or model files.

## Supported sources and extraction

The indexer supports `.md`, `.pdf`, `.txt`, `.docx`, and `.pptx`. It reuses the deterministic extractors in `src/lecture_slm/ingestion/extractors/` and consumes their `NormalizedDocument` and `NormalizedSection` results directly. It does not invoke the teaching-material candidate builder.

Markdown notes may contribute safe YAML frontmatter, tags, aliases, wikilinks (`[[target]]`, `[[target|alias]]`), embed-link relationships (`![[target]]`), note/folder path, and heading hierarchy. Unknown YAML fields are retained as metadata. Malformed YAML is recorded as a warning. Plugins and Dataview are not interpreted; links are metadata only and are not recursively inlined.

PDF extraction retains page provenance and current extractor warnings. Image-only PDFs are not OCR'd. PPTX slide and DOCX structure follow the shared normalized representation; the index does not invent PDF headings that extraction did not identify.

## Index storage and versions

`data/knowledge/knowledge.sqlite` is the primary store. It contains independent knowledge source/path records, chunks, an embedding cache, FTS5, and schema/index metadata. A generated vault ID is stored locally, not in the vault. Source IDs use the shared SHA-256 content identity; exact byte-identical sources share chunks and retain alternate relative paths.

Indexing is incremental. An unchanged path/hash skips extraction and embedding; changed bytes are extracted and chunked again; existing content-addressed embeddings are reused when their cache key matches model, FastEmbed version, and normalized chunk text. New, changed, unchanged, deleted, failed, reused-chunk, and embedding counts are reported. Failed sources are recorded and retried by a later indexing run. Per-source SQLite transactions ensure a failure does not discard already indexed sources.

The index records schema version, embedding model/provider version, and chunking-strategy version. Schema version 2 adds deterministic chunk roles; chunking version 2 assigns those roles during extraction. Existing schema-version-1 indexes must be rebuilt rather than migrated in place. Incompatible settings fail clearly; `--rebuild` is explicit and resets only the generated index, never source files. Use separate data directories for different vault roots.

## Chunking, embeddings, and search

Chunks are made from existing normalized sections, not arbitrary whole-document windows. Heading/page/slide boundaries are preserved; related small sections remain intact; large sections are split to a target of 450 estimated tokens, maximum 650, with 60-token overlap by default. Estimates reuse generation's conservative character/token estimator. Chunk IDs and previous/next links are deterministic. PDFs retain page numbers and PPTX chunks retain slide numbers.

FastEmbed is behind an embedding-provider protocol. The default model is `BAAI/bge-small-en-v1.5`; query and passage APIs are used separately. Vectors are float32 data in SQLite and cosine similarity is computed locally with NumPy. The vector dimension comes from FastEmbed's model metadata rather than a hard-coded constant.

SQLite FTS5/BM25 provides exact lexical matching over title, section, text, tags, and path. Dense candidates and lexical candidates are fused with Reciprocal Rank Fusion; raw BM25 and cosine scores are not added together. Exact source and section resolution happens before ranking, and transparent title/section boosts are recorded with rank diagnostics. Filters support source, title, section, course, tags, document type, folder, and page. Optional neighbor expansion is bounded. A reranker interface is prepared, but no reranker model is enabled by default.

Each chunk is classified deterministically as `content`, `reference`, `metadata`, or `navigation`. Only content chunks are embedded and eligible for factual retrieval by default; reference lists, administrative blocks, and Obsidian navigation remain indexed for inspection and can be requested explicitly with `scripts/retrieve.py --include-role`. Inferred document-title mentions are soft signals unless the query is a high-confidence document lookup; explicitly supplied source filters remain hard constraints. A small query-term-coverage boost is reported alongside the unchanged RRF ranking signals. Neighbor expansion is content-only by default and remains within the same page or slide.

## Retrieval quality evaluation

Retrieval-evaluation tooling is separate from SFT evaluation prompts and generation. Author private JSONL cases manually, one `RetrievalEvalCase` per query. Expected sources, sections, and pages are optional; only add answer keys after verifying the actual index. Cases without relevance labels produce no Hit@K/MRR score rather than an assumed miss. Start with approximately 20-30 balanced real queries.

See [RETRIEVAL_EVALUATION.md](./RETRIEVAL_EVALUATION.md) for the case schema, metric definitions, diagnostics, privacy boundaries, and full review workflow.

Dry-run before an explicit real index operation:

```powershell
uv run python scripts/index_knowledge.py "D:\Obsidian Vault" --dry-run
```

The dry run counts supported/unsupported files, reports ignored directories and exact-byte duplicate candidates, estimates workload by supported-file count and byte volume, and shows the target data directory, embedding model, and whether the configured model-cache directory contains files. It hashes supported files for duplicate detection but does not extract content, initialize/download FastEmbed, or write state. It does not modify the vault.

After reviewing the report, explicitly index and inspect status:

```powershell
# Only for an existing schema-version-1 index:
uv run python scripts/index_knowledge.py "D:\Obsidian Vault" --rebuild
# For a new index or a normal incremental update:
uv run python scripts/index_knowledge.py "D:\Obsidian Vault"
uv run python scripts/index_knowledge.py --status
```

Use `--rebuild` once after upgrading a schema-version-1 index to schema version 2; it recreates only the generated local index and embeds content-role chunks. Index output combines run counters with active format and role counts, approximate chunk-size distribution, warnings, and failed sources. Successful indexing alone says nothing about retrieval quality.

Create the private case file at `data/knowledge/evaluation/real-vault.jsonl`. Example shape (replace the query with one you selected manually; fill expected values only after checking the indexed source):

```json
{"id":"case-001","query":"your manually chosen retrieval query","category":"conceptual_topic","source_title":null,"section":null,"course":null,"tags":[],"expected_source_ids":[],"expected_source_titles":[],"expected_sections":[],"expected_pages":[],"notes":"Add the answer key only after inspecting the index."}
```

Compare retrieval modes and bounded neighbor expansion:

```powershell
uv run python scripts/evaluate_retrieval.py data/knowledge/evaluation/real-vault.jsonl --mode all --compare-neighbors
uv run python scripts/review_retrieval.py data/knowledge/evaluation/results/<run-id>
```

`all` runs lexical-only, semantic-only, and hybrid RRF. `--compare-neighbors` adds hybrid expansion 0 and 1. Candidate counts, top-k, RRF constant, and neighbor expansion can be overridden. Lexical-only evaluation does not initialize embeddings; semantic and hybrid use the configured local FastEmbed model. Run files record an opaque vault ID (never its absolute root), index build metadata, Git revision if available, case version, settings, initialization time, per-stage latency, candidates/final diagnostics, and assembled context. Results contain private passage text and remain under ignored `data/knowledge/evaluation/`.

Metrics include Hit@1/3/5/K, MRR, source/section resolution accuracy, fail-closed accuracy, and ambiguity-detection accuracy where expected values are present. Source- and section-resolution methods are recorded separately. Human review records passage relevance, failure categories, assembled-context sufficiency, and notes; there is no LLM judge and no generation call. Unknown ground truth is reported as `null`. Compare mode metrics/deltas with human relevance, context sufficiency, source/section resolution, and latency before selecting settings.

Inspect query diagnostics and source extraction without generation:

```powershell
uv run python scripts/retrieve.py "rules of inference" --explain
uv run python scripts/inspect_knowledge_source.py --source-title "A title from the index"
```

Explain mode shows source/section resolution, lexical/semantic/fused candidates, boosts, neighbor additions, and assembled-context selection. Source inspection is read-only, uses previews by default, and supports source ID or relative path if a title is ambiguous.

The first evaluation pass is retrieval plus context assembly only: do not run Quick/Standard/Deep generation, fine-tune, or change embedding models. Evaluation schemas live in `lecture_slm.knowledge.evaluation`; repository teaching-material ingestion excludes `data/knowledge/`, so private queries/results do not become training candidates.

## Commands

Dry run reports eligible/unsupported files, ignored directories, duplicate candidates, approximate workload, configured output/model, and apparent local cache status without embedding or writing index state:

```powershell
uv run python scripts/index_knowledge.py "D:\Obsidian Vault" --dry-run
```

Index, inspect status, rebuild, or retrieve:

```powershell
uv run python scripts/index_knowledge.py "D:\Obsidian Vault"
uv run python scripts/index_knowledge.py --status
uv run python scripts/index_knowledge.py "D:\Obsidian Vault" --rebuild
uv run python scripts/retrieve.py "rules of inference" --source-title "Foundations of Computation" --section "1.6" --top-k 8
```

Generation retrieval is opt-in. Explicit `--source-file` may be combined with retrieved knowledge:

```powershell
uv run python scripts/generate.py --profile standard --task lecture --instruction "Create a lecture on rules of inference" --retrieve --source-title "Foundations of Computation" --section "1.6" --knowledge-course CSC220 --tag logic
```

The `--course` option remains the existing course-profile file. Use `--knowledge-course` for the retrieval metadata filter.

## Configuration

See `configs/knowledge/default.yaml`. The vault path is unset by default. `LECTURE_SLM_VAULT_PATH` overrides YAML; an explicit CLI path overrides both. Configuration validates token limits, overlap, candidate pool sizes, final result limits, and context budgets. Quick, Standard, and Deep source-context budgets default to 1,500, 3,500, and 7,000 estimated tokens.

## Context assembly and provenance

Retrieval results are converted to the existing generation `SourceMaterial` model. Context assembly selects matches under the profile's explicit source-token budget, groups adjacent chunks in source order, removes only exact repeated overlap, and preserves the original source wording. Metadata retains relative path, source/hash/version, chunk IDs, headings, page/slide numbers, and lexical/semantic/fused diagnostics. No LLM summarizes retrieved text. Planner and Writer receive only the existing `SourceMaterial` list and remain unaware of its origin.

## Limitations and troubleshooting

- Retrieval quality is not yet calibrated against a real-vault pilot; inspect exact source/section diagnostics and warnings.
- PDF section headings are only available when extraction exposes them; no heading is guessed.
- There is no OCR, semantic duplicate detection, graph UI, external vector database, cross-encoder implementation, or automatic citation renderer.
- To change embedding models or chunking behavior, use `--rebuild`; incompatible indexes are not silently mixed.
- If a note fails extraction, the index command reports its path, exception type, message, and retry eligibility. Correct the source/dependency issue and rerun indexing.
- If FTS5 is unavailable in the local SQLite build, status reports it and lexical retrieval fails rather than substituting a homemade scorer.
- Ollama is required only by generation, not for indexing or retrieval.
