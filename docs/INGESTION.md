# Teaching Material Ingestion

This phase converts explicitly supplied instructor files into deterministic, reviewable candidate examples. It does not train a model. Imported material is never automatically approved for training.

## Scope and Privacy

Supported formats are `.pptx`, `.docx`, `.pdf`, `.md`, and `.txt`. Extraction is local and deterministic. There is no Ollama call, cloud upload, OCR, link following, embedding, vector database, vault scan, or synthetic generation. PDFs with little extractable text receive warnings and require future manual/OCR handling.

Only the file or directory passed to `scripts/import_teaching_materials.py` is scanned. Common generated/system directories, including `.git`, `.venv`, `node_modules`, `evals`, and `artifacts`, are ignored. Evaluation prompts, benchmark results, and runtime generation artifacts are not eligible sources.

Private state is written below ignored `data/ingestion/`:

- `manifests/sources.json` records every discovered source/version.
- `normalized/<source-id>.json` stores normalized extracted documents.
- `candidates.json` stores the durable review queue.
- `data/processed/` stores explicit approved dataset exports.

## Identity and Incremental Import

The original source bytes are SHA-256 hashed. A stable content identity is `src-<sha256>`, rather than a random UUID or filename. A changed file receives a new content identity and a new source version with `previous_source_id`; the old record is retained. Byte-identical files retain both paths in the manifest, point to one canonical content identity, and do not create duplicate candidates.

An unchanged successfully extracted path is reused without re-extraction. `--force-reextract` is available when parser behavior or metadata requires a refresh. Source paths in manifests are relative to the explicitly supplied import root.

## Normalization

All extractors return `NormalizedDocument` and `NormalizedSection` models. Slide boundaries, headings, paragraphs, list items, tables, page numbers, hierarchy, and source locations are retained where the parser exposes them. The original wording is not corrected, summarized, expanded, or rewritten. The candidate formatter only adds stable structural labels such as `Slide 1`.

PowerPoint uses `python-pptx`, Word uses `python-docx`, and PDF uses `pypdf`. Markdown and text use UTF-8 standard-library parsing. Screenshot meaning, visual layout, embedded images, and OCR are intentionally out of scope.

## Candidates and Provenance

Deterministic filename/title rules classify strong artifact signals as lecture, slides, lab, activity, instructor guide, assessment, homework, or explanation. Ambiguous material remains unknown and does not produce a guessed candidate. Course and level are optional until review.

A candidate contains source IDs and source locations, reconstructed-instruction metadata, the preserved artifact, confidence, authorship, quality tier, review state, and status history. Imported instructor material starts as:

- `review_status: pending`
- `quality_tier: B`
- `authorship: unknown`
- `instruction_source: reconstructed_instruction`

Approval is an explicit review action. Approval promotes instructor-created/unknown candidates to Tier A, or AI-assisted human-reviewed candidates to Tier C. Tier D is reserved for synthetic/unreviewed material. Rejections and notes remain in history.

## Commands

Dry run a supplied folder:

```powershell
uv run python scripts/import_teaching_materials.py "C:\Teaching Materials" --dry-run
```

Import it locally, optionally recording course and level:

```powershell
uv run python scripts/import_teaching_materials.py "C:\Teaching Materials" --course cis101 --level introductory
```

Validate candidate schema, source references, quality consistency, and leakage safeguards:

```powershell
uv run python scripts/validate_candidates.py
```

Review pending candidates. Entering no action skips; approval requires the explicit `a` action. The noninteractive form is useful for scripted local review tests:

```powershell
uv run python scripts/review_dataset.py --status pending
uv run python scripts/review_dataset.py --candidate-id <id> --action a --reviewer instructor
```

Inspect pending or approved statistics:

```powershell
uv run python scripts/dataset_stats.py
uv run python scripts/dataset_stats.py --approved data/processed/approved-sft-v0.1.0.jsonl
```

Export only explicitly approved candidates:

```powershell
uv run python scripts/export_dataset.py --version 0.1.0
```

The export is model-independent JSONL and is validated through the existing `DatasetExample` schema. It creates a versioned dataset manifest containing counts, task/course/tier distributions, candidate IDs, source hashes, schema version, exporter version, and Git commit when available. Existing versions are never overwritten without `--overwrite`.

## Future Work

A reviewed candidate can later become an SFT record. Future preference data may relate an AI draft, instructor correction, and approved output, but DPO examples are not created here. Chat-template tokenization, train/validation splitting, QLoRA, RAG, embeddings, and vault-wide import remain later phases.
