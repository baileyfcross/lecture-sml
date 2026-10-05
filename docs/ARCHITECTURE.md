# Architecture

## Design boundary

This repository separates instructional behavior, local knowledge, course profiles, and course history. Vault content remains external and private; the repository contains only the indexing/retrieval implementation. A vault dump or generated knowledge index is never a training corpus by default.

## Teaching material path

Explicit source path -> deterministic file discovery -> type-specific extraction -> normalized document -> conservative structural classification -> pending candidate -> human review -> approved model-independent dataset record.

The ingestion subsystem is local and separate from runtime generation. It preserves source hashes, source versions, section/page/slide locations, extraction warnings, authorship metadata, and review history. Imported instructor material starts unapproved at Tier B. Evaluation prompts, benchmark outputs, and generated runtime artifacts are excluded from discovery/export.

See `docs/INGESTION.md` for commands and the manifest/candidate contracts.

## Training path

Approved teaching candidates -> dataset processing -> dataset validation -> QLoRA -> evaluation -> merge -> GGUF -> Ollama.

Training is not implemented in v0. The schemas preserve task, provenance, quality tier, and split information so a later pipeline can be reproducible.

## Runtime path

Vault filesystem (explicit, read-only root) -> Knowledge Index (shared deterministic extractors -> section-aware chunks -> local FastEmbed vectors + SQLite FTS5) -> hybrid retrieval/source resolution -> Context Assembler -> existing `SourceMaterial` + explicit source material -> `GenerationRequest` -> Quick Writer OR Planner -> structured TeachingPlan -> Writer -> educational output.

The runtime implementation is in `src/lecture_slm/generation/`; the local knowledge path is in `src/lecture_slm/knowledge/`. Retrieval is opt-in and happens before the existing router. Planner and Writer receive only the established `SourceMaterial` objects and do not know whether material was supplied manually or retrieved. Quick skips planning; Standard and Deep use separate Planner and Writer Ollama requests. Deep reviewer support is an interface only and disabled by default.

Knowledge records and teaching-material review candidates are separate stores and workflows. Knowledge content is not automatically approved, exported, or discovered as training data. `PreviousCourseContext` remains caller-provided course history; a vault note is not assumed to have been taught. Indexing and retrieval do not require Ollama or an active Obsidian process.

The vault is read-only. Database, local embedding cache, and model files live under the configured knowledge data directory outside the vault. The SQLite index stores source hashes, relative paths, extraction warnings, chunks, metadata, FTS5 text, and float32 embeddings. An embedding/chunking version mismatch fails clearly and requires an explicit rebuild.

Model defaults live in `configs/models/`; workflow settings live separately in `configs/generation/profiles.yaml`. Context selection uses a conservative character/token estimate and picks the smallest configured tier that accommodates estimated input plus output budget and safety margin. Generation results record stage-specific timings, tokens, context choices, errors, and prompt versions. See `docs/GENERATION.md` for the complete runtime contract.

## Reproducibility

A future run should live under `artifacts/runs/<run-id>/` with `config.yaml`, `dataset_manifest.json`, `metrics.json`, `adapter/`, `merged/`, `gguf/`, `eval-baseline.json`, `eval-trained.json`, and `run.json`. It should record model revision, dataset version, Git commit, seed, metrics, export format, and quantization format.
