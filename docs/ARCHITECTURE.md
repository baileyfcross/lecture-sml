# Architecture

## Design boundary

This repository trains and evaluates instructional behavior. The separate Obsidian vault/indexing application remains responsible for retrieval and private knowledge. A vault dump is not a training corpus by default.

## Training path

Teaching materials -> dataset processing -> dataset validation -> QLoRA -> evaluation -> merge -> GGUF -> Ollama.

Training is not implemented in v0. The schemas preserve task, provenance, quality tier, and split information so a later pipeline can be reproducible.

## Runtime path

User request + course profile + pedagogy profile + caller-supplied source material + caller-supplied recent course context -> Quick Writer OR Planner -> structured TeachingPlan -> Writer -> educational output.

The runtime implementation is in `src/lecture_slm/generation/`. Quick skips planning; Standard and Deep use separate Planner and Writer Ollama requests. Deep reviewer support is an interface only and disabled by default. Course/source/history content is supplied by the caller. The Obsidian vault and indexing application remain separate; no retrieval or integration is implemented here.

Model defaults live in `configs/models/`; workflow settings live separately in `configs/generation/profiles.yaml`. Context selection uses a conservative character/token estimate and picks the smallest configured tier that accommodates estimated input plus output budget and safety margin. Generation results record stage-specific timings, tokens, context choices, errors, and prompt versions. See `docs/GENERATION.md` for the complete runtime contract.

## Reproducibility

A future run should live under `artifacts/runs/<run-id>/` with `config.yaml`, `dataset_manifest.json`, `metrics.json`, `adapter/`, `merged/`, `gguf/`, `eval-baseline.json`, `eval-trained.json`, and `run.json`. It should record model revision, dataset version, Git commit, seed, metrics, export format, and quantization format.
