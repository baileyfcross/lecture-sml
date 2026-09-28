# Architecture

## Design boundary

This repository trains and evaluates instructional behavior. The separate Obsidian vault/indexing application remains responsible for retrieval and private knowledge. A vault dump is not a training corpus by default.

## Training path

Teaching materials -> dataset processing -> dataset validation -> QLoRA -> evaluation -> merge -> GGUF -> Ollama.

Training is not implemented in v0. The schemas preserve task, provenance, quality tier, and split information so a later pipeline can be reproducible.

## Runtime path

User request + course profile + pedagogy profile + retrieved vault context + recent course context -> Lecture SLM through Ollama -> educational output.

The current runtime implements only the Ollama client and configuration boundary. Course and pedagogy context can be supplied by callers without baking individual courses into the model.

## Reproducibility

A future run should live under `artifacts/runs/<run-id>/` with `config.yaml`, `dataset_manifest.json`, `metrics.json`, `adapter/`, `merged/`, `gguf/`, `eval-baseline.json`, `eval-trained.json`, and `run.json`. It should record model revision, dataset version, Git commit, seed, metrics, export format, and quantization format.
