# Lecture SLM

Lecture SLM is a local foundation for fine-tuning a relatively small language model to reproduce an instructor's teaching style: lecture organization, scaffolding, examples, exercises, tone, and formatting.

The architecture keeps four concerns separate:

- **Fine-tuning** teaches how the instructor teaches.
- **RAG and the Obsidian vault** provide what the instructor knows.
- **Course profiles** describe who is being taught.
- **Course history** describes where the class currently is.

The vault/indexing application and its private content remain outside this repository.

## Status

v0 is a repository, schema, evaluation, and Ollama baseline foundation. It does not train a model, ingest a vault, run a vector database, or provide an automatic pedagogy judge.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Ollama
- A locally available `qwen3.5:9b` Ollama model

Install `uv` using the official instructions for your operating system.

## Setup and checks

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Validate the committed sample dataset:

```powershell
uv run python scripts/validate_dataset.py data/example_dataset.jsonl
```

## Ollama baseline

Pulling a model can download several gigabytes, so run this explicitly:

```powershell
ollama pull qwen3.5:9b
ollama create lecture-slm-baseline -f ollama/Modelfile.baseline
```

Set `OLLAMA_HOST` when the Ollama server is not local:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
uv run python scripts/test_ollama.py
```

Validate the evaluation prompt set and referenced profiles, then run the baseline when ready. Each run is stored in an ignored `evals/results/<run-id>/` directory. Use `--limit 2` for a small smoke evaluation, `--run-dir` to resume, and `--rerun` to explicitly append new attempts for successful prompts.

```powershell
uv run python scripts/validate_eval.py
uv run python scripts/run_baseline_eval.py --limit 2
uv run python scripts/review_eval.py evals/results/<run-id> --reviewer instructor
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for the human-review process and full-run/resume commands. Evaluation prompts remain separate from training data and must never be included automatically.

## Data privacy

Do not commit private vault exports, raw course materials, credentials, model weights, generated embeddings, processed training data, logs, or evaluation outputs. Prefer small, reviewed, provenance-preserving examples over bulk vault dumps.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Training is deferred until the schemas, data review process, and baseline evaluation are trustworthy.
