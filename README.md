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

Install `uv` using the official instructions for your operating system. On Windows PowerShell, the project memory for this environment recommends using `npm.cmd` when npm is needed because the PowerShell shim may be blocked.

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

Run the unscored baseline evaluation. Responses are written to the ignored `evaluation-results/` directory:

```powershell
uv run python scripts/run_baseline_eval.py
```

Evaluation prompts are intentionally separate from training data. Never automatically include evaluation examples in training data.

## Data privacy

Do not commit private vault exports, raw course materials, credentials, model weights, generated embeddings, processed training data, logs, or evaluation outputs. Prefer small, reviewed, provenance-preserving examples over bulk vault dumps.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Training is deferred until the schemas, data review process, and baseline evaluation are trustworthy.
