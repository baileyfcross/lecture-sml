# Lecture SLM

Lecture SLM is a local foundation for fine-tuning a relatively small language model to reproduce an instructor's teaching style: lecture organization, scaffolding, examples, exercises, tone, and formatting.

The architecture keeps four concerns separate:

- **Fine-tuning** teaches how the instructor teaches.
- **Local knowledge retrieval** provides what the instructor knows.
- **Course profiles** describe who is being taught.
- **Course history** describes where the class currently is.

The vault content remains external and private. Knowledge indexing code lives in this repository, while the generated local index is kept in ignored `data/knowledge/` state. Obsidian is an optional editor only; indexing and inference read the vault as a read-only filesystem directory and do not require the Obsidian application.

## Status

The project includes a local SQLite/FTS5 and FastEmbed knowledge-indexing and retrieval path. A real-vault pilot and retrieval-quality evaluation are still pending. It does not train a model or provide an automatic pedagogy judge.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Ollama
- A locally available `qwen3.5:9b` Ollama model
- FastEmbed downloads its embedding model locally on the first explicit index/retrieval operation; normal tests do not download models.

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
on Windows:
$env:OLLAMA_HOST = "http://localhost:11434"

on Mac:
export OLLAMA_HOST="http://localhost:11434"

For Persistance on Mac:
echo 'export OLLAMA_HOST="http://localhost:11434"' >> ~/.zshrc

Test with script:
uv run python scripts/test_ollama.py
```

Validate the evaluation prompt set and referenced profiles, then run the baseline when ready. Each run is stored in an ignored `evals/results/<run-id>/` directory. Use `--limit 2` for a small smoke evaluation, `--run-dir` to resume, and `--rerun` to explicitly append new attempts for successful prompts.

```powershell
uv run python scripts/validate_eval.py
uv run python scripts/run_baseline_eval.py --limit 2
uv run python scripts/review_eval.py evals/results/<run-id> --reviewer instructor
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for the human-review process and full-run/resume commands. Evaluation prompts remain separate from training data and must never be included automatically.

To characterize context-size and thinking performance before choosing quality-baseline settings, run `uv run python scripts/benchmark_ollama.py`. This does not launch the 26-prompt baseline; see [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for metrics and server diagnostics.

The separate baseline evaluation profile is [configs/evaluation/qwen35-9b-baseline.yaml](configs/evaluation/qwen35-9b-baseline.yaml). Its evaluation-only overrides do not change the model defaults. The eight-prompt profile validation command and interpretation are documented in [docs/EVALUATION.md](docs/EVALUATION.md); the full 26-prompt quality baseline remains a later, explicit run.

## Generation Runtime

The development generation runner supports Quick (Writer only), Standard (Planner -> Writer), and Deep (larger Planner -> Writer, optional reviewer prepared but disabled):

```powershell
uv run python scripts/generate.py --profile quick --task explanation --instruction "Explain DNS using one concrete example."
uv run python scripts/generate.py --profile standard --task lecture --instruction "Create a short introductory lesson on DNS." --course configs/courses/example-course.yaml --pedagogy configs/pedagogy/default.yaml --save-run
uv run python scripts/generate.py --profile deep --task explanation --instruction "Connect DNS to domain names and IP addresses." --source-file path/to/source.md
```

Generation accepts explicit source files and previous-course context. Knowledge retrieval remains opt-in; when enabled, retrieved material is converted to the same `SourceMaterial` contract and joined with any explicit source file. See [docs/GENERATION.md](docs/GENERATION.md) for generation details and [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md) for indexing/retrieval operations.

## Local Web Interface

Lecture SLM includes a lightweight browser UI backed by the same local FastAPI service and `GenerationService` as the CLI. Start the API from the repository root:

```powershell
uv run python scripts/serve.py --model-config configs/models/lecture-slm.yaml
```

The server binds to `127.0.0.1:8000` by default. For frontend development, install the web dependencies once and run Vite in a second terminal:

```powershell
cd web
npm install
npm run dev
```

Open the local address printed by Vite (normally [http://localhost:5173](http://localhost:5173)). Its development server proxies API requests to FastAPI on `127.0.0.1:8000`.

To build the frontend and serve it directly from FastAPI instead:

```powershell
cd web
npm run build
cd ..
uv run python scripts/serve.py --model-config configs/models/lecture-slm.yaml
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The API can still start if the frontend has not been built; the root page will explain how to build it. Swagger remains available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The web interface lets you:

- Choose a generation task and the Quick, Standard, or Deep profile.
- Enter a multiline instruction and optionally enable Vault retrieval with a configurable top-k.
- Follow generation progress and backend timing estimates as they stream.
- View approved output as Markdown and copy it.
- Inspect source counts, grounding decisions, timings, errors, and saved-run paths.
- Distinguish grounding failures (where no output is approved) from API or connection errors.
- See API, model, and knowledge-index readiness.
- Select an optional local Workspace, manage its folders and typed Context/History/Reference items, and explicitly save approved output to Workspace History.

The browser communicates with the Lecture SLM API only; it does not call Ollama directly or store prompts and generated output in browser storage. Only the selected Workspace ID is persisted as a browser preference. Workspace content remains in the separate, Git-ignored Workspace SQLite store; see [docs/GENERATION.md](docs/GENERATION.md) for role boundaries, budgets, API routes, and frontend development details.

## Local Knowledge

Configure `LECTURE_SLM_VAULT_PATH` or provide a vault path explicitly. The default YAML contains no private path. Indexing is never automatic:

```powershell
uv run python scripts/index_knowledge.py "D:\Obsidian Vault" --dry-run
uv run python scripts/index_knowledge.py "D:\Obsidian Vault"
uv run python scripts/retrieve.py "rules of inference" --source-title "Foundations of Computation" --section "1.6"
uv run python scripts/generate.py --profile standard --task lecture --instruction "Create a lecture on rules of inference" --retrieve --source-title "Foundations of Computation" --section "1.6"
```

All generated state stays outside the vault in `data/knowledge/` by default. The indexer never writes to source files. See [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md) for the read-only and privacy guarantees, supported files, configuration, rebuilds, and limitations.

## Data privacy

Do not commit private vault exports, raw course materials, credentials, model weights, generated embeddings, processed training data, logs, or evaluation outputs. Prefer small, reviewed, provenance-preserving examples over bulk vault dumps.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Training is deferred until the schemas, data review process, and baseline evaluation are trustworthy.

## Teaching Material Ingestion

The deterministic ingestion pipeline turns explicitly supplied `.pptx`, `.docx`, `.pdf`, `.md`, and `.txt` files into private, reviewable candidates. It does not scan the vault, call Ollama, rewrite source text, or approve material automatically. See [docs/INGESTION.md](docs/INGESTION.md) for the full contract.

```powershell
uv run python scripts/import_teaching_materials.py "C:\Teaching Materials" --dry-run
uv run python scripts/import_teaching_materials.py "C:\Teaching Materials"
uv run python scripts/validate_candidates.py
uv run python scripts/review_dataset.py --status pending
uv run python scripts/dataset_stats.py
uv run python scripts/export_dataset.py --version 0.1.0
```


## Example Retrieval CLI Prompts

```bash
uv run python scripts/retrieve.py "predicate logic" --explain

uv run python scripts/retrieve.py "rules of inference" --explain
```