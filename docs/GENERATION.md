# Generation Architecture

Generation is runtime orchestration over the current Ollama model. It is separate from training, evaluation scoring, and the Obsidian/RAG application. The same runtime can later target an Ollama fine-tuned tag such as `lecture-slm:v1` by changing model configuration only; it has no Hugging Face or training-stack dependency.

## Request inputs

`GenerationRequest` contains a canonical educational task, profile name, authoritative instruction, optional typed course and pedagogy profiles, structured source records, prior topics/course history, output preferences, and metadata. `SourceMaterial` includes source ID, title, section, text, and metadata. Callers provide retrieved materials later; this package does not retrieve, index, embed, or access the Obsidian vault.

Prompt builders keep instruction, course constraints, pedagogy, teaching plan, sources, and prior-course context in separate labeled JSON blocks. The writer is explicitly asked to ground claims in supplied sources and follow, rather than redo, the teaching plan.

## Profiles and routing

`configs/generation/profiles.yaml` contains versioned Quick, Standard, and Deep settings. Model defaults remain in `configs/models/qwen35-9b.yaml`; the generation profile selects a workflow and stage-specific settings without mutating those defaults.

- **Quick:** one Writer request, no Planner; useful for short explanations, rewriting, single slides, and small edits.
- **Standard:** Planner request with thinking enabled and structured JSON output, then a separate Writer request with thinking disabled. This is the default workflow for normal educational artifacts.
- **Deep:** larger Planner budget/context followed by Writer with configurable context tiers. Reviewer support is represented by a protocol, but review is disabled by default and no automatic judge is implemented.

Planner failure is explicit and prevents Writer execution. Writer failure retains the completed plan in `GenerationResult`. No silent Planner-to-Quick fallback occurs. Planner retries are profile-configurable, limited to one retry, and currently configured as zero.

## Teaching plan

`TeachingPlan` is a Pydantic schema designed to flex across lectures, slides, labs, activities, guides, assessments, and explanations. It includes objectives, prerequisites, prior-knowledge connections, sequence steps, concepts, examples, misconceptions, practice, assessment checks, synthesis, source usage, artifact structure, and writer notes. Ollama's native structured-output `format` is used with the plan JSON schema. Malformed JSON/schema output is returned as a distinct failed Planner stage with its raw text retained; it is not silently repaired.

Prompt templates are versioned as `planner-v1` and `writer-v1`, and both versions are saved in result metadata.

## Task budgets and context selection

Each generation profile has a global stage output fallback and optional canonical-task overrides. Writer always uses `think: false`; planning consumes its separate reasoning/output budget. Values are in YAML, not scattered Python constants.

For each stage, context selection estimates assembled prompt tokens using approximately one token per four characters, multiplies the estimate by the configured safety margin, reserves the stage output budget, then chooses the smallest configured context tier that can fit the result. If no tier can accommodate it, the stage fails explicitly instead of silently sending an undersized context. The estimate and chosen tier are recorded. This is a planning heuristic, not tokenizer integration.

Timeouts are independent from context and output caps. Quick writer: 300 seconds; Standard Planner/Writer: 1500/900 seconds; Deep Planner/Writer: 1500/1800 seconds. The timeout bounds a stage request; it does not imply a completion deadline. Qwen3.5 consumed the earlier 512-, 1024-, and 2048-token Planner caps in its thinking field without returning JSON, while the Deep smoke produced a valid plan within its 3072-token cap. Standard and Deep therefore currently use 3072-token Planner caps as generation-profile experiments, not model defaults. At the configured fallback rate, the CLI can estimate stage duration as output tokens divided by tokens/second. That value is labeled approximate and is not used as a deadline.

## Progress and results

The pipeline emits `ProgressEvent` callbacks for preparing, planning, writing, reviewing, complete, and failed states. Events can include elapsed time, generated tokens, observed tokens/second, and a clearly approximate remaining-time estimate. The CLI prints these events without logging source documents.

A `GenerationResult` retains Planner and Writer records separately, including timings, tokens, stop reasons, selected contexts, estimated input sizes, errors, final output, model defaults, effective profile, and prompt versions. Reviewer output is an actionable `ReviewFeedback` shape; it has no arbitrary score field.

Run persistence is opt-in:

```powershell
uv run python scripts/generate.py --profile standard --task lecture --instruction "Create a short introductory lesson about DNS." --save-run
```

Saved development artifacts go under ignored `artifacts/generations/<run-id>/`: `request.json`, `plan.json` when planning ran, `result.json`, and `output.md`. The stage-separated files preserve Planner success if Writer fails, providing the foundation for future resume from a saved plan. Full resume behavior is not implemented yet.

## Training boundary

This runtime does not train models, create adapters, ingest vault content, build embeddings, or implement RAG. Fine-tuning will teach how the instructor teaches; course profiles/context define who is being taught; retrieved sources and course history will supply what is known and where the class currently is. Runtime generation and the later training pipeline remain separate systems.
