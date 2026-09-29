# Generation Architecture

Generation is runtime orchestration over the current Ollama model. It is separate from training, evaluation scoring, and the Obsidian/RAG application. The same runtime can later target an Ollama fine-tuned tag such as `lecture-slm:v1` by changing model configuration only; it has no Hugging Face or training-stack dependency.

## Request inputs

`GenerationRequest` contains a canonical educational task, profile name, authoritative instruction, optional typed course and pedagogy profiles, structured source records, prior topics/course history, output preferences, and metadata. `SourceMaterial` includes source ID, title, section, text, and metadata. Callers provide retrieved materials later; this package does not retrieve, index, embed, or access the Obsidian vault.

Prompt builders keep instruction, course constraints, pedagogy, teaching plan, sources, and prior-course context in separate labeled JSON blocks. The writer is explicitly asked to ground claims in supplied sources and follow, rather than redo, the teaching plan.

## Profiles and routing

`configs/generation/profiles.yaml` contains versioned Quick, Standard, and Deep settings. Model defaults remain in `configs/models/qwen35-9b.yaml`; the generation profile selects a workflow and stage-specific settings without mutating those defaults.

- **Quick:** one Writer request, no Planner; useful for short explanations, rewriting, single slides, and small edits.
- **Standard:** concise task-specific Planner request with thinking disabled and temperature 0, then a separate Writer request with thinking disabled. This is the recommended workflow for normal educational artifacts.
- **Deep:** larger Planner budget/context followed by Writer with configurable context tiers. Reviewer support is represented by a protocol, but review is disabled by default and no automatic judge is implemented.

Planner failure is explicit and prevents Writer execution. Writer failure retains the completed plan in `GenerationResult`. No silent Planner-to-Quick fallback occurs. Planner retries are profile-configurable, limited to one retry, and currently configured as zero.

## Teaching plan

Planning uses a shared `TeachingPlanBase` plus task-specific `ExplanationPlan`, `LecturePlan`, `SlidesPlan`, `LabPlan`, `ActivityPlan`, `InstructorGuidePlan`, and `AssessmentPlan` schemas. A typed registry maps canonical task enums to schemas; homework shares `AssessmentPlan` rather than adding a task type. Short explanations therefore require only a concept, assumptions, explanation sequence, example, misconceptions, and check, while lecture plans include objectives, sequence, practice, synthesis, timing, and source coverage. Ollama's native structured-output `format` uses the selected task schema. Malformed JSON/schema output is a distinct failed Planner stage with raw response preserved; missing fields are not silently supplied.

Standard and Deep planning are separate: Standard uses concise plan-only wording, `think: false`, and deterministic Planner temperature 0. Deep uses the model's thinking capability but persists only the validated TeachingPlan. Planner versions are `planner-standard-v2` for concise Standard planning and `planner-v1` for Deep. Ollama thinking character count is captured when available; raw thinking text is never passed to Writer or persisted.

Planner templates are versioned as `planner-standard-v2` for concise Standard planning and `planner-v1` for Deep planning. The Writer template is `writer-v1`. The selected versions are saved in result metadata.

## Task budgets and context selection

Each generation profile has a global stage output fallback and optional canonical-task overrides. Writer always uses `think: false`; planning consumes its separate reasoning/output budget. Values are in YAML, not scattered Python constants.

For each stage, context selection estimates assembled prompt tokens using approximately one token per four characters, multiplies the estimate by the configured safety margin, reserves the stage output budget, then chooses the smallest configured context tier that can fit the result. If no tier can accommodate it, the stage fails explicitly instead of silently sending an undersized context. The estimate and chosen tier are recorded. This is a planning heuristic, not tokenizer integration.

Timeouts are independent from context and output caps. Quick Writer: 300 seconds; Standard Planner/Writer: 480/900 seconds; Deep Planner/Writer: 1500/1800 seconds. Standard task-specific Planner caps are 384 tokens for explanations, 768 for slides, activities, instructor guides, assessments, and homework, 1024 for labs, and 1536 for lectures. Deep keeps larger per-task planning caps up to 3072 for lectures. Standard Writer explanation output is 768 tokens; other Writer budgets remain independently configured. These are generation-profile settings, not model defaults. At the configured fallback rate, the CLI can estimate stage duration as output tokens divided by tokens/second. That value is labeled approximate and is not used as a deadline.

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
