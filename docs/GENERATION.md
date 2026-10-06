# Generation Architecture

Generation is runtime orchestration over the current Ollama model. It is separate from training and evaluation scoring. Optional local knowledge retrieval is performed by the CLI before the existing generation router; the router and stage implementations do not access the vault. The same runtime can later target an Ollama fine-tuned tag such as `lecture-slm:v1` by changing model configuration only; it has no Hugging Face or training-stack dependency.

## Request inputs

`GenerationRequest` contains a canonical educational task, profile name, authoritative instruction, optional typed course and pedagogy profiles, structured source records, prior topics/course history, output preferences, and metadata. `SourceMaterial` includes source ID, title, section, text, and metadata. Explicit source files and opt-in retrieved materials are both appended to the same source list; retrieval provenance is retained in source metadata. See `docs/KNOWLEDGE.md` for the retrieval implementation.

Prompt builders keep instruction, course constraints, pedagogy, teaching plan, sources, and prior-course context in separate labeled JSON blocks. The writer is explicitly asked to ground claims in supplied sources and follow, rather than redo, the teaching plan.

When source material is supplied, the writer grounds not only definitions and technical claims but also framing, motivation, conclusions, significance, and claims about importance or a concept's broader role. It must not infer historical reasons from a capability, or inflate a supported point into an unsupported claim of importance, essential status, or a broader connection. Pedagogical examples, analogies, practice, transitions, and learner-focused explanations remain allowed; source-free requests retain normal access to model knowledge.

## Profiles and routing

`configs/generation/profiles.yaml` contains versioned Quick, Standard, and Deep settings. Model defaults remain in `configs/models/qwen35-9b.yaml`; the generation profile selects a workflow and stage-specific settings without mutating those defaults.

- **Quick:** one Writer request, no Planner; useful for short explanations, rewriting, single slides, and small edits.
- **Standard:** concise task-specific Planner request with thinking disabled and temperature 0, then a separate Writer request with thinking disabled. This is the recommended workflow for normal educational artifacts.
- **Deep:** larger Planner budget/context followed by Writer with configurable context tiers. Reviewer support is represented by a protocol, but review is disabled by default and no automatic judge is implemented.

Planner failure is explicit and prevents Writer execution. Writer failure retains the completed plan in `GenerationResult`. No silent Planner-to-Quick fallback occurs. Planner retries are profile-configurable, limited to one retry, and currently configured as zero.

## Teaching plan

Planning uses a shared `TeachingPlanBase` plus task-specific `ExplanationPlan`, `LecturePlan`, `SlidesPlan`, `LabPlan`, `ActivityPlan`, `InstructorGuidePlan`, and `AssessmentPlan` schemas. A typed registry maps canonical task enums to schemas; homework shares `AssessmentPlan` rather than adding a task type. Short explanations therefore require only a concept, assumptions, explanation sequence, example, misconceptions, and check, while lecture plans include objectives, sequence, practice, synthesis, timing, and source coverage. Ollama's native structured-output `format` uses the selected task schema. Malformed JSON/schema output is a distinct failed Planner stage with raw response preserved; missing fields are not silently supplied.

Standard and Deep planning are separate: Standard uses concise plan-only wording, `think: false`, and deterministic Planner temperature 0. Deep uses the model's thinking capability but persists only the validated TeachingPlan. Planner versions are `planner-standard-v3` for concise Standard planning and `planner-v2` for Deep. Ollama thinking character count is captured when available; raw thinking text is never passed to Writer or persisted.

When source material is supplied, the Planner limits factual topics to what those sources support, while retaining freedom to add pedagogical structure and illustrative examples. The Writer treats source scope as higher priority than the teaching plan and omits or narrows unsupported factual plan requirements. Without supplied sources, both stages retain normal model-knowledge behavior.

Planner templates are versioned as `planner-standard-v3` for concise Standard planning and `planner-v2` for Deep planning. The Writer template is `writer-v7`. The selected versions are saved in result metadata.

## Task budgets and context selection

Each generation profile has a global stage output fallback and optional canonical-task overrides. Writer always uses `think: false`; planning consumes its separate reasoning/output budget. Values are in YAML, not scattered Python constants.

For each stage, context selection estimates assembled prompt tokens using approximately one token per four characters, multiplies the estimate by the configured safety margin, reserves the stage output budget, then chooses the smallest configured context tier that can fit the result. If no tier can accommodate it, the stage fails explicitly instead of silently sending an undersized context. The estimate and chosen tier are recorded. This is a planning heuristic, not tokenizer integration.

Timeouts and task-specific output budgets are independently configured in the versioned profiles YAML; they are not model defaults.

Stage-time estimates use the task-specific output budget. The first model stage uses the configured fallback generation speed; later stages may use an observed generation rate from an earlier successful stage in the same request. Estimates remain approximate and are not deadlines.

## Progress and results

The pipeline emits `ProgressEvent` callbacks for preparing, planning, writing, reviewing, complete, and failed states. Events can include elapsed time, generated tokens, observed tokens/second, and a clearly approximate remaining-time estimate. The CLI prints these events without logging source documents.

A `GenerationResult` retains Planner and Writer records separately, including timings, tokens, stop reasons, selected contexts, estimated input sizes, errors, final output, model defaults, effective profile, and prompt versions. Reviewer output is an actionable `ReviewFeedback` shape; it has no arbitrary score field.

Run persistence is opt-in:

```powershell
uv run python scripts/generate.py --profile standard --task lecture --instruction "Create a short introductory lesson about DNS." --save-run
```

Saved development artifacts go under ignored `artifacts/generations/<run-id>/`. A run includes `request.json`, the complete `sources.json` and readable `sources.md` for material actually assembled into the request, `result.json`, `diagnostics.md`, and `output.md` when available. Retrieval runs also include `retrieval.json`; stages that ran include their exact prompt snapshots (`planner_prompt.json` and/or `writer_prompt.json`), and a validated Planner result is retained in `plan.json`. These opt-in saved runs are intended for development diagnostics, retrieval and grounding inspection, reproducibility, and investigating whether a questionable claim came from retrieved context or model generation. The stage-separated files preserve Planner success if Writer fails, providing the foundation for future resume from a saved plan. Full resume behavior is not implemented yet.

## Training boundary

This runtime does not train models or create adapters. Fine-tuning teaches how the instructor teaches; course profiles/context define who is being taught; retrieved sources supply factual knowledge; and caller-provided previous-course context describes where the class currently is. Knowledge indexing/retrieval and the later training pipeline remain separate systems. Retrieval is opt-in with `--retrieve`; existing commands and explicit `--source-file` behavior remain available without it.
