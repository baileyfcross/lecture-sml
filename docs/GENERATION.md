# Generation Architecture

Generation is runtime orchestration over the current Ollama model. It is separate from training and evaluation scoring. Optional local knowledge retrieval is performed by the CLI before the existing generation router; the router and stage implementations do not access the vault. The same runtime can later target an Ollama fine-tuned tag such as `lecture-slm:v1` by changing model configuration only; it has no Hugging Face or training-stack dependency.

## Request inputs

`GenerationRequest` contains a canonical educational task, profile name, authoritative instruction, optional typed course and pedagogy profiles, structured source records, prior topics/course history, output preferences, and metadata. `SourceMaterial` includes source ID, title, section, text, and metadata. Explicit source files and opt-in retrieved materials are both appended to the same source list; retrieval provenance is retained in source metadata. See `docs/KNOWLEDGE.md` for the retrieval implementation.

Prompt builders keep instruction, course constraints, pedagogy, teaching plan, sources, and prior-course context in separate labeled JSON blocks. The writer is explicitly asked to ground claims in supplied sources and follow, rather than redo, the teaching plan.

When source material is supplied, the writer grounds not only definitions and technical claims but also framing, motivation, conclusions, significance, and claims about importance or a concept's broader role. It must not infer historical reasons from a capability, or inflate a supported point into an unsupported claim of importance, essential status, or a broader connection. Pedagogical examples, analogies, practice, transitions, and learner-focused explanations remain allowed; source-free requests retain normal access to model knowledge.

## Profiles and routing

`configs/generation/profiles.yaml` contains versioned Quick, Standard, and Deep settings. Model defaults remain in `configs/models/qwen35-9b.yaml`; the generation profile selects a workflow and stage-specific settings without mutating those defaults.

When Vault retrieval is enabled, the original instruction remains unchanged for planning and generation, while a deterministic canonical query strips recognized leading request scaffolding such as “Can you explain,” “Create a lecture about,” and “What is.” The resulting query is used consistently by lexical and semantic retrieval and query-term coverage; retrieval diagnostics preserve both the original instruction and canonical query. No general stopword removal or additional retrieval pass is performed.

- **Quick:** one Writer request, no Planner; useful for short explanations, rewriting, single slides, and small edits.
- **Standard:** concise task-specific Planner request with thinking disabled and temperature 0, then a separate Writer request with thinking disabled. This is the recommended workflow for normal educational artifacts.
- **Deep:** larger Planner budget/context followed by Writer with configurable context tiers. Like Standard, sourced Deep requests receive a bounded source-grounding review and, only when needed, one revision followed by a final review.

Planner failure is explicit and prevents Writer execution. Writer failure retains the completed plan in `GenerationResult`. No silent Planner-to-Quick fallback occurs. Planner retries are profile-configurable, limited to one retry, and currently configured as zero.

## Teaching plan

Planning uses a shared `TeachingPlanBase` plus task-specific `ExplanationPlan`, `LecturePlan`, `SlidesPlan`, `LabPlan`, `ActivityPlan`, `InstructorGuidePlan`, and `AssessmentPlan` schemas. A typed registry maps canonical task enums to schemas; homework shares `AssessmentPlan` rather than adding a task type. Short explanations therefore require only a concept, assumptions, explanation sequence, example, misconceptions, and check, while lecture plans include objectives, sequence, practice, synthesis, timing, and source coverage. Ollama's native structured-output `format` uses the selected task schema. Malformed JSON/schema output is a distinct failed Planner stage with raw response preserved; missing fields are not silently supplied.

Standard and Deep planning are separate: Standard uses concise plan-only wording, `think: false`, and deterministic Planner temperature 0. Deep uses the model's thinking capability but persists only the validated TeachingPlan. Planner versions are `planner-standard-v3` for concise Standard planning and `planner-v2` for Deep. Ollama thinking character count is captured when available; raw thinking text is never passed to Writer or persisted.

When source material is supplied, the Planner limits factual topics to what those sources support, while retaining freedom to add pedagogical structure and illustrative examples. The Writer treats source scope as higher priority than the teaching plan and omits or narrows unsupported factual plan requirements. Without supplied sources, both stages retain normal model-knowledge behavior.

Planner templates are versioned as `planner-standard-v3` for concise Standard planning and `planner-v2` for Deep planning. The Writer template is `writer-v7`. The selected versions are saved in result metadata.

For Standard and Deep requests with assembled source material, each source is first converted into a deterministic evidence ledger with stable IDs like `S01-E001`, `S01-E002`, ... . Before any LLM call, conservative normalization handles Markdown, inline math, wiki links, punctuation, and whitespace; exact normalized equality or whole-claim containment in a supplied source class the claim `direct_supported`. These claims retain deterministic evidence IDs and are not sent to the reviewer. Only unresolved claims are sent to Ollama at temperature 0, keyed by claim ID. The reviewer may classify them `supported`, `pedagogical`, or `unsupported`; supported entailments must include one or more evidence IDs. The application verifies that each evidence ID exists in the ledger and converts unsupported evidence or fabricated IDs into unsupported claims. It merges deterministic and reviewer decisions into one ordered ledger and enforces that every extracted claim appears exactly once. Only unsupported claims produce issues and require revision. A required revision receives the complete artifact and findings; it is limited to one attempt and receives the same deterministic pre-pass plus evidence-ID validation in final review. Reviewer/reviser errors, malformed or incomplete IDs, invalid evidence IDs, or a second revision request fail closed: no unapproved candidate is exposed as `final_output`. The original Writer output and revision candidate remain independently available in result diagnostics. Quick and source-free requests are unchanged, and the existing opt-in `ReviewFeedback` reviewer remains a separate compatibility feature.

Sourced plans include a common `source_scope` assessment: `sufficient`, `partial`, or `insufficient`, with supported topics and requested topics not covered by the sources. Partial coverage narrows the factual plan and output without removing non-factual user constraints. Insufficient coverage preserves the plan and retrieval diagnostics but fails before Writer, rather than silently generating an unsupported artifact. Source-free plans set `source_scope` to null and retain normal behavior.

The grounding prompts are versioned as `grounding-review-v6` and `grounding-revision-v2`. The reviewer output schema uses classification-specific models: `supported` requires at least one `evidence_ids` value, while `unsupported` and `pedagogical` require the field to be present and empty. The review prompt contains only unresolved claim IDs plus the deterministic evidence ledger, while direct matches are decided locally. Presentation-only emphasized Markdown labels (including optional blockquote labels) are ignored for deterministic direct-source matching; factual claim text remains unchanged. Saved grounding JSON preserves each final ledger claim's ID, text, classification, support method, evidence IDs, reason, and any issue category, along with the evidence ledger itself. `diagnostics.md` summarizes direct matches, reviewer decisions, pedagogical and unsupported claims, evidence validation failures, and coverage status. The revision prompt carries actionable findings rather than the full ledger, keeping the complete artifact and relevant source context within the configured context tier. Reviewer entailment remains a model judgment bounded by verifiable evidence IDs, not a formal proof; inspect its findings and retained candidates when source accuracy is critical.

## Rendering approved output

The web UI renders approved Markdown with KaTeX for inline `$...$` and `\(...\)` math and display `$$...$$` and `\[...\]` math. Multiple independent inline `$...$` expressions are supported on the same Markdown line. KaTeX's HTML and MathML output passes through DOMPurify before insertion; untrusted generated HTML remains sanitized. For example, a universal statement can be written as:

```latex
\[
\forall x \, P(x)
\]
```

Copy continues to copy the original Markdown and LaTeX delimiters.

## Task budgets and context selection

Each generation profile has a global stage output fallback and optional canonical-task overrides. Writer always uses `think: false`; planning consumes its separate reasoning/output budget. Values are in YAML, not scattered Python constants.

For each stage, context selection estimates assembled prompt tokens using approximately one token per four characters, multiplies the estimate by the configured safety margin, reserves the stage output budget, then chooses the smallest configured context tier that can fit the result. If no tier can accommodate it, the stage fails explicitly instead of silently sending an undersized context. The estimate and chosen tier are recorded. This is a planning heuristic, not tokenizer integration.

Timeouts and task-specific output budgets are independently configured in the versioned profiles YAML; they are not model defaults.

Stage-time estimates use the task-specific output budget. The first model stage uses the configured fallback generation speed; later stages may use an observed generation rate from an earlier successful stage in the same request. Estimates remain approximate and are not deadlines.

## Progress and results

The pipeline emits `ProgressEvent` callbacks for preparing, planning, writing, reviewing, complete, and failed states. Events can include elapsed time, generated tokens, observed tokens/second, and a clearly approximate remaining-time estimate. The CLI prints these events without logging source documents.

A `GenerationResult` retains Planner and Writer records separately, including timings, tokens, stop reasons, selected contexts, estimated input sizes, errors, final output, model defaults, effective profile, and prompt versions. Automatic grounding review records retain the initial and final decisions, structured issue excerpts, timing, and raw response independently from the original Writer output and any revision. The legacy opt-in reviewer returns actionable `ReviewFeedback`; it has no arbitrary score field.

Run persistence is opt-in:

```powershell
uv run python scripts/generate.py --profile standard --task lecture --instruction "Create a short introductory lesson about DNS." --save-run
```

Saved development artifacts go under ignored `artifacts/generations/<run-id>/`. A run includes `request.json`, the complete `sources.json` and readable `sources.md` for material actually assembled into the request, `result.json`, `diagnostics.md`, and `output.md` only when a final artifact is approved. Retrieval runs also include `retrieval.json`; stages that ran include their exact prompt snapshots (`planner_prompt.json`, `writer_prompt.json`, and applicable grounding review/revision prompt JSON files), and a validated Planner result is retained in `plan.json`. Grounded runs retain `writer_output.md`, structured `grounding_review_initial.json` and (after revision) `grounding_review_final.json`, and `grounding_revision.json`; `revised_output.md` is retained even when the final review fails. Diagnostics summarize decisions, flagged claim excerpts, errors, and stage timings. These opt-in files support development diagnostics, retrieval and grounding inspection, reproducibility, and investigating whether a questionable claim came from retrieved context or model generation. The stage-separated files preserve Planner success if Writer fails, providing the foundation for future resume from a saved plan. Full resume behavior is not implemented yet.

When retrieved sources produce a partial first-pass scope assessment, generation makes one bounded expansion attempt using the unsupported requested topics, anchored to the canonical query. It runs at most three expansion queries, each with at most two results, merges and deduplicates those passages with the initial results, then reassesses scope once. There are never more than two retrieval rounds or two scope assessments. Sufficient first-pass scope proceeds directly; insufficient first-pass scope still stops before Writer. If expanded scope remains partial, generation proceeds narrowed to supported topics; insufficient final scope stops before Writer. Expansion does not recurse or alter grounding. Explicit-only source material and source-free requests are not expanded.

Saved `retrieval.json` and `diagnostics.md` record retrieval rounds, per-query result summaries and timings, expansion topics and queries, unique added chunks, and initial/final scope assessments.

## Training boundary

This runtime does not train models or create adapters. Fine-tuning teaches how the instructor teaches; course profiles/context define who is being taught; retrieved sources supply factual knowledge; and caller-provided previous-course context describes where the class currently is. Knowledge indexing/retrieval and the later training pipeline remain separate systems. Retrieval is opt-in with `--retrieve`; existing commands and explicit `--source-file` behavior remain available without it.

## Local HTTP API

The local FastAPI service exposes the same `GenerationService` and `GenerationPipeline` as the CLI. It does not call Ollama as an alternate generation path, and sourced Standard/Deep requests retain the existing grounding review, bounded revision, and final approval rules.

Start the API from the repository root:

```powershell
uv run python scripts/serve.py --model-config configs/models/lecture-slm.yaml
```

It binds to `127.0.0.1:8000` by default. Open the interactive Swagger interface at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs); the OpenAPI document is at [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json). The launcher also accepts `--generation-config`, `--knowledge-config`, `--host`, and `--port`. Binding to an address other than localhost (for example, `--host 0.0.0.0`) exposes the API to the network and must be an intentional choice. The first version has no authentication and does not enable permissive CORS.

Available endpoints:

- `GET /api/health` reports API process health, configured-model availability, and knowledge-index availability separately.
- `GET /api/tasks` and `GET /api/profiles` report the configured task and generation profile values.
- `POST /api/generate` runs one request and returns its status, approved output (or `null`), grounding summary, source counts, timing summary, errors, and optional saved-run path. A pipeline-level grounding failure is returned as a structured `failed` result rather than an HTTP server error.
- `POST /api/generate/stream` sends Server-Sent Events named `progress` followed by `result`. Progress contains only the existing public `ProgressEvent` metadata; no model reasoning is streamed.
- `GET /api/runs/{request_id}` returns one of the 100 most recent API results held in process memory. Saved-run artifacts remain the persistent diagnostic record.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "explanation",
    "profile": "standard",
    "instruction": "Explain predicate logic.",
    "retrieve": true,
    "retrieval_top_k": 2,
    "save_run": true
  }'
```

PowerShell equivalent:

```powershell
$body = @{
  task = "explanation"
  profile = "standard"
  instruction = "Explain predicate logic."
  retrieve = $true
  retrieval_top_k = 2
  save_run = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/generate" `
  -Method Post -ContentType "application/json" -Body $body
```

A JavaScript/TypeScript client can use the same JSON contract:

```typescript
const response = await fetch("http://127.0.0.1:8000/api/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    task: "explanation",
    profile: "standard",
    instruction: "Explain predicate logic.",
    retrieve: true,
    retrieval_top_k: 2,
  }),
});

const result = await response.json();
console.log(result.output);
```

Quick, Standard, and Deep keep their normal Lecture SLM semantics. The API does not bypass retrieval, grounding, revision limits, or fail-closed behavior, and it does not expose full source documents or internal model configuration in ordinary generation responses.

## Local browser interface

The small TypeScript/Vite interface in `web/` talks only to the local HTTP API. For development, start the API in one terminal:

```powershell
uv run python scripts/serve.py --model-config configs/models/lecture-slm.yaml
```

Then start the browser UI in another:

```powershell
cd web
npm install
npm run dev
```

Vite serves the interface on its normal development port and proxies `/api` requests to `http://127.0.0.1:8000`. The browser streams generation through `POST /api/generate/stream`; it does not contact Ollama, load local files, or implement retrieval or grounding itself.

To build and serve the integrated local interface:

```powershell
cd web
npm run build
cd ..
uv run python scripts/serve.py --model-config configs/models/lecture-slm.yaml
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). FastAPI serves the built `web/dist` files when present; without a build, the API still starts and `/` explains how to build the UI. API routes, Swagger at `/docs`, and `/openapi.json` remain available in either case.

Frontend checks are `npm run typecheck`, `npm test`, and `npm run build` from `web/`. Browser preferences are limited to the last task, profile, retrieval toggle, and retrieval top-k in local storage; prompts and generated content are not persisted there.
