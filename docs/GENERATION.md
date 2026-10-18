# Generation Architecture

Generation is runtime orchestration over the current Ollama model. It is separate from training and evaluation scoring. Optional local knowledge retrieval is performed by the CLI before the existing generation router; the router and stage implementations do not access the vault. The same runtime can later target an Ollama fine-tuned tag such as `lecture-slm:v1` by changing model configuration only; it has no Hugging Face or training-stack dependency.

## Request inputs

`GenerationRequest` contains a canonical educational task, profile name, authoritative instruction, optional typed course and pedagogy profiles, structured source records, prior topics/course history, output preferences, and metadata. `SourceMaterial` includes source ID, title, section, text, and metadata. Explicit source files and opt-in retrieved materials are both appended to the same source list; retrieval provenance is retained in source metadata. See `docs/KNOWLEDGE.md` for the retrieval implementation.

Prompt builders keep instruction, course constraints, pedagogy, teaching plan, sources, and prior-course context in separate labeled JSON blocks. The writer is explicitly asked to ground claims in supplied sources and follow, rather than redo, the teaching plan.

When source material is supplied, the writer grounds not only definitions and technical claims but also framing, motivation, conclusions, significance, and claims about importance or a concept's broader role. It must not infer historical reasons from a capability, or inflate a supported point into an unsupported claim of importance, essential status, or a broader connection. Pedagogical examples, analogies, practice, transitions, and learner-focused explanations remain allowed; source-free requests retain normal access to model knowledge.

## Profiles and routing

`configs/generation/profiles.yaml` contains versioned Quick, Standard, and Deep settings. Model defaults remain in `configs/models/qwen35-9b.yaml`; the generation profile selects a workflow and stage-specific settings without mutating those defaults.

When Vault retrieval is enabled, the original instruction remains unchanged for planning and generation. `build_factual_retrieval_query` in `knowledge/query.py` removes bounded, phrase-aware request and artifact wording while preserving subject terms and punctuation. For course-relative or ambiguous requests, it may add a concise Workspace domain anchor and the most recent History item's first Markdown heading as query intent (falling back to its title); a self-contained factual topic does not inherit an unrelated Workspace name. For example, `Write out an introduction to quantum computing` remains `quantum computing` in a Workspace named `Quantum`. Saved retrieval diagnostics record the raw instruction, factual query, query terms, and whether/why a Workspace anchor was applied. This deterministic normalization is used consistently by lexical and semantic retrieval and query-term coverage.

- **Quick:** one Writer request, no Planner; useful for short explanations, rewriting, single slides, and small edits.
- **Standard:** concise task-specific Planner request with thinking disabled and temperature 0, then a separate Writer request with thinking disabled. This is the recommended workflow for normal educational artifacts.
- **Deep:** larger Planner budget/context followed by Writer with configurable context tiers. Like Standard, sourced Deep requests receive a bounded source-grounding review and, only when needed, one revision followed by a final review.

Planner failure is explicit and prevents Writer execution. Writer failure retains the completed plan in `GenerationResult`. No silent Planner-to-Quick fallback occurs. Planner retries are profile-configurable, limited to one retry, and currently configured as zero.

## Teaching plan

Planning uses a shared `TeachingPlanBase` plus task-specific `ExplanationPlan`, `LecturePlan`, `SlidesPlan`, `LabPlan`, `ActivityPlan`, `InstructorGuidePlan`, and `AssessmentPlan` schemas. A typed registry maps canonical task enums to schemas; homework shares `AssessmentPlan` rather than adding a task type. Short explanations therefore require only a concept, assumptions, explanation sequence, example, misconceptions, and check, while lecture plans include objectives, sequence, practice, synthesis, timing, and source coverage. Ollama's native structured-output `format` uses the selected task schema. Malformed JSON/schema output is a distinct failed Planner stage with raw response preserved; missing fields are not silently supplied.

Standard and Deep planning are separate: Standard uses concise plan-only wording, `think: false`, and deterministic Planner temperature 0. Deep uses the model's thinking capability but persists only the validated TeachingPlan. Initial Planner versions are `planner-standard-v5` for concise Standard planning and `planner-v4` for Deep. Ollama thinking character count is captured when available; raw thinking text is never passed to Writer or persisted.

When source material is supplied, the Planner limits factual topics to what those sources support, while retaining freedom to add pedagogical structure and illustrative examples. The Writer treats source scope as higher priority than the teaching plan and omits or narrows unsupported factual plan requirements. Without supplied sources, both stages retain normal model-knowledge behavior.

Planner templates are versioned as `planner-standard-v5` for concise Standard planning, `planner-v4` for Deep planning, and `planner-reassessment-v3` for post-retrieval plan reassessment. Their source-scope contract separates supported facts, unsupported requested facts, indispensable core gaps, and optional enrichment. Core topics may include foundational knowledge not literally named by the request; optional enrichment does not cause partial or insufficient coverage. The reassessment prompt treats only `SourceMaterial` as factual authority; Workspace Context and History can inform sequence and continuity but cannot fill technical evidence gaps. The Writer template is `writer-v11`. The selected versions are saved in result metadata.

For Standard and Deep requests with assembled source material, each source is first converted into a deterministic evidence ledger with stable IDs like `S01-E001`, `S01-E002`, ... . Before any LLM call, conservative normalization handles Markdown, inline math, wiki links, punctuation, and whitespace; exact normalized equality or whole-claim containment in a supplied source class the claim `direct_supported`. These claims retain deterministic evidence IDs and are not sent to the reviewer. Only unresolved claims are sent to Ollama at temperature 0, keyed by claim ID. The reviewer may classify them `supported`, `pedagogical`, or `unsupported`; supported entailments must include one or more evidence IDs. The application verifies that each evidence ID exists in the ledger and converts unsupported evidence or fabricated IDs into unsupported claims. It merges deterministic and reviewer decisions into one ordered ledger and enforces that every extracted claim appears exactly once. Only unsupported claims produce issues and require revision. A required revision receives the complete artifact and findings; it is limited to one attempt and receives the same deterministic pre-pass plus evidence-ID validation in final review. Reviewer/reviser errors, malformed or incomplete IDs, invalid evidence IDs, or a second revision request fail closed: no unapproved candidate is exposed as `final_output`. The original Writer output and revision candidate remain independently available in result diagnostics. Quick and source-free requests are unchanged, and the existing opt-in `ReviewFeedback` reviewer remains a separate compatibility feature.

Sourced plans include a common `source_scope` assessment: `sufficient`, `partial`, or `insufficient`, with supported topics, unsupported requested topics, indispensable core gaps, and optional enrichment. Core gaps are preserved through normalization even without lexical overlap and, if still unsupported after the one bounded retrieval reassessment, deterministically fail closed before Writer. Non-core requested gaps may still be omitted or deferred, with the Writer guard preventing unsupported instruction. Optional topics never trigger expansion, plan conflicts, or Writer blocking. Unsupported requested topics are normalized against authoritative requested-course scope (instruction, course/profile context, prior topics/context, and workspace continuity context) instead of retrieved-source noise. Source-free plans set `source_scope` to null and retain normal behavior.

Before strict plan validation, Planner JSON receives deterministic source-scope canonicalization: category entries are trimmed, blank entries removed, and case/whitespace-equivalent duplicates collapsed while preserving the first occurrence. Core gaps are added to requested gaps when missing, and requested/core gaps are removed from optional enrichment; optional topics are never promoted. This repairs only list hierarchy and shape conflicts—malformed list values and supported/requested contradictions still fail validation. Each planning stage persists the original raw response and compact raw/canonical topic lists with repair diagnostics.

The grounding prompts are versioned as `grounding-review-v12` and `grounding-revision-v5`. Classification-specific reviewer response schemas keep supported claims compact: factual `supported` claims require one or more `evidence_ids`, `continuity_supported` claims require one or more `continuity_ids`, pedagogical claims require an empty evidence list, and unsupported claims retain a concise model-generated reason with optional category. Workspace continuity has two checks: a conservative claim-language gate identifies claims that describe course/project history or sequence, and a claim becomes reviewable only if selected continuity spans have meaningful normalized topic overlap with the claim. Structural words such as “next session” or “Lecture 3” establish course sequence but do not establish the topic. Ordinary stop words, generic framing terms, and structural terms do not count as topic overlap; topic-empty claims remain unreviewable, and matching a session number alone cannot support a different topic. A future claim is reviewable only when selected explicit-sequence spans collectively cover all meaningful topic terms in the claim, excluding generic descriptors; unrelated spans in the same History item cannot inherit a future marker. The reviewer receives only reviewable claims and must verify that the evidence describes the same topic and sequence; one matching topic does not establish additional topics or applications in a combined claim. Selection diagnostics retain selected candidate claim/topic terms, overlap terms, scores, and IDs, plus counts of structural-only candidates and incomplete future claims rejected. The application rejects `continuity_supported` for all other claims and validates every cited continuity ID against the spans selected for that exact claim. Thus an empty ledger, a structural-only match, an irrelevant item, or an item omitted by the cap cannot support a continuity claim. Mixed historical-and-domain assertions are not eligible for continuity-only support; they must be narrowed to the historical statement or supported as factual claims from `SourceMaterial`.
Grounding also applies deterministic current-artifact pedagogical handling before semantic review. In addition to instructional flow, narrowly recognized artifact organization or purpose, intended audience, estimated activity or reading duration, and current-artifact learner activities do not require source evidence. Domain assertions and mixed factual claims remain subject to grounding; past and future course-sequence claims remain subject to Workspace continuity evidence.
Before Writer runs, application-side validation checks whether a `partial` source scope is internally consistent with the effective teaching plan. If unsupported requested topics are still taught substantively in plan content, generation fails closed before Writer. When the unsupported scope is core to the requested artifact, effective source scope is escalated to `insufficient` and diagnostics record raw, normalized, and effective statuses with explicit plan/source-scope conflicts and writer-blocked state.

For factual review, multiple deterministic evidence spans—including spans from different supplied sources—may jointly support a claim when their stated facts straightforwardly compose into it. This permits direct compositional entailment, not new claims about significance, importance, prevalence, causation, motive, superiority, performance, necessity, or broader application. Every cited evidence ID is still validated against the deterministic evidence ledger. The single bounded revision prefers deleting unnecessary unsupported claims or replacing them with simpler factual wording closely shaped by supplied evidence rather than synthesizing a broader statement.

When a revision is needed, the revision prompt carries structured issue details (claim ID and excerpt, category, reason, available source IDs, and continuity failure details when applicable) plus specific corrective guidance. The reviser is instructed to make surgical changes and preserve content already supported by evidence. It still receives at most one attempt, followed by the normal final review.

During that final review, exact unchanged claims are matched to their initial assessments by conservative normalized text equality, with duplicate claims matched one-to-one. Factual supported or unsupported decisions are reused only when the ordered factual evidence ledger is unchanged; pedagogical decisions may be reused for exact unchanged text. Direct matches are recomputed, continuity decisions are reviewed normally, and changed or new claims alone are sent to the final reviewer. Reused assessments are remapped to the revised claim IDs and pass through the same coverage and evidence-ID validation as new assessments. If no claims need review, the final ledger is assembled without an additional reviewer call. The run diagnostics record compact reuse counts and whether the reviewer call was skipped.

For Workspace-backed requests, the Writer is also told not to assert specific future lecture, session, lab, lesson, or assignment topics unless a Workspace Context item explicitly establishes that sequence. Workspace History by itself cannot establish what a future session will cover. Generic pedagogical transitions remain allowed, and this safeguard does not change source-free Writer behavior. The continuity ledger is claim-aware and compact: only unresolved claims that pass the deterministic continuity gate are eligible, the reviewer receives only selected continuity spans (bounded per claim and globally), and continuity IDs are validated fail-closed. The application still synthesizes deterministic internal reasons for supported and pedagogical outcomes while preserving full authoritative claim/provenance records in the merged ledger. For each review, the structured response schema restricts supported-claim evidence IDs to the exact factual IDs in that review's deterministic ledger; if there are none, the supported branch is omitted. This schema constraint complements application-side validation, and the application never repairs a mismatched ID. Evidence IDs and continuity IDs are both verified against deterministic ledgers; invalid or fabricated IDs fail closed.

The reviewer output allowance is selected from the unresolved claims sent to the model: 4096 tokens for up to 40 claims and 8192 tokens above 40. Direct-supported claims are excluded from this count and do not trigger an LLM review. Context selection reserves the chosen output budget from the existing configured tiers. Initial and final reviews calculate their budgets independently, so a smaller revised claim set does not inherit a larger initial allowance. When Ollama reports a length stop at the selected token limit and the structured response cannot be validated, the failure is reported as reviewer-output truncation; malformed JSON without that evidence remains a normal parsing or validation error. Diagnostics retain claim counts, selected budget, timing, context, stop reason, and the raw reviewer response, including truncated output. The review prompt contains only unresolved claim IDs plus the deterministic evidence ledger, while direct matches are decided locally. Presentation-only emphasized Markdown labels (including optional blockquote labels) are ignored for deterministic direct-source matching; factual claim text remains unchanged. Saved grounding JSON preserves each final ledger claim's ID, text, classification, support method, evidence IDs, reason, and any issue category, along with the evidence ledger itself. `diagnostics.md` summarizes direct matches, reviewer decisions, pedagogical and unsupported claims, evidence validation failures, and coverage status. The revision prompt carries actionable findings rather than the full ledger, keeping the complete artifact and relevant source context within the configured context tier. Reviewer entailment remains a model judgment bounded by verifiable evidence IDs, not a formal proof; inspect its findings and retained candidates when source accuracy is critical.

## Rendering approved output

The web UI renders approved Markdown with KaTeX for inline `$...$` and `\(...\)` math and display `$$...$$` and `\[...\]` math. Multiple independent inline `$...$` expressions are supported on the same Markdown line. KaTeX's HTML and MathML output passes through DOMPurify before insertion; untrusted generated HTML remains sanitized. For example, a universal statement can be written as:

```latex
\[
\forall x \, P(x)
\]
```

Copy continues to copy the original Markdown and LaTeX delimiters.

## Task budgets and context selection

Each generation profile has a global stage output fallback and optional canonical-task overrides. For thinking-enabled Planner stages, the structured output budget remains the expected final plan size; the separate thinking reserve provides additional reasoning capacity. The total generation budget is the structured output budget plus the thinking reserve, and context selection reserves that total. Writer always uses `think: false` and continues to use only its task output budget. Values are in YAML, not scattered Python constants.

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

Saved development artifacts go under ignored `artifacts/generations/<run-id>/`. A run includes `request.json`, the complete `sources.json` and readable `sources.md` for material actually assembled into the request, `result.json`, `diagnostics.md`, and `output.md` only when a final artifact is approved. Retrieval runs also include `retrieval.json`; stages that ran include their exact prompt snapshots (`planner_prompt.json`, `writer_prompt.json`, and applicable grounding review/revision prompt JSON files), and a validated Planner result is retained in `plan.json`. Grounded runs retain `writer_output.md`, structured `grounding_review_initial.json` and (after revision) `grounding_review_final.json`, and `grounding_revision.json`; `revised_output.md` is retained even when the final review fails. Diagnostics summarize decisions, flagged claim excerpts, errors, and stage timings. These files support development diagnostics, retrieval and grounding inspection, reproducibility, and investigating whether a questionable claim came from retrieved context or model generation. The stage-separated files preserve Planner success if Writer fails, providing the foundation for future resume from a saved plan. Full resume behavior is not implemented yet. CLI generation still saves runs only when `--save-run` is specified.

Web generations are always saved to this archive directory, and the top-level **History** button lazily reads those local archives so completed and failed runs remain available after a browser refresh or backend restart. The History list is limited to the 100 newest entries by default (maximum 500); opening a row loads its saved result and approved `output.md` without calling a model, retrieval, or embeddings. Failed archives remain available for diagnostics but cannot be copied to a Workspace. Adding an approved output creates a separate Workspace History item; deleting the archive permanently removes its saved diagnostics but does not delete that Workspace copy. History is local to this application and filesystem.

When retrieved sources produce a partial first-pass scope assessment, generation makes one bounded expansion attempt using only normalized unsupported requested topics, prioritizing core topics before non-core gaps. Optional topics are never searched. `build_retrieval_expansion_plan` derives up to three distinct compact factual queries, using the Workspace domain anchor when one was intentionally applied or otherwise the factual query as context. It does not append the raw generation instruction. Each query still has at most two results; passages are merged and deduplicated with the initial results, then scope is reassessed once. The reassessment receives the expanded factual source set and the already-validated initial TeachingPlan, and conservatively updates that plan rather than repeating full pedagogical planning. Only `SourceMaterial`—explicit caller material, Workspace References, and retrieved global knowledge—may establish domain facts. Workspace Context and History may establish sequence and instructional continuity but may not fill technical factual gaps. Reassessment uses `planner-reassessment-v3`, `think: false`, temperature 0, zero thinking reserve, and the same task-specific structured-output budget as the Planner profile. This lets Deep perform its extended pedagogical reasoning once, while retrieval expansion updates factual coverage without paying for a second reasoning pass. There are never more than two retrieval rounds, two source-scope assessments, or one reassessment. Sufficient first-pass scope proceeds directly; insufficient first-pass scope still stops before Writer. If expanded scope remains partial, generation proceeds narrowed to supported topics; any remaining core gap forces effective insufficient scope and stops before Writer. Expansion does not recurse or alter grounding. Explicit-only source material and source-free requests are not expanded.

Saved runs retain separate `planner_initial` and `planner_reassessment` timing/context/token diagnostics. If reassessment occurs, the initial and reassessment prompts, plans, raw structured responses, prompt versions, and reassessment error details are saved separately; the existing `planner_prompt.json` and `plan.json` continue to represent the effective final Planner result.

Saved `retrieval.json` and `diagnostics.md` record query transformation and anchor decisions, normalized expansion topics and queries, and bounded per-query fusion traces. Fusion traces include lexical/dense ranks and scores, their RRF contributions, fused rank and score, final selection, and rejection reason. Assembly traces distinguish deduplication, role filtering, and source-budget exclusion from a fusion cutoff.

## Local Workspaces

Workspaces provide explicit local continuity for a course or project. Their SQLite database defaults to `data/workspaces/workspaces.sqlite` and is ignored by Git. Workspace data is separate from the Obsidian Vault and the global knowledge index; creating, editing, searching, or deleting Workspace items never indexes or mutates either system.

Each item has one role:

- **Context** describes continuity such as course conventions. It can guide planning and wording but is not factual evidence.
- **History** stores previously approved work or prior-session context. It can guide continuity but is not factual evidence.
- **Reference** is the only Workspace role converted to `SourceMaterial`, and therefore the only Workspace role that may support factual claims or appear in grounding citations.

For grounding review, Context and History are considered only as continuity evidence and only for claims that describe course/project history or sequence. Eligibility by wording alone is insufficient: at least one relevant selected span must be available for the claim, and that span must overlap the meaningful topic terms. Future-sequence claims require explicit sequence evidence in the selected span itself (such as a roadmap or an upcoming/next-session statement); a numbered Context lecture title is an exception because it identifies the topic assigned to that lecture. A future marker elsewhere in a History item does not carry over to unrelated spans in that item. Topical overlap alone is insufficient, and a past History item without explicit future/roadmap evidence cannot establish future topics. In generated output, only repeat or closely paraphrase an explicitly stated future topic. Do not append related but unstated applications, examples, or subtopics. A future statement in History does not authorize another future topic; a current Context roadmap takes precedence over a conflicting or newer History statement.

Context and History are selected by pinned status, recent History (two by default), and Workspace-local FTS5 matches, then bounded by the profile budgets in `configs/workspaces/default.yaml` (Quick 2048, Standard 4096, Deep 8192 estimated tokens). Workspace References share the existing factual source budget: caller-supplied sources take precedence, then pinned/matching Workspace References, and global Vault retrieval uses only any remaining budget. Workspace identity is added to factual retrieval only when course-relative intent, missing subject matter, or ambiguity makes it useful; a self-contained factual request is not automatically prefixed. This query use does not promote Context or History to factual evidence.

Pass only a `workspace_id` to the generation API or `scripts/generate.py --workspace-id`. The service loads and materializes Workspace content internally; API clients cannot send arbitrary raw Workspace context. Workspace selection and exact selected Context/History/Reference snapshots are included in opt-in saved runs as `workspace_context.json`.

The API exposes `GET/POST /api/workspaces`, `GET/PATCH/DELETE /api/workspaces/{workspace_id}`, and scoped folder/item routes below `/api/workspaces/{workspace_id}/folders` and `/items`. Folder deletion is limited to empty folders, and cross-Workspace folder/item references are rejected. `POST /api/workspaces/{workspace_id}/history` saves only a completed approved API result; saving is explicit and creates a History item. Promote an existing item to Reference explicitly with `PATCH /api/workspaces/{workspace_id}/items/{item_id}` and `{"role":"reference"}`.

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
