# Baseline Evaluation

## Purpose

The baseline measures the untouched configured model before any fine-tuning. It establishes what `qwen3.5:9b` can already do for our educational tasks and provides a reference for later SFT, quantized, exported, or preference-trained variants. The starter set contains 26 versioned prompts. It is an initial engineering framework, not a scientifically validated rubric.

The health check is separate: `scripts/test_ollama.py` checks connectivity and a tiny response. Baseline evaluation uses the full configured context, explicit thinking behavior, configured seed and sampling parameters, and a bounded 2048-token output limit. It does not reuse the health check's `num_ctx: 2048` or `num_predict: 8`.

## Prompts and expected characteristics

Prompts live in `evals/prompts/baseline.jsonl` and use the Pydantic `EvaluationPrompt` schema. The tasks align with the shared educational task enum. Each prompt may refer to course and pedagogy profiles, provide context and source material, and declare applicable dimensions and tags.

Generative tasks often have many good responses, so prompts do not define one answer key. `expected_characteristics` records observable qualities for review, such as objective alignment, an example before independent practice, concise slides, or not introducing unsupported material. Challenge tags call out abstraction, slide density, missing prerequisites, misconceptions, source restriction, and continuity.

Evaluation prompts are kept outside training datasets. Do not automatically copy them into training, validation, or synthetic-data workflows; evaluation leakage would invalidate comparisons.

## Dimensions and human review

Review retains distinct dimensions: factual accuracy, source grounding, instruction following, course-level appropriateness, prerequisite awareness, pedagogical sequence, scaffolding, connection to prior knowledge, worked examples, practice opportunities, misconception handling, learning-objective alignment, slide density, clarity, structural coherence, and instructor style.

For each applicable dimension, a reviewer can assign 1 (poor), 2 (weak), 3 (acceptable), 4 (good), 5 (excellent), or `N/A`. Notes, observed strengths, and observed problems are optional. Unreviewed results are not assigned a score. There is no automatically calculated overall score and no LLM-as-a-judge.

## Validate and run

Validate every prompt, duplicate ID, dimension, expected-characteristic record, and referenced profile before generation:

```powershell
uv run python scripts/validate_eval.py
```

The model YAML describes model defaults; `configs/evaluation/qwen35-9b-baseline.yaml` is the versioned baseline evaluation profile. It selects 4096 context and `think: false`, preserves the configured sampling values and seed, and defines task-specific evaluation output limits. The profile does not modify production defaults. `prerequisite_reasoning` is represented by the existing `prerequisite-reasoning` prompt tag, not a new task enum.

Run the full baseline only when ready. Calls are sequential; each prompt result is appended as it finishes:

```powershell
uv run python scripts/run_baseline_eval.py
```

For a bounded smoke run, use `--limit 2`. Specify `--run-dir` to resume that run; successful prompt IDs are skipped and failed prompts are retried as a new attempt. `--rerun` explicitly appends another attempt for successful prompts instead of replacing them.

The default `--think configured` honors the explicit model setting. Explicit `--think disabled` and `--max-output-tokens` overrides are available for infrastructure-only smoke testing when configured long-form generation is too slow; overrides are recorded separately from the model configuration and must not be treated as a comparable quality baseline. They do not alter model YAML defaults. The 600-second evaluation timeout is independently configured from context and output-token limits; see [docs/PERFORMANCE.md](PERFORMANCE.md) for measured behavior and task-budget planning.

```powershell
uv run python scripts/run_baseline_eval.py --limit 2 --think disabled --max-output-tokens 512 --run-dir evals/results/baseline-smoke
uv run python scripts/run_baseline_eval.py --limit 2 --think disabled --max-output-tokens 512 --run-dir evals/results/baseline-smoke
uv run python scripts/run_baseline_eval.py --limit 2 --think disabled --max-output-tokens 512 --run-dir evals/results/baseline-smoke --rerun
```

Review completed outputs interactively. Use `q` at a score prompt to stop; completed prompt reviews are saved and skipped on the next invocation.

```powershell
uv run python scripts/review_eval.py evals/results/baseline-smoke --reviewer instructor
```

Validate a representative profile before the full 26-prompt quality baseline:

```powershell
uv run python scripts/run_baseline_eval.py --profile configs/evaluation/qwen35-9b-baseline.yaml --run-kind profile_validation --prompt-ids lecture-70-min-intro slides-source-progression explain-recursion-freshman lab-guided-lists activity-60-min-integration guide-exercise-answers assessment-formative-sql prerequisites-complexity --run-dir evals/results/qwen35-9b-profile-validation-v2
```

This eight-prompt run validates completion budgets, not the final quality baseline. Results distinguish base model configuration from evaluation profile and record effective settings per prompt. Length stops and outputs reaching their token caps are marked potentially truncated. Simple structure checks flag obvious missing slide sections, lab elements, or unfinished endings; they do not assess instructional quality.

### Initial profile-validation observations

The first profile-validation directory (`qwen35-9b-profile-validation-v1`) contains duplicate attempts because two runner invocations overlapped during terminal recovery. Treat its completed outputs as indicative examples, not as one clean controlled run. One completed attempt per task showed:

| Task | Budget | Generated | Generation sec | Total sec | Stop | Truncated | Structure |
| --- | ---: | ---: | ---: | ---: | --- | :---: | :---: |
| Lecture | 2048 | 1667 | 616.5 | 750.0 | stop | no | complete |
| Slides | 1024 | 856 | 292.6 | 574.4 | stop | no | complete |
| Explanation | 512 | 512 | 173.5 | 489.8 | length | yes | incomplete |
| Lab | 1536 | 1536 | 567.1 | 869.6 | length | yes | incomplete |
| Activity | 1024 | 1024 | 372.8 | 707.1 | length | yes | incomplete |
| Instructor guide | 1024 | 931 | 318.8 | 719.4 | stop | no | complete |
| Assessment | 1024 | 582 | 197.6 | 540.2 | stop | no | complete |
| Prerequisite reasoning | 512 | 512 | 173.3 | 396.9 | length | yes | incomplete |

The later clean run (`qwen35-9b-profile-validation-v2`) completed 1 of 8 prompts and failed 7 due to Ollama disconnects, connection failures, HTTP 500, or timeout. Its one completed prerequisite-reasoning output reached 512 tokens and was marked truncated. Consequently, the budget evidence is useful but provisional: explanation, lab, activity, and prerequisite reasoning budgets appear insufficient for their selected prompts; the successful lecture, slides, instructor-guide, and assessment outputs stopped normally. Do not treat either run as the 26-prompt quality baseline, and do not change budgets solely from this small, partly unstable sample.

## Run records and comparison

Each ignored `evals/results/<run-id>/` directory contains:

- `run.json`: model, redacted configuration, host fingerprint, generation settings including `think`, dataset version/hash, profile hashes, Git commit when available, seed, and project version.
- `responses.jsonl`: one append-only success or failure record per attempt, including prompt, expected characteristics, output, raw nanosecond and converted-second timings, token counts, completion status, and error details.
- `review.jsonl`: append-only human reviews at individual dimension level.
- `summary.json`: completed, failed, and skipped counts; it contains no aggregate quality score.

The raw `.env` host is never written to run metadata. A host fingerprint allows server consistency checks without retaining the address. Model comparisons should use the same prompt dataset/version, course/pedagogy profile hashes, and generation controls, then compare each dimension and runtime measurement independently.
