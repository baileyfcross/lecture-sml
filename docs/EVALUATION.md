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

## Run records and comparison

Each ignored `evals/results/<run-id>/` directory contains:

- `run.json`: model, redacted configuration, host fingerprint, generation settings including `think`, dataset version/hash, profile hashes, Git commit when available, seed, and project version.
- `responses.jsonl`: one append-only success or failure record per attempt, including prompt, expected characteristics, output, raw nanosecond and converted-second timings, token counts, completion status, and error details.
- `review.jsonl`: append-only human reviews at individual dimension level.
- `summary.json`: completed, failed, and skipped counts; it contains no aggregate quality score.

The raw `.env` host is never written to run metadata. A host fingerprint allows server consistency checks without retaining the address. Model comparisons should use the same prompt dataset/version, course/pedagogy profile hashes, and generation controls, then compare each dimension and runtime measurement independently.
