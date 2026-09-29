# Inference Performance Characterization

The performance benchmark is separate from quality evaluation. It uses one fixed short DNS-generation prompt and the native Ollama API so model timing metadata stays structured. It does not score outputs, compare pedagogy, or declare a winning configuration.

## Run

```powershell
uv run python scripts/benchmark_ollama.py
```

The context sweep runs the same prompt at 4096, 8192, 16384, and 32768 with `think=false` and `num_predict=256`. It then compares `think=false` and `think=true` at the smallest tested context that accommodates the approximate maximum assembled baseline input plus 50% headroom. Seed, temperature, top-p, prompt, keep-alive, and output cap are otherwise held constant. The selected comparison context is printed and stored in `run.json`.

Use `--think-context` only to choose another member of the same measured context sweep, `--timeout-seconds` to set the benchmark request timeout, and `--num-predict` to set a different benchmark-only output cap. These options do not edit model YAML. Generated files go under ignored `evals/performance/<run-id>/`: `run.json`, append-only `results.jsonl`, and `summary.json`.

The output includes raw Ollama nanoseconds and converted seconds, prompt/generation token counts, throughput, `done_reason`, separate thinking output when returned by Ollama, and whether the output limit appears reached. If `done_reason` is absent, reaching the exact token cap is conservatively marked as potentially truncated. Timeout and other failures are recorded as cases, not omitted.

## Prompt-size estimate

The benchmark estimates each complete assembled baseline input, including system instructions and referenced course/pedagogy profiles, as `ceil(characters / 4)`. This is an approximate planning heuristic, not an Ollama tokenizer count. It reports minimum, median, and maximum and never changes the baseline context setting.

Current estimate for the 26 baseline prompts: **216 minimum / 262 median / 1329 maximum approximate tokens**, with 5315 maximum assembled characters. These are not tokenizer measurements, but no current prompt appears to require a 32768-token context window. The think comparison therefore used the smallest tested context, 4096, which leaves substantial headroom over the estimate.

## Observed benchmark (2026-09-28)

All cases used the same 39-token fixed prompt, seed 3407, temperature 0.5, top-p 0.9, and `num_predict=256`. The context sweep used `think=false`.

| Context | Think | Prompt tok/s | Generation tok/s | Total sec | Load sec | Prompt eval sec | Generation sec | Generated | Stop |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4096 | false | 58.47 | 2.90 | 67.74 | 0.00 | 0.67 | 66.94 | 194 | normal stop |
| 8192 | false | 9.02 | 2.89 | 82.43 | 10.97 | 4.33 | 67.13 | 194 | normal stop |
| 16384 | false | 9.08 | 2.90 | 82.34 | 11.20 | 4.29 | 66.85 | 194 | normal stop |
| 32768 | false | 9.45 | 2.89 | 82.74 | 11.55 | 4.13 | 67.06 | 194 | normal stop |

The isolated context cases generated the same 194-token response and stopped normally. Generation duration stayed near 67 seconds (about 2.9 generated tokens/s) across all windows. The first case was unusually fast for prompt evaluation (0.67 sec versus about 4.1–4.3 sec afterward), likely a warm-up/cache effect. Total duration differed by about 15 seconds between that first case and later cases, so a repeated randomized-order benchmark would be needed to attribute small differences to context size. An earlier overlapping sweep is preserved in its run directory but is not used in this table.

The separate 4096-context thinking comparison was:

| Context | Think | Prompt tok/s | Generation tok/s | Total sec | Load sec | Prompt eval sec | Generation sec | Prompt tokens | Generated | Stop |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4096 | false | 9.68 | 2.90 | 84.30 | 13.33 | 4.03 | 66.94 | 39 | 194 | normal stop |
| 4096 | true | 66.23 | 2.90 | 89.01 | 0.00 | 0.56 | 88.38 | 37 | 256 | output limit |

The think-enabled case reached the 256-token cap (`done_reason=length`), emitted about 982 characters in Ollama's separate `thinking` field, and returned an empty final answer. Ollama did not expose a separate thinking-token count; `eval_count` reported 256 generated tokens. This is a capped reasoning trace, not a completed educational answer. The false/true runs had very different load and prompt-evaluation times, so compare the captured profiles rather than interpreting their total time as a causal think-mode effect. No winner is declared.

## Output budgets

The production model configuration retains the 2048 global fallback and an empty `inference.task_output_tokens` mapping. The separate `configs/evaluation/qwen35-9b-baseline.yaml` profile selects 4096 context, disables thinking, preserves temperature/top-p/seed, and sets task-specific evaluation-only budgets. The global fallback remains 2048 for tasks absent from the profile. These settings do not replace production defaults.

The two infrastructure smoke lectures each generated exactly 512 tokens and ended mid-structure, so both were truncated and are not quality-baseline results. Preliminary planning ranges to test later are approximately 256–512 tokens for short explanations, 512–1024 for concise slides, 1024–2048 for labs, and 2048–4096 for full lectures. These are provisional estimates only; benchmark and human-review evidence should inform actual task budgets.

## Timeout and generation speed

The evaluation request timeout is configured independently from context and output length. Model defaults remain 600 seconds; the performance benchmark uses a separate 300-second timeout; the baseline evaluation profile uses 900 seconds. At about 2.9 tokens/second, the largest configured 2048-token output takes roughly 706 seconds for generation alone; 900 seconds leaves a limited margin for prompt evaluation and runtime variation. A timeout is a failure guard, not an output budget. Context size affects prompt processing and memory needs; `num_predict` caps generated tokens; generation throughput determines how long that cap takes. Larger timeout values do not improve throughput and should not be increased without measured need.

## Ubuntu server diagnostics

While benchmark requests are running, inspect Ollama's loaded-model placement on the Ubuntu server:

```bash
ollama ps
```

Check the `PROCESSOR` column for CPU, GPU, or a CPU/GPU split. On an NVIDIA host, `watch -n 1 nvidia-smi` can show GPU utilization and memory while the model is active; `top` or `htop` can show CPU load. Availability depends on the server's hardware and installed tools. These are diagnostic commands only: this project does not modify Ollama environment variables, drivers, thread counts, or model files.
