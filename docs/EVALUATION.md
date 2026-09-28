# Evaluation

The evaluation skeleton stores prompt IDs, model names, timestamps, raw responses, latency, configuration, and optional dimension-level scores. The rubric keeps these dimensions separate: factual accuracy, instruction following, course-level appropriateness, pedagogical sequencing, scaffolding, learning-objective alignment, slide text density, worked examples, student practice, misconception handling, prior-knowledge connection, source grounding, instructor voice/style, and structural coherence.

The baseline runner intentionally does not pretend to be an AI judge. Scores can be added through human review or a deliberately designed evaluator later. This enables comparisons among the baseline, fine-tuned versions, and post-quantization Ollama exports without hiding regressions behind one composite score.
