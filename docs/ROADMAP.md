# Roadmap

Implemented: repository/environment, Ollama baseline client and Modelfile, dataset schemas, explicit pedagogy configuration, versioned human-review rubric, 26 validated evaluation prompts, resumable result runner and terminal review, two-prompt infrastructure smoke, inference performance characterization, and the Quick/Standard/Deep planner-writer runtime foundation. The full baseline quality evaluation is not yet complete. A baseline evaluation profile exists; its representative validation attempt was disrupted by Ollama instability and needs a clean rerun later. Current development phase: generation architecture.

1. Phase 0 - Repository/environment (implemented)
2. Phase 1 - Ollama baseline (implemented)
3. Phase 2 - Dataset schemas (implemented)
4. Phase 3 - Pedagogy rubric (foundation implemented)
5. Phase 4 - Import existing lectures/labs
6. Phase 5 - Build evaluation suite (complete: 26 prompts, validator, rubric, review workflow)
7. Phase 6 - Baseline evaluation (infrastructure complete; full Qwen3.5-9B baseline quality evaluation NOT YET COMPLETE)
	- Evaluation infrastructure - complete
	- Two-prompt evaluation smoke - complete
	- Performance characterization - complete
	- Baseline profile validation - not cleanly completed; retry after Ollama stability is restored
	- Full baseline quality evaluation - not complete
8. Phase 7 - Generation architecture (current: mocked workflow complete; real smoke verification pending)
9. Phase 8 - Baseline generation comparison (Quick vs Standard vs Deep)
10. Phase 9 - Import and review selected teaching materials
11. Phase 10 - QLoRA supervised fine-tuning
12. Phase 11 - Evaluate fine-tuned model
13. Phase 12 - Merge and GGUF export
14. Phase 13 - Ollama deployment
15. Phase 14 - Vault RAG integration
16. Phase 15 - Preference collection and optimization experiments
