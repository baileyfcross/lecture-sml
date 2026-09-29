# Roadmap

Implemented: repository/environment, Ollama baseline client and Modelfile, dataset schemas, explicit pedagogy configuration, versioned human-review rubric, 26 validated evaluation prompts, resumable result runner and terminal review, two-prompt infrastructure smoke, inference performance characterization, task-specific Quick/Standard/Deep planner-writer runtime, and deterministic teaching-material ingestion with human review/export. The full baseline quality evaluation is not yet complete.

1. Phase 0 - Repository/environment (implemented)
2. Phase 1 - Ollama baseline (implemented)
3. Phase 2 - Dataset schemas (implemented)
4. Phase 3 - Pedagogy rubric (foundation implemented)
5. Phase 4 - Teaching material ingestion and human dataset review (implemented; explicit-path deterministic extraction, manifests, candidates, review queue, approved-only export)
6. Phase 5 - Build evaluation suite (complete: 26 prompts, validator, rubric, review workflow)
7. Phase 6 - Baseline evaluation (infrastructure complete; full Qwen3.5-9B baseline quality evaluation NOT YET COMPLETE)
	- Evaluation infrastructure - complete
	- Two-prompt evaluation smoke - complete
	- Performance characterization - complete
	- Baseline profile validation - not cleanly completed; retry after Ollama stability is restored
	- Full baseline quality evaluation - not complete
8. Phase 7 - Generation architecture (implemented: Quick/Standard/Deep routing, task-specific plans, progress, context selection, and stage metadata; Quick and final Standard smokes passed; Deep current routing covered by mocks after the live cap check)
9. Phase 8 - Baseline generation comparison (Quick vs Standard vs Deep)
10. Phase 9 - Build and analyze the initial approved dataset
11. Phase 10 - Training environment and QLoRA supervised fine-tuning
12. Phase 11 - Evaluate fine-tuned model against baseline
13. Phase 12 - Merge and GGUF export
14. Phase 13 - Ollama deployment
15. Phase 14 - Vault RAG integration
16. Phase 15 - Preference collection and optimization experiments
