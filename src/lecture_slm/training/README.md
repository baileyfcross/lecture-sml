# Planned Training Pipeline

Training is intentionally deferred in v0. The planned first pipeline is:

Qwen3.5-9B -> 4-bit QLoRA -> LoRA adapters -> supervised fine-tuning -> evaluation -> merge adapters -> GGUF export -> quantization -> Ollama -> post-quantization evaluation.

The future training dependency group is only a declaration of likely tooling. CUDA, PyTorch, and platform-specific installation must be selected deliberately when training begins; the bootstrap does not install GPU packages or download model weights.

## Chat template compatibility

The chat template used during fine-tuning **must remain compatible with the chat template used during inference and export**. Do not invent or hard-code a custom Qwen template in this repository yet. When training is implemented, obtain the appropriate template from the tokenizer/model configuration and preserve it as a versioned run artifact.

If Qwen3.5 creates toolchain or export compatibility issues, Qwen3-8B is an explicit compatibility fallback. The rest of the dataset, evaluation, configuration, and inference architecture should not need to change.

Each run should eventually record the base model and revision, training configuration, dataset manifest/version, Git commit, seed, metrics, evaluation results, export format, and quantization format under `artifacts/runs/<run-id>/`.
