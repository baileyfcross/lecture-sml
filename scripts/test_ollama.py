"""Send a small smoke-test request to the configured Ollama model."""

import argparse
import sys
from pathlib import Path

from lecture_slm.config.loader import load_model_config
from lecture_slm.inference.ollama_client import OllamaClient, OllamaError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/models/qwen35-9b.yaml"))
    args = parser.parse_args()
    try:
        config = load_model_config(args.config)
        response = OllamaClient(config.inference.host).chat(
            model=config.ollama_name,
            system_message="Answer briefly and clearly.",
            user_message="In one sentence, what is a variable in programming?",
            options={
                "temperature": config.inference.temperature,
                "top_p": config.inference.top_p,
                "seed": config.inference.seed,
            },
        )
    except (OSError, ValueError, OllamaError) as error:
        print(f"Ollama smoke test failed: {error}", file=sys.stderr)
        return 1
    print(response.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
