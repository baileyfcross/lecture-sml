"""Run prompts against the baseline model and store unscored responses."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lecture_slm.config.loader import load_model_config
from lecture_slm.inference.ollama_client import OllamaClient, OllamaError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/models/qwen35-9b.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("evals/prompts/baseline.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-results"))
    args = parser.parse_args()
    try:
        config = load_model_config(args.config)
        prompts = [
            json.loads(line)
            for line in args.prompts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        client = OllamaClient(config.inference.host)
        results: list[dict[str, object]] = []
        for prompt in prompts:
            started = time.perf_counter()
            response = client.chat(
                model=config.ollama_name,
                system_message=(
                    "You are Lecture SLM, an educational assistant. Follow the requested "
                    "task and do not invent source material."
                ),
                user_message=str(prompt["prompt"]),
                options={
                    "temperature": config.inference.temperature,
                    "top_p": config.inference.top_p,
                    "seed": config.inference.seed,
                },
            )
            results.append(
                {
                    "prompt_id": prompt["id"],
                    "task": prompt.get("task"),
                    "model_name": response.model,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "prompt": prompt["prompt"],
                    "response": response.content,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "configuration": config.model_dump(mode="json"),
                }
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, OllamaError) as error:
        print(f"Baseline evaluation failed: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} responses to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
