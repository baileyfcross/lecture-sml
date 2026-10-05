"""Send a small smoke-test request to the configured Ollama model."""

import argparse
import sys
from pathlib import Path

from lecture_slm.config.loader import load_model_config
from lecture_slm.inference.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaIncompleteResponseError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)

SMOKE_TEST_TIMEOUT_SECONDS = 120.0


def _format_duration(duration_ns: int | None) -> str:
    if duration_ns is None:
        return "unavailable"
    return f"{duration_ns / 1_000_000_000:.2f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/models/qwen35-9b.yaml"))
    args = parser.parse_args()
    try:
        config = load_model_config(args.config)
        client = OllamaClient(
            config.inference.host,
            timeout=SMOKE_TEST_TIMEOUT_SECONDS,
        )
        client.ensure_model_available(config.ollama_name)
        response = client.chat(
            model=config.ollama_name,
            user_message="Reply with only the word OK.",
            options={"num_predict": 8, "num_ctx": 2048},
            think=False,
            keep_alive="10m",
        )
    except OllamaConnectionError as error:
        print(f"Ollama server unreachable: {error}", file=sys.stderr)
        return 1
    except OllamaTimeoutError as error:
        print(f"Ollama smoke test timed out: {error}", file=sys.stderr)
        return 1
    except OllamaModelNotFoundError as error:
        print(f"Configured Ollama model missing: {error}", file=sys.stderr)
        return 1
    except OllamaHTTPError as error:
        print(f"Ollama HTTP error: {error}", file=sys.stderr)
        return 1
    except OllamaIncompleteResponseError as error:
        print(f"Ollama model response did not complete: {error}", file=sys.stderr)
        return 1
    except OllamaResponseError as error:
        print(f"Malformed Ollama response: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Ollama smoke test failed: {error}", file=sys.stderr)
        return 1
    print("Ollama smoke test passed.")
    print(f"Server: {config.inference.host}")
    print(f"Model: {config.ollama_name}")
    print(f"Response: {response.content.strip()}")
    print(f"Total duration: {_format_duration(response.total_duration_ns)}")
    print(f"Load duration: {_format_duration(response.load_duration_ns)}")
    print(f"Prompt processing: {_format_duration(response.prompt_eval_duration_ns)}")
    print(f"Generation: {_format_duration(response.eval_duration_ns)}")
    if response.prompt_tokens is not None:
        print(f"Prompt tokens: {response.prompt_tokens}")
    if response.completion_tokens is not None:
        print(f"Generated tokens: {response.completion_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
