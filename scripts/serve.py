"""Run the local Lecture SLM HTTP API."""

import argparse
import sys
from pathlib import Path

import uvicorn

from lecture_slm.api.app import create_app
from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.service import GenerationService


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/lecture-slm.yaml"),
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/profiles.yaml"),
    )
    parser.add_argument(
        "--knowledge-config",
        type=Path,
        default=Path("configs/knowledge/default.yaml"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8000)
    args = parser.parse_args()

    try:
        service = GenerationService(
            model_config=load_model_config(args.model_config),
            profiles=load_generation_profiles(args.generation_config),
            knowledge_config_path=args.knowledge_config,
        )
    except (OSError, ValueError) as error:
        print(f"API startup failed: {error}", file=sys.stderr)
        return 1

    uvicorn.run(create_app(service), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
