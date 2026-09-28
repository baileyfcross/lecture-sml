"""Validate a JSON or JSONL dataset file."""

import argparse
import logging
import sys
from pathlib import Path

from lecture_slm.data.validator import validate_file
from lecture_slm.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a JSON or JSONL dataset")
    args = parser.parse_args()
    configure_logging()
    try:
        records = validate_file(args.path)
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1
    logging.getLogger(__name__).info("Validated %d records from %s", len(records), args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
