"""Import local teaching files into the deterministic review pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path

from lecture_slm.ingestion.importer import import_teaching_materials
from lecture_slm.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="One source file or an explicitly supplied directory"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/ingestion"))
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--course")
    parser.add_argument("--level")
    parser.add_argument("--show-warnings", action="store_true")
    args = parser.parse_args()
    configure_logging()
    try:
        report = import_teaching_materials(
            args.path,
            data_dir=args.data_dir,
            recursive=args.recursive,
            course=args.course,
            level=args.level,
            dry_run=args.dry_run,
            force_reextract=args.force_reextract,
        )
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).error("Import failed: %s", error)
        return 1
    print(json.dumps(report.as_dict(), indent=2))
    if args.show_warnings and report.errors:
        print("Errors:")
        print("\n".join(report.errors))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
