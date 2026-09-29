"""Export only explicitly approved candidates as a versioned JSONL dataset."""

import argparse
import sys
from pathlib import Path

from lecture_slm.ingestion.export import ExportError, export_approved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/ingestion/candidates.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--version", required=True, help="Dataset version, for example 0.1.0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        output, manifest, count = export_approved(
            args.candidates, args.output_dir, version=args.version, overwrite=args.overwrite
        )
    except (ExportError, OSError, ValueError) as error:
        print(f"Dataset export failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported {count} approved records to {output}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
