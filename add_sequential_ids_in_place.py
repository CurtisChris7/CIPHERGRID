#!/usr/bin/env python3
"""Add sequential integer IDs to a JSONL file, overwriting it in place.

Each non-empty JSONL record receives an ``id`` field starting at 0. Any
existing ``id`` field is replaced. The update is written to a temporary file
in the same directory and atomically replaces the original only after every
row has been processed successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def add_ids_in_place(input_path: Path) -> int:
    """Add sequential IDs to *input_path* and replace the file in place.

    Returns the number of JSON records written.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    original_mode = stat.S_IMODE(input_path.stat().st_mode)
    record_count = 0
    temp_path: Path | None = None

    try:
        with input_path.open("r", encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=input_path.parent,
            prefix=f".{input_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temp_path = Path(destination.name)

            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                try:
                    record: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} contains {type(record).__name__}, "
                        "but each JSONL row must be a JSON object."
                    )

                # Put id first and replace any existing id value.
                updated_record = {"id": record_count}
                updated_record.update({key: value for key, value in record.items() if key != "id"})

                json.dump(updated_record, destination, ensure_ascii=False)
                destination.write("\n")
                record_count += 1

            destination.flush()
            os.fsync(destination.fileno())

        os.chmod(temp_path, original_mode)
        os.replace(temp_path, input_path)
        temp_path = None
        return record_count

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add id values 0..N-1 to a JSONL file and overwrite it in place."
    )
    parser.add_argument("jsonl_file", type=Path, help="Path to the JSONL file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = add_ids_in_place(args.jsonl_file)
    print(f"Updated {count} rows in place: {args.jsonl_file}")


if __name__ == "__main__":
    main()
