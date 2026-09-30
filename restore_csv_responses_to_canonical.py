#!/usr/bin/env python3
"""
Restore custom CipherGrid action tokens in a CSV results file to the canonical
action symbology.

Only the selected response column is modified. All other columns and values are
preserved.

Example:
    python3 restore_csv_responses_to_canonical.py \
        --input results_custom.csv \
        --mapping custom_vocabulary.json \
        --output results_canonical.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Mapping


CANONICAL_ACTIONS: Dict[str, str] = {
    "UP": "ma",
    "RIGHT": "ga",
    "LEFT": "za",
    "DOWN": "va",
    "PICKUP": "ya",
    "ATTACK": "ba",
    "PLACE": "sa",
    "SWIM": "ra",
    "RESIGN": "ta",
}


class RestorationError(ValueError):
    """Raised when the mapping or CSV contents cannot be restored safely."""


def normalize_action_mapping(raw: Mapping[str, object]) -> Dict[str, str]:
    """
    Return a custom-token -> canonical-token mapping.

    Accepted action mapping keys:
      - semantic names: UP, RIGHT, LEFT, ...
      - canonical codes: ma, ga, za, ...

    The full vocabulary file may contain top-level "tiles" and "actions"
    objects. A flat action-only mapping is also accepted.
    """
    canonical_to_name = {
        canonical: name for name, canonical in CANONICAL_ACTIONS.items()
    }

    normalized_by_name: Dict[str, str] = {}

    for raw_key, raw_value in raw.items():
        key = str(raw_key)
        key_upper = key.upper()
        key_lower = key.lower()

        if key_upper in CANONICAL_ACTIONS:
            action_name = key_upper
        elif key_lower in canonical_to_name:
            action_name = canonical_to_name[key_lower]
        else:
            raise RestorationError(
                f"Unknown action mapping key {key!r}. Use one of the semantic "
                f"names ({', '.join(CANONICAL_ACTIONS)}) or canonical codes "
                f"({', '.join(CANONICAL_ACTIONS.values())})."
            )

        if action_name in normalized_by_name:
            raise RestorationError(
                f"Duplicate mapping supplied for action {action_name}."
            )

        if not isinstance(raw_value, str):
            raise RestorationError(
                f"Custom token for {action_name} must be a string."
            )

        custom_token = raw_value
        if not custom_token:
            raise RestorationError(
                f"Custom token for {action_name} cannot be empty."
            )
        if any(character.isspace() for character in custom_token):
            raise RestorationError(
                f"Custom token {custom_token!r} for {action_name} cannot "
                "contain whitespace."
            )
        if "-" in custom_token:
            raise RestorationError(
                f"Custom token {custom_token!r} for {action_name} cannot "
                "contain '-', because responses are hyphen-delimited."
            )

        normalized_by_name[action_name] = custom_token

    missing = [
        action_name
        for action_name in CANONICAL_ACTIONS
        if action_name not in normalized_by_name
    ]
    if missing:
        raise RestorationError(
            "Missing action mappings: " + ", ".join(missing)
        )

    custom_to_canonical: Dict[str, str] = {}
    for action_name, custom_token in normalized_by_name.items():
        if custom_token in custom_to_canonical:
            other_canonical = custom_to_canonical[custom_token]
            raise RestorationError(
                f"Custom token {custom_token!r} is assigned more than once "
                f"({other_canonical} and {CANONICAL_ACTIONS[action_name]})."
            )
        custom_to_canonical[custom_token] = CANONICAL_ACTIONS[action_name]

    return custom_to_canonical


def load_action_mapping(mapping_path: Path) -> Dict[str, str]:
    try:
        with mapping_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise RestorationError(
            f"Could not read mapping file {mapping_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RestorationError(
            f"Mapping file is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RestorationError("The mapping file must contain a JSON object.")

    if "actions" in data:
        action_data = data["actions"]
        if not isinstance(action_data, dict):
            raise RestorationError(
                "The top-level 'actions' value must be a JSON object."
            )
    else:
        # Permit a standalone flat action mapping.
        action_data = data

    return normalize_action_mapping(action_data)


def restore_response(
    response: str,
    custom_to_canonical: Mapping[str, str],
    csv_row_number: int,
    response_column: str,
) -> str:
    stripped = response.strip()

    # Preserve empty response cells.
    if not stripped:
        return response

    custom_tokens = stripped.split("-")
    if any(token == "" for token in custom_tokens):
        raise RestorationError(
            f"CSV row {csv_row_number}, column {response_column!r}: "
            f"malformed response {response!r}."
        )

    unknown = [
        token for token in custom_tokens
        if token not in custom_to_canonical
    ]
    if unknown:
        raise RestorationError(
            f"CSV row {csv_row_number}, column {response_column!r}: "
            "unknown custom action token(s): "
            + ", ".join(sorted(set(unknown)))
        )

    return "-".join(
        custom_to_canonical[token] for token in custom_tokens
    )


def validate_paths(
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    try:
        same_path = input_path.resolve() == output_path.resolve()
    except OSError:
        same_path = input_path == output_path

    if output_path.exists() and not overwrite:
        raise RestorationError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    if same_path and not overwrite:
        raise RestorationError(
            "Input and output paths are identical. Use --overwrite for an "
            "atomic in-place conversion."
        )


def restore_csv(
    input_path: Path,
    mapping_path: Path,
    output_path: Path,
    response_column: str,
    overwrite: bool,
) -> int:
    custom_to_canonical = load_action_mapping(mapping_path)
    validate_paths(input_path, output_path, overwrite)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    restored_count = 0

    try:
        # utf-8-sig transparently accepts CSVs with or without a UTF-8 BOM.
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)

            if reader.fieldnames is None:
                raise RestorationError(
                    f"CSV file has no header row: {input_path}"
                )

            if response_column not in reader.fieldnames:
                raise RestorationError(
                    f"CSV is missing column {response_column!r}. "
                    "Available columns: " + ", ".join(reader.fieldnames)
                )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)

                writer = csv.DictWriter(
                    destination,
                    fieldnames=reader.fieldnames,
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()

                # Physical row 1 is the header, so data starts at row 2.
                for csv_row_number, row in enumerate(reader, start=2):
                    response = row.get(response_column)
                    if response is None:
                        raise RestorationError(
                            f"CSV row {csv_row_number} has no value for "
                            f"column {response_column!r}."
                        )

                    restored = restore_response(
                        response,
                        custom_to_canonical,
                        csv_row_number,
                        response_column,
                    )

                    if restored != response and response.strip():
                        restored_count += 1

                    row[response_column] = restored
                    writer.writerow(row)

        os.replace(temporary_path, output_path)
        temporary_path = None

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return restored_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore custom CipherGrid action tokens in a CSV response column "
            "to the canonical action symbols."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV results file.",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        type=Path,
        help="Custom vocabulary JSON containing the action mappings.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV with canonical response symbols.",
    )
    parser.add_argument(
        "--response-column",
        default="response",
        help="Column to restore. Default: response.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        restored_count = restore_csv(
            input_path=args.input,
            mapping_path=args.mapping,
            output_path=args.output,
            response_column=args.response_column,
            overwrite=args.overwrite,
        )
    except (OSError, csv.Error, RestorationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Restored {restored_count} nonblank response values: "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
