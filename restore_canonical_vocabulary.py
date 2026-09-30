#!/usr/bin/env python3
"""
Restore a fully custom-vocabulary CipherGrid benchmark JSONL to canonical
symbology, including restoration of the custom row marker to canonical "wa".

Prompt line breaks may be real newlines or literal ``\\n`` sequences; their
original representation is preserved.

Only prompt-base, query, and answer are modified. All other fields, including
the base64 image, are preserved unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple


CANONICAL_ROW_MARKER = "wa"

TILES: Dict[str, str] = {
    "WALL": "ca",
    "OPEN": "pa",
    "GOAL": "da",
    "START": "na",
    "MONSTER": "ea",
    "REVERSAL": "xa",
    "SWORD": "oa",
    "PLANK": "la",
    "WATER": "ha",
    "TRAP": "ka",
}

ACTIONS: Dict[str, str] = {
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


class VocabularyError(ValueError):
    """Raised when a vocabulary or encoded benchmark field is invalid."""


def normalize_entries(
    raw: Mapping[str, object],
    expected: Mapping[str, str],
    group_name: str,
) -> Dict[str, str]:
    code_to_name = {code: name for name, code in expected.items()}
    normalized: Dict[str, str] = {}

    for raw_key, raw_value in raw.items():
        key = str(raw_key)
        upper_key = key.upper()
        lower_key = key.lower()

        if upper_key in expected:
            name = upper_key
        elif lower_key in code_to_name:
            name = code_to_name[lower_key]
        else:
            raise VocabularyError(
                f"Unknown {group_name} mapping key {key!r}. Use semantic names "
                f"({', '.join(expected)}) or canonical codes "
                f"({', '.join(expected.values())})."
            )

        if name in normalized:
            raise VocabularyError(
                f"Duplicate mapping supplied for {group_name} entry {name}."
            )
        if not isinstance(raw_value, str):
            raise VocabularyError(
                f"Custom token for {group_name} entry {name} must be a string."
            )
        normalized[name] = raw_value

    missing = [name for name in expected if name not in normalized]
    if missing:
        raise VocabularyError(
            f"Missing {group_name} mappings: {', '.join(missing)}"
        )

    return normalized


def validate_custom_tokens(
    custom_row_marker: str,
    tile_named: Mapping[str, str],
    action_named: Mapping[str, str],
) -> None:
    named_tokens = {
        "ROW_MARKER": custom_row_marker,
        **tile_named,
        **action_named,
    }
    seen: Dict[str, str] = {}

    for name, token in named_tokens.items():
        if not isinstance(token, str):
            raise VocabularyError(f"Custom token for {name} must be a string.")
        if token == "":
            raise VocabularyError(f"Custom token for {name} cannot be empty.")
        if any(character.isspace() for character in token):
            raise VocabularyError(
                f"Custom token {token!r} for {name} cannot contain whitespace."
            )
        if any(delimiter in token for delimiter in ("-", ",", ":")):
            raise VocabularyError(
                f"Custom token {token!r} for {name} cannot contain '-', ',' or ':'."
            )
        if token in seen:
            raise VocabularyError(
                f"Custom token {token!r} is assigned to both {seen[token]} "
                f"and {name}. All row, tile, and action tokens must be unique."
            )
        seen[token] = name


def load_vocabulary(
    path: Path,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    Return:
      custom row marker,
      custom tile token -> canonical tile code,
      custom action token -> canonical action code.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise VocabularyError(f"Could not read mapping file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise VocabularyError("The vocabulary mapping must be a JSON object.")

    nested = any(key in data for key in ("row_marker", "tiles", "actions"))

    if nested:
        unknown_top_level = set(data) - {"row_marker", "tiles", "actions"}
        if unknown_top_level:
            raise VocabularyError(
                "Unknown top-level mapping keys: "
                + ", ".join(sorted(map(str, unknown_top_level)))
            )

        custom_row_marker = data.get("row_marker")
        tile_raw = data.get("tiles")
        action_raw = data.get("actions")

        if not isinstance(custom_row_marker, str):
            raise VocabularyError(
                "Nested mapping must contain a string-valued 'row_marker'."
            )
        if not isinstance(tile_raw, dict) or not isinstance(action_raw, dict):
            raise VocabularyError(
                "Nested mapping must contain object-valued 'tiles' and 'actions'."
            )

        tile_named = normalize_entries(tile_raw, TILES, "tile")
        action_named = normalize_entries(action_raw, ACTIONS, "action")
    else:
        tile_raw: Dict[str, object] = {}
        action_raw: Dict[str, object] = {}
        custom_row_marker = None
        tile_codes = set(TILES.values())
        action_codes = set(ACTIONS.values())

        for key, value in data.items():
            key_string = str(key)
            upper_key = key_string.upper()
            lower_key = key_string.lower()

            if upper_key == "ROW_MARKER" or lower_key == CANONICAL_ROW_MARKER:
                if custom_row_marker is not None:
                    raise VocabularyError("Duplicate row-marker mapping supplied.")
                if not isinstance(value, str):
                    raise VocabularyError(
                        "Custom token for ROW_MARKER must be a string."
                    )
                custom_row_marker = value
            elif upper_key in TILES or lower_key in tile_codes:
                tile_raw[key_string] = value
            elif upper_key in ACTIONS or lower_key in action_codes:
                action_raw[key_string] = value
            else:
                raise VocabularyError(f"Unknown mapping key {key_string!r}.")

        if custom_row_marker is None:
            raise VocabularyError(
                "Missing row-marker mapping. Supply 'ROW_MARKER' or canonical "
                f"key {CANONICAL_ROW_MARKER!r}."
            )

        tile_named = normalize_entries(tile_raw, TILES, "tile")
        action_named = normalize_entries(action_raw, ACTIONS, "action")

    validate_custom_tokens(custom_row_marker, tile_named, action_named)

    tile_reverse = {custom: TILES[name] for name, custom in tile_named.items()}
    action_reverse = {
        custom: ACTIONS[name] for name, custom in action_named.items()
    }
    return custom_row_marker, tile_reverse, action_reverse


def convert_grid_display(
    encoded: str,
    custom_row_marker: str,
    tile_reverse: Mapping[str, str],
    context: str,
) -> str:
    converted_rows = []
    custom_prefix = f"{custom_row_marker}:"
    canonical_prefix = f"{CANONICAL_ROW_MARKER}:"

    for row_index, row in enumerate(encoded.split(","), start=1):
        if not row.startswith(custom_prefix):
            raise VocabularyError(
                f"{context}: row {row_index} does not begin with "
                f"{custom_prefix!r}."
            )

        cells = row[len(custom_prefix):].split("-")
        if not cells or any(cell == "" for cell in cells):
            raise VocabularyError(f"{context}: row {row_index} is malformed.")

        unknown = [cell for cell in cells if cell not in tile_reverse]
        if unknown:
            raise VocabularyError(
                f"{context}: unknown custom tile token(s): "
                + ", ".join(sorted(set(unknown)))
            )

        converted_rows.append(
            canonical_prefix
            + "-".join(tile_reverse[cell] for cell in cells)
        )

    return ",".join(converted_rows)


def convert_action_sequence(
    encoded: str,
    action_reverse: Mapping[str, str],
    context: str,
) -> str:
    tokens = encoded.split("-")
    if not tokens or any(token == "" for token in tokens):
        raise VocabularyError(f"{context}: malformed action sequence.")

    unknown = [token for token in tokens if token not in action_reverse]
    if unknown:
        raise VocabularyError(
            f"{context}: unknown custom action token(s): "
            + ", ".join(sorted(set(unknown)))
        )

    return "-".join(action_reverse[token] for token in tokens)


def uniquely_tokenize_query(
    encoded: str,
    custom_row_marker: str,
    tile_reverse: Mapping[str, str],
    context: str,
) -> List[str]:
    """Tokenize a delimiter-free custom query, requiring one unique parse."""
    lexical_to_canonical = dict(tile_reverse)
    lexical_to_canonical[custom_row_marker] = CANONICAL_ROW_MARKER
    lexical_tokens = sorted(
        lexical_to_canonical,
        key=lambda token: (-len(token), token),
    )

    length = len(encoded)
    ways = [0] * (length + 1)
    choice: List[Optional[Tuple[str, int]]] = [None] * (length + 1)
    ways[length] = 1

    for position in range(length - 1, -1, -1):
        total = 0
        unique_choice: Optional[Tuple[str, int]] = None

        for token in lexical_tokens:
            next_position = position + len(token)
            if next_position > length or not encoded.startswith(token, position):
                continue

            suffix_ways = ways[next_position]
            if suffix_ways == 0:
                continue

            if total == 0 and suffix_ways == 1:
                unique_choice = (token, next_position)
            else:
                unique_choice = None

            total = min(2, total + suffix_ways)

        ways[position] = total
        if total == 1:
            choice[position] = unique_choice

    if ways[0] == 0:
        raise VocabularyError(
            f"{context}: query cannot be segmented using the custom row/tile "
            "vocabulary."
        )
    if ways[0] > 1:
        raise VocabularyError(
            f"{context}: query has multiple valid tokenizations. Choose a less "
            "ambiguous custom row/tile vocabulary."
        )

    canonical_tokens: List[str] = []
    position = 0
    while position < length:
        selected = choice[position]
        if selected is None:
            raise VocabularyError(
                f"{context}: internal tokenization failure at offset {position}."
            )

        lexical, next_position = selected
        canonical_tokens.append(lexical_to_canonical[lexical])
        position = next_position

    if not canonical_tokens or canonical_tokens[0] != CANONICAL_ROW_MARKER:
        raise VocabularyError(
            f"{context}: query must begin with custom row marker "
            f"{custom_row_marker!r}."
        )
    if canonical_tokens[-1] == CANONICAL_ROW_MARKER:
        raise VocabularyError(f"{context}: query cannot end with a row marker.")
    if any(
        left == CANONICAL_ROW_MARKER and right == CANONICAL_ROW_MARKER
        for left, right in zip(canonical_tokens, canonical_tokens[1:])
    ):
        raise VocabularyError(f"{context}: query contains an empty row.")

    return canonical_tokens


def convert_query(
    encoded: str,
    custom_row_marker: str,
    tile_reverse: Mapping[str, str],
    context: str,
) -> str:
    return "".join(
        uniquely_tokenize_query(
            encoded,
            custom_row_marker,
            tile_reverse,
            context,
        )
    )


def convert_prompt(
    prompt: str,
    custom_row_marker: str,
    tile_reverse: Mapping[str, str],
    action_reverse: Mapping[str, str],
    context: str,
) -> str:
    converted_lines = []

    grid_pattern = re.compile(
        r"^(\s*\d+\)\s+)(\S+)([ \t]*(?:\r?\n)?)$"
    )
    response_pattern = re.compile(
        r"^(\s*->\s+)(\S+)([ \t]*(?:\r?\n)?)$"
    )

    # Capture separators so both actual and literal newlines survive unchanged.
    # Do not unescape the entire prompt: that could alter unrelated text.
    parts = re.split(r"(\\n|\r\n|\r|\n)", prompt)
    for part_index, line in enumerate(parts):
        if part_index % 2:
            converted_lines.append(line)
            continue
        line_number = part_index // 2 + 1
        grid_match = grid_pattern.match(line)
        if grid_match:
            prefix, encoded, suffix = grid_match.groups()
            converted = convert_grid_display(
                encoded,
                custom_row_marker,
                tile_reverse,
                f"{context}, prompt line {line_number}",
            )
            converted_lines.append(prefix + converted + suffix)
            continue

        response_match = response_pattern.match(line)
        if response_match:
            prefix, encoded, suffix = response_match.groups()
            converted = convert_action_sequence(
                encoded,
                action_reverse,
                f"{context}, prompt line {line_number}",
            )
            converted_lines.append(prefix + converted + suffix)
            continue

        converted_lines.append(line)

    result = "".join(converted_lines)

    action_alternatives = "|".join(
        sorted(
            (re.escape(token) for token in action_reverse),
            key=len,
            reverse=True,
        )
    )
    fenced_pattern = re.compile(
        rf"---((?:{action_alternatives})(?:-(?:{action_alternatives}))*)---"
    )

    def replace_fenced(match: re.Match[str]) -> str:
        return "---" + convert_action_sequence(
            match.group(1),
            action_reverse,
            f"{context}, fenced prompt example",
        ) + "---"

    return fenced_pattern.sub(replace_fenced, result)


def convert_record(
    record: dict,
    custom_row_marker: str,
    tile_reverse: Mapping[str, str],
    action_reverse: Mapping[str, str],
    line_number: int,
) -> dict:
    context = f"JSONL line {line_number}"

    for field in ("prompt-base", "query", "answer"):
        if field not in record:
            raise VocabularyError(f"{context}: missing required field {field!r}.")
        if not isinstance(record[field], str):
            raise VocabularyError(f"{context}: field {field!r} must be a string.")

    converted = dict(record)
    converted["prompt-base"] = convert_prompt(
        record["prompt-base"],
        custom_row_marker,
        tile_reverse,
        action_reverse,
        context,
    )
    converted["query"] = convert_query(
        record["query"],
        custom_row_marker,
        tile_reverse,
        f"{context}, query",
    )
    converted["answer"] = convert_action_sequence(
        record["answer"],
        action_reverse,
        f"{context}, answer",
    )
    return converted


def convert_file(
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    overwrite: bool,
) -> int:
    custom_row_marker, tile_reverse, action_reverse = load_vocabulary(mapping_path)

    try:
        same_path = input_path.resolve() == output_path.resolve()
    except OSError:
        same_path = input_path == output_path

    if output_path.exists() and not overwrite:
        raise VocabularyError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    if same_path and not overwrite:
        raise VocabularyError(
            "Input and output paths are the same. Use --overwrite for an atomic "
            "in-place conversion."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    converted_count = 0

    try:
        with input_path.open("r", encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_name = destination.name

                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        destination.write(line)
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise VocabularyError(
                            f"JSONL line {line_number}: invalid JSON: {exc}"
                        ) from exc

                    if not isinstance(record, dict):
                        raise VocabularyError(
                            f"JSONL line {line_number}: record must be a JSON object."
                        )

                    converted = convert_record(
                        record,
                        custom_row_marker,
                        tile_reverse,
                        action_reverse,
                        line_number,
                    )
                    destination.write(
                        json.dumps(converted, ensure_ascii=False) + "\n"
                    )
                    converted_count += 1

        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)

    return converted_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a custom CipherGrid row marker, tile vocabulary, and action "
            "vocabulary to canonical symbology in a JSONL benchmark."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Custom JSONL.")
    parser.add_argument(
        "--mapping",
        required=True,
        type=Path,
        help="The same custom vocabulary JSON used for forward conversion.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Canonical output JSONL."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file, including atomic in-place conversion.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        count = convert_file(
            args.input,
            args.output,
            args.mapping,
            args.overwrite,
        )
    except (OSError, VocabularyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Restored {count} JSONL records to canonical vocabulary: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
