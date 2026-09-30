#!/usr/bin/env python3
"""
Apply a custom CipherGrid vocabulary to a JSONL dataset.

This converter supports prompt-base strings whose line breaks are stored either
as real newlines or as literal ``\n`` characters. It converts:

1. Demonstration grid rows inside prompt-base:
      wa:da-pa-pa
2. Demonstration action responses inside prompt-base:
      -> ga-ma-ma
3. Fenced action examples inside prompt-base:
      ---ma-ga-va---
4. query and answer fields, detecting whether each contains:
      - a hyphen-delimited action sequence, or
      - a delimiter-free compact grid encoding.

The output is written atomically and then verified against the source file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


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

CANONICAL_TILE_CODES = tuple(TILES.values())
CANONICAL_ACTION_CODES = tuple(ACTIONS.values())
CANONICAL_GRID_CODES = set(CANONICAL_TILE_CODES) | {CANONICAL_ROW_MARKER}

_TILE_ALT = "|".join(map(re.escape, CANONICAL_TILE_CODES))
_ACTION_ALT = "|".join(map(re.escape, CANONICAL_ACTION_CODES))

GRID_ROW_PATTERN = re.compile(
    rf"{re.escape(CANONICAL_ROW_MARKER)}:(?:{_TILE_ALT})(?:-(?:{_TILE_ALT}))*"
)
ARROW_ACTION_PATTERN = re.compile(
    rf"(?P<prefix>->\s*)(?P<sequence>(?:{_ACTION_ALT})(?:-(?:{_ACTION_ALT}))*)"
)
FENCED_ACTION_PATTERN = re.compile(
    rf"(?P<prefix>---)(?P<sequence>(?:{_ACTION_ALT})(?:-(?:{_ACTION_ALT}))*)(?P<suffix>---)"
)


class ConversionError(ValueError):
    pass


def load_mapping(path: Path) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Could not read mapping file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConversionError("Mapping must be a JSON object.")

    row_marker = data.get("row_marker")
    tiles = data.get("tiles")
    actions = data.get("actions")

    if not isinstance(row_marker, str) or not row_marker:
        raise ConversionError("'row_marker' must be a non-empty string.")
    if not isinstance(tiles, dict):
        raise ConversionError("'tiles' must be an object.")
    if not isinstance(actions, dict):
        raise ConversionError("'actions' must be an object.")

    missing_tiles = [name for name in TILES if name not in tiles]
    missing_actions = [name for name in ACTIONS if name not in actions]
    if missing_tiles:
        raise ConversionError(f"Missing tile mappings: {', '.join(missing_tiles)}")
    if missing_actions:
        raise ConversionError(f"Missing action mappings: {', '.join(missing_actions)}")

    extra_tiles = [name for name in tiles if name not in TILES]
    extra_actions = [name for name in actions if name not in ACTIONS]
    if extra_tiles:
        raise ConversionError(f"Unknown tile mappings: {', '.join(extra_tiles)}")
    if extra_actions:
        raise ConversionError(f"Unknown action mappings: {', '.join(extra_actions)}")

    tile_map = {TILES[name]: str(tiles[name]) for name in TILES}
    action_map = {ACTIONS[name]: str(actions[name]) for name in ACTIONS}

    all_tokens = [row_marker, *tile_map.values(), *action_map.values()]
    if any(token == "" for token in all_tokens):
        raise ConversionError("Custom tokens cannot be empty.")
    if len(all_tokens) != len(set(all_tokens)):
        raise ConversionError("Every custom token must be unique.")
    if any("-" in token or ":" in token or "," in token for token in all_tokens):
        raise ConversionError("Custom tokens cannot contain '-', ':' or ','.")
    if any(any(character.isspace() for character in token) for token in all_tokens):
        raise ConversionError("Custom tokens cannot contain whitespace.")

    # Compact grids have no delimiter. Equal-width row/tile tokens make decoding
    # deterministic. Action token widths do not matter because actions use '-'.
    compact_widths = {len(row_marker), *(len(token) for token in tile_map.values())}
    if len(compact_widths) != 1:
        raise ConversionError(
            "row_marker and all tile tokens must have the same character width "
            "because compact grid fields contain no delimiters."
        )

    return row_marker, tile_map, action_map


def convert_grid_row(
    encoded_row: str,
    row_marker: str,
    tile_map: Mapping[str, str],
) -> str:
    canonical_marker, encoded_cells = encoded_row.split(":", 1)
    if canonical_marker != CANONICAL_ROW_MARKER:
        raise ConversionError(f"Unexpected row marker: {canonical_marker!r}")

    cells = encoded_cells.split("-")
    try:
        converted_cells = [tile_map[cell] for cell in cells]
    except KeyError as exc:
        raise ConversionError(f"Unknown tile token in prompt row: {exc.args[0]!r}") from exc

    return f"{row_marker}:" + "-".join(converted_cells)


def convert_action_sequence(
    sequence: str,
    action_map: Mapping[str, str],
) -> str:
    tokens = sequence.split("-")
    try:
        return "-".join(action_map[token] for token in tokens)
    except KeyError as exc:
        raise ConversionError(f"Unknown action token: {exc.args[0]!r}") from exc


def convert_prompt(
    prompt: str,
    row_marker: str,
    tile_map: Mapping[str, str],
    action_map: Mapping[str, str],
) -> Tuple[str, Dict[str, int]]:
    counts = {
        "grid_rows": 0,
        "arrow_sequences": 0,
        "fenced_sequences": 0,
    }

    def replace_grid(match: re.Match[str]) -> str:
        counts["grid_rows"] += 1
        return convert_grid_row(match.group(0), row_marker, tile_map)

    def replace_arrow(match: re.Match[str]) -> str:
        counts["arrow_sequences"] += 1
        return (
            match.group("prefix")
            + convert_action_sequence(match.group("sequence"), action_map)
        )

    def replace_fenced(match: re.Match[str]) -> str:
        counts["fenced_sequences"] += 1
        return (
            match.group("prefix")
            + convert_action_sequence(match.group("sequence"), action_map)
            + match.group("suffix")
        )

    # These regexes work whether separators are actual newlines or literal '\n'.
    converted = GRID_ROW_PATTERN.sub(replace_grid, prompt)
    converted = ARROW_ACTION_PATTERN.sub(replace_arrow, converted)
    converted = FENCED_ACTION_PATTERN.sub(replace_fenced, converted)

    return converted, counts


def is_action_sequence(value: str) -> bool:
    if not value:
        return False
    return all(token in ACTIONS.values() for token in value.split("-"))


def is_compact_grid(value: str) -> bool:
    if not value or len(value) % 2 != 0:
        return False
    return all(
        value[index:index + 2] in CANONICAL_GRID_CODES
        for index in range(0, len(value), 2)
    )


def convert_compact_grid(
    value: str,
    row_marker: str,
    tile_map: Mapping[str, str],
) -> str:
    converted = []
    for index in range(0, len(value), 2):
        token = value[index:index + 2]
        if token == CANONICAL_ROW_MARKER:
            converted.append(row_marker)
        elif token in tile_map:
            converted.append(tile_map[token])
        else:
            raise ConversionError(
                f"Unknown compact-grid token {token!r} at character offset {index}."
            )
    return "".join(converted)


def convert_encoded_field(
    value: str,
    row_marker: str,
    tile_map: Mapping[str, str],
    action_map: Mapping[str, str],
    context: str,
) -> Tuple[str, str]:
    action_match = is_action_sequence(value)
    grid_match = is_compact_grid(value)

    if action_match and grid_match:
        raise ConversionError(f"{context} is ambiguous.")
    if action_match:
        return convert_action_sequence(value, action_map), "actions"
    if grid_match:
        return convert_compact_grid(value, row_marker, tile_map), "grid"

    preview = value[:100] + ("..." if len(value) > 100 else "")
    raise ConversionError(
        f"{context} is neither a canonical action sequence nor a compact grid: "
        f"{preview!r}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(
                    f"JSONL line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ConversionError(
                    f"JSONL line {line_number} is not a JSON object."
                )
            records.append(record)
    return records


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def verify_output(
    source_records: list[dict[str, Any]],
    output_records: list[dict[str, Any]],
    row_marker: str,
    tile_map: Mapping[str, str],
    action_map: Mapping[str, str],
) -> None:
    if len(source_records) != len(output_records):
        raise ConversionError(
            f"Verification failed: source has {len(source_records)} records but "
            f"output has {len(output_records)}."
        )

    immutable_fields = {"prompt-base", "query", "answer"}

    custom_action_tokens = set(action_map.values())
    compact_width = len(row_marker)
    custom_grid_tokens = set(tile_map.values()) | {row_marker}

    for index, (source, output) in enumerate(
        zip(source_records, output_records), start=1
    ):
        if set(source) != set(output):
            raise ConversionError(
                f"Verification failed on line {index}: field names changed."
            )

        for key in source:
            if key not in immutable_fields and source[key] != output[key]:
                raise ConversionError(
                    f"Verification failed on line {index}: field {key!r} changed."
                )

        prompt = output["prompt-base"]
        if GRID_ROW_PATTERN.search(prompt):
            raise ConversionError(
                f"Verification failed on line {index}: canonical grid remains in prompt-base."
            )
        if ARROW_ACTION_PATTERN.search(prompt):
            raise ConversionError(
                f"Verification failed on line {index}: canonical arrow action remains."
            )
        if FENCED_ACTION_PATTERN.search(prompt):
            raise ConversionError(
                f"Verification failed on line {index}: canonical fenced action remains."
            )

        for field_name in ("query", "answer"):
            source_value = source[field_name]
            output_value = output[field_name]

            if is_action_sequence(source_value):
                tokens = output_value.split("-")
                if not tokens or not all(token in custom_action_tokens for token in tokens):
                    raise ConversionError(
                        f"Verification failed on line {index}: {field_name} "
                        "is not a valid custom action sequence."
                    )
            elif is_compact_grid(source_value):
                if len(output_value) % compact_width != 0:
                    raise ConversionError(
                        f"Verification failed on line {index}: {field_name} "
                        "has invalid custom compact-grid length."
                    )
                tokens = [
                    output_value[position:position + compact_width]
                    for position in range(0, len(output_value), compact_width)
                ]
                if not tokens or not all(token in custom_grid_tokens for token in tokens):
                    raise ConversionError(
                        f"Verification failed on line {index}: {field_name} "
                        "contains an invalid custom grid token."
                    )
            else:
                raise ConversionError(
                    f"Verification failed on line {index}: source {field_name} "
                    "has an unsupported encoding."
                )


def convert_file(
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    overwrite: bool,
) -> None:
    if not input_path.is_file():
        raise ConversionError(f"Input file not found: {input_path}")
    if not mapping_path.is_file():
        raise ConversionError(f"Mapping file not found: {mapping_path}")
    if output_path.exists() and not overwrite:
        raise ConversionError(
            f"Output already exists: {output_path}. Add --overwrite to replace it."
        )

    row_marker, tile_map, action_map = load_mapping(mapping_path)
    source_records = read_jsonl(input_path)

    output_records = []
    total_grid_rows = 0
    total_arrow_sequences = 0
    total_fenced_sequences = 0
    layouts: Dict[str, int] = {}

    for line_number, source in enumerate(source_records, start=1):
        for required_field in ("prompt-base", "query", "answer"):
            if required_field not in source:
                raise ConversionError(
                    f"JSONL line {line_number} is missing {required_field!r}."
                )
            if not isinstance(source[required_field], str):
                raise ConversionError(
                    f"JSONL line {line_number}, {required_field}: expected a string."
                )

        output = dict(source)

        converted_prompt, counts = convert_prompt(
            source["prompt-base"],
            row_marker,
            tile_map,
            action_map,
        )
        output["prompt-base"] = converted_prompt

        output["query"], query_type = convert_encoded_field(
            source["query"],
            row_marker,
            tile_map,
            action_map,
            f"JSONL line {line_number}, query",
        )
        output["answer"], answer_type = convert_encoded_field(
            source["answer"],
            row_marker,
            tile_map,
            action_map,
            f"JSONL line {line_number}, answer",
        )

        layout = f"query={query_type}, answer={answer_type}"
        layouts[layout] = layouts.get(layout, 0) + 1

        total_grid_rows += counts["grid_rows"]
        total_arrow_sequences += counts["arrow_sequences"]
        total_fenced_sequences += counts["fenced_sequences"]
        output_records.append(output)

    write_jsonl_atomic(output_path, output_records)

    # Re-open the exact saved file before reporting success.
    saved_records = read_jsonl(output_path)
    verify_output(
        source_records,
        saved_records,
        row_marker,
        tile_map,
        action_map,
    )

    print(f"Converted and verified {len(saved_records)} records.")
    print(f"Prompt grid rows converted: {total_grid_rows}")
    print(f"Prompt arrow action sequences converted: {total_arrow_sequences}")
    print(f"Prompt fenced action examples converted: {total_fenced_sequences}")
    for layout, count in sorted(layouts.items()):
        print(f"{layout}: {count}")
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a custom CipherGrid vocabulary to a JSONL dataset."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        convert_file(
            args.input,
            args.output,
            args.mapping,
            args.overwrite,
        )
    except (OSError, ConversionError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
