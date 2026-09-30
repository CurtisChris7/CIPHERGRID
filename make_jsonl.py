"""
Create a JSONL dataset where each line has the form:

{
  "prompt-base": "...",
  "query": "...",
  "answer": "...",
  "image": "<base64 string>",
  "complex": <bool>,
  "size": "<string>"
}

Global fields:
- `complex` (bool): Applied uniformly to all records. Enabled via --complex.
- `size` (string): Applied uniformly to all records. Provided via --size.

Inputs:
- Prompt-base file (applied to all records)
- Queries file (1 query per line)
- Answers file (1 answer per line)
- A single image file, encoded once and reused for all records

Options:
- --allow-mismatch: Allow query/answer line-count mismatch (truncate to shortest)
- --compact: Write compact JSON (no spaces)
- --complex: Mark all records as complex (default: false)
- --size: Size label applied to all records (e.g., small, medium, large)

Example:
  python make_jsonl.py \
    --prompt prompt.txt \
    --queries queries.txt \
    --answers answers.txt \
    --image example.png \
    --size medium \
    --complex \
    --out dataset.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import List


TILES = {
    "ca": "WALL",
    "pa": "OPEN",
    "da": "GOAL",
    "na": "START",
    "ea": "MONSTER",
    "xa": "REVERSAL",
    "oa": "SWORD",
    "la": "PLANK",
    "ha": "WATER",
    "ka": "TRAP",
    "wa": "ROW"
}

ACTIONS = {
    "ma": "UP",
    "ga": "RIGHT",
    "za": "LEFT",
    "va": "DOWN",
    "ya": "PICKUP",
    "ba": "ATTACK",
    "sa": "PLACE",
    "ra": "SWIM",
    "ta": "RESIGN",
}

ROW_PREFIX = "wa:"


# =========================
# Universal Decoder
# =========================
# =========================
# Universal Decoder
# =========================
def _chunk_or_split_tokens(s: str) -> list[str]:
    """
    Tokenize either:
      - separated: 'wa:oa-la-xa' / 'ma-ga-za' / space/newline separated
      - compact:   'waxaeaeana...' (2-char tokens)
    """
    s = s.strip()
    if not s:
        return []

    # If it's already separated, use that.
    if ("-" in s) or (" " in s) or ("\n" in s) or ("\t" in s):
        parts = [p.strip() for p in s.replace("\t", " ").replace("\n", "-").replace(" ", "-").split("-")]
        return [p for p in parts if p]

    # Otherwise: compact string -> 2-char chunking
    compact = "".join(ch for ch in s if ch.isalpha())

    # If odd length, fall back to "unknown whole token"
    if len(compact) % 2 != 0:
        return [s]

    return [compact[i:i+2] for i in range(0, len(compact), 2)]


def _format_rows_from_decoded_tokens(decoded_tokens: list[str]) -> str | None:
    """
    If decoded_tokens contains ROW separators, format as:
      ROW:...-...,ROW:...-...
    Return None if no ROW structure is present.
    """
    if "ROW" not in decoded_tokens:
        return None

    rows: list[list[str]] = []
    current: list[str] | None = None

    for tok in decoded_tokens:
        if tok == "ROW" or tok == "ROW:":
            if current is not None:
                rows.append(current)
            current = []
        else:
            if current is None:
                # if it didn't start with ROW, still treat as first row
                current = []
            current.append(tok)

    if current is not None:
        rows.append(current)

    # if it's just "ROW" with nothing else, bail
    if not rows or all(len(r) == 0 for r in rows):
        return None

    out = []
    for i, r in enumerate(rows):
        row_str = "ROW:" + "-".join(r)
        if i > 0:
            row_str = "," + row_str
        out.append(row_str)

    return "".join(out)


def decode_string(s: str) -> str:
    """
    Takes a single string argument.
    Automatically detects grid or action string.
    Returns a decoded string.

    Behavior:
    - If input contains 'wa:' lines => returns comma-separated ROW:... strings
    - Else if token stream contains 'wa' (ROW tile token) => returns comma-separated ROW:... strings
    - Else returns space-separated decoded tokens
    """
    s = s.strip()
    if not s:
        return ""

    # ---- GRID MODE A: explicit wa: line-prefix grids ----
    if ROW_PREFIX in s:
        out_rows: list[str] = []

        for raw in s.splitlines():
            line = raw.strip()
            if not line or not line.startswith(ROW_PREFIX):
                continue

            payload = line[len(ROW_PREFIX):].strip()
            tokens = _chunk_or_split_tokens(payload)
            decoded_tiles = [TILES.get(t, f"UNK({t})") for t in tokens]

            row_str = "ROW:" + "-".join(decoded_tiles)
            if out_rows:
                row_str = "," + row_str
            out_rows.append(row_str)

        return "".join(out_rows)

    # Tokenize once for the remaining modes
    tokens = _chunk_or_split_tokens(s)

    # ---- GRID MODE B: compact/non-wa: grids using 'wa' token as row delimiter ----
    # If the raw token stream contains the ROW tile token ("wa"), treat it as row separator.
    if "wa" in tokens:
        # Split on 'wa' (ROW token). Ignore leading empty chunk if string starts with 'wa'.
        rows_tokens: list[list[str]] = []
        cur: list[str] = []
        seen_any = False

        for t in tokens:
            if t == "wa":
                if seen_any:  # we only push if we've started collecting a row
                    rows_tokens.append(cur)
                cur = []
                seen_any = True
            else:
                cur.append(t)

        if seen_any:
            rows_tokens.append(cur)

        # Decode each row as tiles
        out = []
        for i, row in enumerate(rows_tokens):
            decoded_tiles = [TILES.get(t, f"UNK({t})") for t in row]
            row_str = "ROW:" + "-".join(decoded_tiles)
            if i > 0:
                row_str = "," + row_str
            out.append(row_str)

        return "".join(out)

    # ---- ACTION / GENERIC MODE ----
    decoded = []
    for t in tokens:
        if t in ACTIONS:
            decoded.append(ACTIONS[t])
        elif t in TILES:
            decoded.append(TILES[t])
        else:
            decoded.append(f"UNK({t})")

    # If generic decoding yielded ROW-separated content, format it as you requested.
    maybe_rows = _format_rows_from_decoded_tokens(decoded)
    if maybe_rows is not None:
        return maybe_rows

    return " ".join(decoded)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n\r")


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create JSONL with base64 image field.")
    ap.add_argument("--complex", action="store_true",
                help="Mark all records as complex.")
    ap.add_argument("--size", type=str, required=True,
                help="Size label applied to all records (e.g., small, medium, large).")
    ap.add_argument("--prompt", required=True, help="Prompt-base file.")
    ap.add_argument("--decode", action="store_true", help="Decodes the gridworld")
    ap.add_argument("--queries", required=True, help="Queries file (1 per line).")
    ap.add_argument("--answers", required=True, help="Answers file (1 per line).")
    ap.add_argument("--image", required=True, help="Image file to encode as base64.")
    ap.add_argument("--out", required=True, help="Output JSONL file.")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="Allow query/answer count mismatch (truncate to shortest).")
    ap.add_argument("--compact", action="store_true",
                    help="Write compact JSON (no spaces).")

    args = ap.parse_args()

    prompt_path = Path(args.prompt)
    queries_path = Path(args.queries)
    answers_path = Path(args.answers)
    image_path = Path(args.image)
    out_path = Path(args.out)

    for p in [prompt_path, queries_path, answers_path, image_path]:
        if not p.exists():
            sys.exit(f"ERROR: File not found: {p}")

    prompt_base = read_text(prompt_path)
    queries = read_lines(queries_path)
    answers = read_lines(answers_path)

    if len(queries) != len(answers):
        msg = f"ERROR: Line count mismatch: queries={len(queries)} answers={len(answers)}"
        if not args.allow_mismatch:
            sys.exit(msg)
        print("WARNING:", msg, file=sys.stderr)

    n = min(len(queries), len(answers))

    image_b64 = encode_image_b64(image_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dump_kwargs = {"ensure_ascii": False}
    if args.compact:
        dump_kwargs["separators"] = (",", ":")

    


    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            if args.decode:
                queries[i] = decode_string(queries[i])

            obj = {
                "prompt-base": prompt_base,
                "query": queries[i],
                "answer": answers[i],
                "image": image_b64,
                "complex": args.complex,
                "size": args.size,
            }
            f.write(json.dumps(obj, **dump_kwargs))
            f.write("\n")

    print(f"Wrote {n} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
