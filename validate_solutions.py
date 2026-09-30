#!/usr/bin/env python3
"""
validate_solutions.py

Inputs:
  1) JSONL file with keys: id, query, answer
  2) Solutions CSV with columns: id, response

Output:
  CSV with columns: id, response, valid

Where:
  valid=1 if either:
    - the provided response matches the JSONL answer, OR
    - the provided response can be legally replayed to end on goal tile 'da'
  else 0

Notes:
- Handles nondeterminism by propagating a set of possible states using solver.successors().
- If an id in JSONL is missing from solutions CSV, response is treated as "" and valid=0.
- Matching is normalized to ignore casing, BOM, and extra whitespace around separators.

Usage:
  python3 validate_solutions.py --jsonl puzzles.jsonl --solutions solutions.csv --out validation.csv
  python3 validate_solutions.py --decode --jsonl puzzles.jsonl --solutions solutions.csv --out validation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, List, Set

import solver
import decode_world


def _norm(s: str) -> str:
    return (s or "").strip().lstrip("\ufeff").strip().lower()


def normalize_action_string(action_str: str) -> str:
    s = _norm(action_str)
    if not s:
        return ""
    toks = [tok.strip() for tok in s.split("-") if tok.strip()]
    return "-".join(toks)


def load_solutions_csv(path: str) -> Dict[str, str]:
    """
    Load solutions into dict: id -> response.
    Requires columns id and response (case/BOM tolerant).
    If duplicate ids exist, the LAST one wins.
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return out

        h = [_norm(x) for x in header]
        if "id" not in h or "response" not in h:
            raise RuntimeError(f'Solutions CSV must have columns "id" and "response": {path}')

        id_i = h.index("id")
        sol_i = h.index("response")

        for row in reader:
            if not row:
                continue
            if id_i >= len(row):
                continue
            rid = str(row[id_i]).strip()
            if not rid:
                continue
            sol = ""
            if sol_i < len(row):
                sol = str(row[sol_i]).strip()
            out[rid] = sol
    return out


def parse_actions(action_str: str) -> List[str]:
    s = normalize_action_string(action_str)
    if not s:
        return []
    return [tok.strip() for tok in s.split("-") if tok.strip()]


def parse_world_text(text: str) -> List[List[str]]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty world text.")

    parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for seg in line.split(","):
            seg = seg.strip()
            if seg:
                parts.append(seg)

    rows: List[List[str]] = []
    for p in parts:
        if not p.startswith("wa:"):
            continue
        tiles = [t.strip() for t in p[3:].split("-") if t.strip()]
        rows.append(tiles)

    if not rows:
        raise ValueError("No 'wa:' rows found in query/world.")

    W = max(len(r) for r in rows)
    for r in rows:
        if len(r) < W:
            r.extend([solver.WALL] * (W - len(r)))

    return rows


def validate_ends_on_goal(grid: List[List[str]], action_str: str) -> bool:
    """
    Faithful replay validator using solver.successors() to handle nondeterminism
    (e.g., 'ra' can choose any adjacent legal tile).
    """
    swords0, planks0, monsters0, traps_mask, starts, goal = solver.build_masks(grid)

    cur: Set[solver.State] = set(
        solver.State(sr, sc, 0, swords0, planks0, monsters0, 0)
        for (sr, sc) in starts
    )

    for act in parse_actions(action_str):
        nxt: Set[solver.State] = set()
        for st in cur:
            for a2, st2 in solver.successors(grid, st, traps_mask):
                if a2 == act:
                    nxt.add(st2)
        if not nxt:
            return False
        cur = nxt

    gr, gc = goal
    return any((st.r, st.c) == (gr, gc) for st in cur)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="JSONL with keys: id, query, answer")
    ap.add_argument("--solutions", required=True, help='Solutions CSV with columns "id","response"')
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--decode", action="store_true", help="Decode query using decode_world.decode_world first")
    args = ap.parse_args()

    sol_map = load_solutions_csv(args.solutions)

    with open(args.jsonl, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8", newline="") as fout:

        w = csv.writer(fout)
        w.writerow(["id", "response", "valid"])

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                rid = str(obj["id"]).strip()
                query = obj["query"]
                answer = str(obj.get("answer", "")).strip()

                if not isinstance(query, str):
                    raise TypeError("query must be a string")

                response = sol_map.get(rid, "")

                # First validity route: exact answer match
                answer_matches = (
                    normalize_action_string(response) == normalize_action_string(answer)
                    if response and answer
                    else False
                )

                # Second validity route: replay validator
                replay_valid = False
                if response:
                    world_text = decode_world.decode_world(query) if args.decode else query
                    grid = parse_world_text(world_text)
                    replay_valid = validate_ends_on_goal(grid, response)

                ok = answer_matches or replay_valid
                w.writerow([rid, response, "1" if ok else "0"])

            except Exception as e:
                rid = ""
                try:
                    rid = str(json.loads(line).get("id", "")).strip()
                except Exception:
                    pass
                print(f"[line {line_no}] error: {e}", file=sys.stderr)
                w.writerow([rid, sol_map.get(rid, ""), "0"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
