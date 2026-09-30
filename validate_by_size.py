#!/usr/bin/env python3
"""
validate_by_size.py

Validate model responses, write per-record validity, and print accuracy grouped by
the ID buckets created in benchmark_metadata.ipynb.

The notebook builds:
  - SIZE_MAP[size] -> [id, ...]
  - REAL_SIZE_MAP[real_size] -> [id, ...]
  - SOLUTION_LENGTH_MAP[solution_length] -> [id, ...]
  - UNSOLVABLE -> [id, ...] when answer == "ta"

This script reconstructs those same buckets directly from the JSONL so you do
not need to export anything from the notebook.

Inputs:
  1) JSONL file with keys: id, query, answer, and usually size
  2) Solutions CSV with columns: id, response

Validity:
  - Solvable records:
      valid=1 if response exactly matches answer after normalization OR
      response can be legally replayed to end on goal tile 'da'
  - Unsolvable records, where normalized answer == "ta":
      valid=1 only if normalized response == "ta"

Outputs:
  - Optional per-record CSV via --out
  - Optional grouped-statistics CSV via --stats-out
  - Printed accuracy tables grouped by --group-by

Usage:
  # Accuracy by benchmark size field, including separate unsolvable summaries
  python3 validate_by_size.py --jsonl benchmarkv3.jsonl --solutions model_outputs.csv --out validation.csv \
      --stats-out validation_by_size.csv \
      --group-by size

  # Accuracy by solution length, matching the distribution chart buckets
  python3 validate_by_size.py \
      --jsonl benchmarkv3.jsonl \
      --solutions model_outputs.csv \
      --out validation.csv \
      --stats-out validation_by_solution_length.csv \
      --group-by solution_length

  # If queries need decoding first
  python3 validate_by_size.py \
      --decode \
      --jsonl benchmarkv3.jsonl \
      --solutions model_outputs.csv \
      --out validation.csv \
      --group-by size
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import solver

# Increase the limit to 1MB or higher as needed
csv.field_size_limit(1024 * 1024) 


def _norm(s: Any) -> str:
    return ("" if s is None else str(s)).strip().lstrip("\ufeff").strip().lower()


def normalize_action_string(action_str: Any) -> str:
    s = _norm(action_str)
    if not s:
        return ""
    toks = [tok.strip() for tok in s.split("-") if tok.strip()]
    return "-".join(toks)


def parse_actions(action_str: str) -> List[str]:
    s = normalize_action_string(action_str)
    if not s:
        return []
    return [tok.strip() for tok in s.split("-") if tok.strip()]


def solution_length(answer: Any) -> Optional[int]:
    s = normalize_action_string(answer)
    if not s:
        return None
    return len(s.split("-"))


def load_solutions_csv(path: str) -> Dict[str, str]:
    """
    Load solutions into dict: id -> response.

    Requires columns id and response, case/BOM tolerant.
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
            if not row or id_i >= len(row):
                continue

            rid = str(row[id_i]).strip()
            if not rid:
                continue

            response = ""
            if sol_i < len(row):
                response = str(row[sol_i]).strip()

            out[rid] = response

    return out


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

    width = max(len(r) for r in rows)
    for r in rows:
        if len(r) < width:
            r.extend([solver.WALL] * (width - len(r)))

    return rows


def validate_ends_on_goal(grid: List[List[str]], action_str: str) -> bool:
    """
    Faithful replay validator using solver.successors() to handle nondeterminism
    such as 'ra' choosing any adjacent legal tile.
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


def get_group_value(record: Dict[str, Any], group_by: str) -> str:
    if group_by == "size":
        return str(record.get("size", "MISSING_SIZE"))

    if group_by == "real_size":
        query = str(record.get("query", ""))
        return str(len(query) // 2)

    if group_by == "solution_length":
        slen = solution_length(record.get("answer", ""))
        return "MISSING_SOLUTION_LENGTH" if slen is None else str(slen)

    raise ValueError(f"Unsupported group_by: {group_by}")


@dataclass
class RowResult:
    rid: str
    group: str
    size: str
    real_size: str
    solution_length: str
    is_unsolvable: bool
    answer: str
    response: str
    exact_match: bool
    replay_valid: bool
    valid: bool
    error: str = ""


@dataclass
class Counter:
    n: int = 0
    correct: int = 0

    def add(self, valid: bool) -> None:
        self.n += 1
        self.correct += int(valid)

    @property
    def accuracy(self) -> float:
        if self.n == 0:
            return 0.0
        return self.correct / self.n


def sort_key(x: str) -> Tuple[int, Any]:
    try:
        return (0, int(x))
    except Exception:
        try:
            return (0, float(x))
        except Exception:
            return (1, x)


def print_table(title: str, rows: List[Tuple[str, Counter]], group_label: str, min_count: int = 1) -> None:
    rows = [(g, c) for g, c in rows if c.n >= min_count]

    print()
    print(title)
    print("-" * len(title))

    if not rows:
        print("(no rows)")
        return

    g_width = max(len(group_label), max(len(str(g)) for g, _ in rows))
    n_width = max(5, max(len(str(c.n)) for _, c in rows))
    c_width = max(7, max(len(str(c.correct)) for _, c in rows))

    print(f"{group_label:<{g_width}}  {'n':>{n_width}}  {'correct':>{c_width}}  {'accuracy':>9}")
    print(f"{'-' * g_width}  {'-' * n_width}  {'-' * c_width}  {'-' * 9}")

    for group, counter in rows:
        print(
            f"{str(group):<{g_width}}  "
            f"{counter.n:>{n_width}}  "
            f"{counter.correct:>{c_width}}  "
            f"{counter.accuracy:>8.2%}"
        )


def write_row_csv(path: str, results: Iterable[RowResult]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow([
            "id",
            "group",
            "size",
            "real_size",
            "solution_length",
            "is_unsolvable",
            "answer",
            "response",
            "exact_match",
            "replay_valid",
            "valid",
            "error",
        ])

        for r in results:
            w.writerow([
                r.rid,
                r.group,
                r.size,
                r.real_size,
                r.solution_length,
                "1" if r.is_unsolvable else "0",
                r.answer,
                r.response,
                "1" if r.exact_match else "0",
                "1" if r.replay_valid else "0",
                "1" if r.valid else "0",
                r.error,
            ])


def write_stats_csv(
    path: str,
    group_by: str,
    overall: Counter,
    solvable_overall: Counter,
    unsolvable_overall: Counter,
    by_group_all: Dict[str, Counter],
    by_group_solvable: Dict[str, Counter],
    by_group_unsolvable: Dict[str, Counter],
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["section", "group_by", "group", "n", "correct", "accuracy"])

        def emit(section: str, group: str, counter: Counter) -> None:
            w.writerow([
                section,
                group_by,
                group,
                counter.n,
                counter.correct,
                f"{counter.accuracy:.6f}",
            ])

        emit("overall", "ALL", overall)
        emit("overall_solvable", "SOLVABLE", solvable_overall)
        emit("overall_unsolvable", "UNSOLVABLE", unsolvable_overall)

        for group, counter in sorted(by_group_all.items(), key=lambda kv: sort_key(kv[0])):
            emit("by_group_all", group, counter)

        for group, counter in sorted(by_group_solvable.items(), key=lambda kv: sort_key(kv[0])):
            emit("by_group_solvable", group, counter)

        for group, counter in sorted(by_group_unsolvable.items(), key=lambda kv: sort_key(kv[0])):
            emit("by_group_unsolvable", group, counter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="JSONL with keys: id, query, answer, and optionally size")
    ap.add_argument("--solutions", required=True, help='Solutions CSV with columns "id" and "response"')
    ap.add_argument("--out", default="", help="Optional per-record validation CSV")
    ap.add_argument("--stats-out", default="", help="Optional grouped-statistics CSV")
    ap.add_argument(
        "--group-by",
        choices=["size", "real_size", "solution_length"],
        default="size",
        help="Which notebook-style ID bucket to use for the printed accuracy table.",
    )
    ap.add_argument("--decode", action="store_true", help="Decode query using decode_world.decode_world before replay")
    ap.add_argument("--min-count", type=int, default=1, help="Only print groups with at least this many records")
    args = ap.parse_args()

    decode_world = None
    if args.decode:
        import decode_world as _decode_world
        decode_world = _decode_world

    sol_map = load_solutions_csv(args.solutions)

    results: List[RowResult] = []

    overall = Counter()
    solvable_overall = Counter()
    unsolvable_overall = Counter()

    by_group_all: DefaultDict[str, Counter] = defaultdict(Counter)
    by_group_solvable: DefaultDict[str, Counter] = defaultdict(Counter)
    by_group_unsolvable: DefaultDict[str, Counter] = defaultdict(Counter)

    with open(args.jsonl, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            rid = ""
            response = ""
            answer = ""
            group = "ERROR"
            size_value = ""
            real_size_value = ""
            solution_length_value = ""
            is_unsolvable = False
            exact_match = False
            replay_valid = False
            valid = False
            error = ""

            try:
                record = json.loads(line)
                rid = str(record["id"]).strip()
                query = record["query"]
                answer = str(record.get("answer", "")).strip()
                response = sol_map.get(rid, "")

                if not isinstance(query, str):
                    raise TypeError("query must be a string")

                size_value = str(record.get("size", "MISSING_SIZE"))
                real_size_value = str(len(query) // 2)
                slen = solution_length(answer)
                solution_length_value = "MISSING_SOLUTION_LENGTH" if slen is None else str(slen)

                group = get_group_value(record, args.group_by)

                norm_answer = normalize_action_string(answer)
                norm_response = normalize_action_string(response)
                is_unsolvable = norm_answer == "ta"

                exact_match = bool(norm_response and norm_answer and norm_response == norm_answer)

                if is_unsolvable:
                    # For unsolvable records, only the explicit RESIGN token is correct.
                    replay_valid = False
                    valid = exact_match
                else:
                    if response:
                        world_text = decode_world.decode_world(query) if decode_world else query
                        grid = parse_world_text(world_text)
                        replay_valid = validate_ends_on_goal(grid, response)
                    valid = exact_match or replay_valid

            except Exception as e:
                error = str(e)
                print(f"[line {line_no}] error: {e}", file=sys.stderr)
                try:
                    maybe_record = json.loads(line)
                    rid = str(maybe_record.get("id", "")).strip()
                    response = sol_map.get(rid, "")
                    answer = str(maybe_record.get("answer", "")).strip()
                    norm_answer = normalize_action_string(answer)
                    is_unsolvable = norm_answer == "ta"
                    group = get_group_value(maybe_record, args.group_by)
                    size_value = str(maybe_record.get("size", "MISSING_SIZE"))
                    query = str(maybe_record.get("query", ""))
                    real_size_value = str(len(query) // 2)
                    slen = solution_length(answer)
                    solution_length_value = "MISSING_SOLUTION_LENGTH" if slen is None else str(slen)
                except Exception:
                    pass

            result = RowResult(
                rid=rid,
                group=group,
                size=size_value,
                real_size=real_size_value,
                solution_length=solution_length_value,
                is_unsolvable=is_unsolvable,
                answer=answer,
                response=response,
                exact_match=exact_match,
                replay_valid=replay_valid,
                valid=valid,
                error=error,
            )
            results.append(result)

            overall.add(valid)
            by_group_all[group].add(valid)

            if is_unsolvable:
                unsolvable_overall.add(valid)
                by_group_unsolvable[group].add(valid)
            else:
                solvable_overall.add(valid)
                by_group_solvable[group].add(valid)

    if args.out:
        write_row_csv(args.out, results)

    if args.stats_out:
        write_stats_csv(
            args.stats_out,
            args.group_by,
            overall,
            solvable_overall,
            unsolvable_overall,
            dict(by_group_all),
            dict(by_group_solvable),
            dict(by_group_unsolvable),
        )

    print()
    print("Overall")
    print("-------")
    print(f"All records:        {overall.correct}/{overall.n} = {overall.accuracy:.2%}")
    print(f"Solvable only:      {solvable_overall.correct}/{solvable_overall.n} = {solvable_overall.accuracy:.2%}")
    print(f"Unsolvable only:    {unsolvable_overall.correct}/{unsolvable_overall.n} = {unsolvable_overall.accuracy:.2%}")

    group_label = args.group_by
    print_table(
        title=f"Accuracy by {args.group_by} — all records",
        rows=sorted(by_group_all.items(), key=lambda kv: sort_key(kv[0])),
        group_label=group_label,
        min_count=args.min_count,
    )

    print_table(
        title=f"Accuracy by {args.group_by} — solvable only",
        rows=sorted(by_group_solvable.items(), key=lambda kv: sort_key(kv[0])),
        group_label=group_label,
        min_count=args.min_count,
    )

    print_table(
        title=f"Accuracy by {args.group_by} — unsolvable only",
        rows=sorted(by_group_unsolvable.items(), key=lambda kv: sort_key(kv[0])),
        group_label=group_label,
        min_count=args.min_count,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
