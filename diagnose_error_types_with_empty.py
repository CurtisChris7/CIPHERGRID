#!/usr/bin/env python3
"""
diagnose_error_types_with_empty_fixed.py

Inputs:
  1) JSONL file with keys: id, query, answer
  2) Solutions CSV with columns: id, response

Output:
  CSV with columns:
    id,response,correct,
    ignoring_obstacles,
    ignoring_enemies,
    moving_out_of_bounds,
    moving_through_traps,
    acting_without_items,
    using_item_actions_without_pickup,
    using_items_more_than_once,
    empty_response,
    illegal_string,
    giving_up,
    applying_ra_incorrectly,
    movement_misunderstanding

Notes:
- Uses the uploaded solver rules directly for faithful replay.
- Handles nondeterminism by propagating all possible states via solver.successors().
- "correct" is 1 if the response exactly matches the gold answer after normalization,
  or if it can be legally replayed to end on the goal tile.
- "applying_ra_incorrectly" is for the case where the full response is legal,
  contains ra, and still does not end on the goal tile when the gold answer also uses ra.
- "movement_misunderstanding" is preserved as a fallback for any incorrect response
  that is not captured by a more specific error category.
- If a row has multiple detectable issues, multiple columns can be 1.

The script also prints the list of IDs whose responses are empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import solver

# Increase the limit to 1MB or higher as needed
csv.field_size_limit(1024 * 1024) 

ERROR_COLUMNS = [
    "ignoring_obstacles",
    "ignoring_enemies",
    "moving_out_of_bounds",
    "moving_through_traps",
    "acting_without_items",
    "using_item_actions_without_pickup",
    "using_items_more_than_once",
    "empty_response",
    "illegal_string",
    "giving_up",
    "applying_ra_incorrectly",
    "movement_misunderstanding",
]


CIPHERGRID_ACTION_TERMS = {
    "ma",  # UP
    "ga",  # RIGHT
    "za",  # LEFT
    "va",  # DOWN
    "ya",  # PICKUP
    "ba",  # ATTACK
    "sa",  # PLACE
    "ra",  # SWIM
    "ta",  # RESIGN
}


@dataclass(frozen=True)
class MetaState:
    st: solver.State
    picked_sword: bool = False
    picked_plank: bool = False
    used_sword: bool = False
    used_plank: bool = False


@dataclass(frozen=True)
class ReplayResult:
    metas: Set[MetaState]
    legal_so_far: bool
    failed_action: Optional[str]
    stopped_on_resign: bool


def empty_flags() -> Dict[str, int]:
    return {k: 0 for k in ERROR_COLUMNS}


def _norm(s: str) -> str:
    return (s or "").strip().lstrip("\ufeff").strip().lower()


def normalize_action_string(action_str: str) -> str:
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


def contains_any_ciphergrid_action(action_str: str) -> bool:
    """Return True if the normalized response contains any legal CIPHERGRID action token."""
    return any(tok in CIPHERGRID_ACTION_TERMS for tok in parse_actions(action_str))


def load_solutions_csv(path: str) -> Dict[str, str]:
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
        resp_i = h.index("response")

        for row in reader:
            if not row or id_i >= len(row):
                continue
            rid = str(row[id_i]).strip()
            if not rid:
                continue
            resp = str(row[resp_i]).strip() if resp_i < len(row) else ""
            out[rid] = resp
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

    W = max(len(r) for r in rows)
    for r in rows:
        if len(r) < W:
            r.extend([solver.WALL] * (W - len(r)))

    return rows


def initial_meta_states(grid: List[List[str]]) -> Tuple[Set[MetaState], int, Tuple[int, int]]:
    swords0, planks0, monsters0, traps_mask, starts, goal = solver.build_masks(grid)
    metas: Set[MetaState] = set()
    for sr, sc in starts:
        metas.add(MetaState(solver.State(sr, sc, 0, swords0, planks0, monsters0, 0)))
    return metas, traps_mask, goal


def advance_meta_states(
    grid: List[List[str]],
    metas: Set[MetaState],
    action: str,
    traps_mask: int,
) -> Set[MetaState]:
    W = len(grid[0])
    out: Set[MetaState] = set()

    for meta in metas:
        st = meta.st
        cur_bit = solver.bit_at(st.r, st.c, W)
        for legal_action, st2 in solver.successors(grid, st, traps_mask):
            if legal_action != action:
                continue

            picked_sword = meta.picked_sword
            picked_plank = meta.picked_plank
            used_sword = meta.used_sword
            used_plank = meta.used_plank

            if action == solver.PICKUP:
                if (st.swords_mask & cur_bit) != 0 and st2.inv == 1:
                    picked_sword = True
                if (st.planks_mask & cur_bit) != 0 and st2.inv == 2:
                    picked_plank = True
            elif action == solver.ATTACK:
                used_sword = True
            elif action == solver.PLACE:
                used_plank = True

            out.add(
                MetaState(
                    st=st2,
                    picked_sword=picked_sword,
                    picked_plank=picked_plank,
                    used_sword=used_sword,
                    used_plank=used_plank,
                )
            )
    return out


def any_goal(metas: Iterable[MetaState], goal: Tuple[int, int]) -> bool:
    gr, gc = goal
    return any((m.st.r, m.st.c) == (gr, gc) for m in metas)


def classify_failed_action(
    grid: List[List[str]],
    metas: Set[MetaState],
    action: str,
    traps_mask: int,
) -> Dict[str, int]:
    flags = empty_flags()
    H, W = len(grid), len(grid[0])

    for meta in metas:
        st = meta.st
        cur_tile = grid[st.r][st.c]
        cur_bit = solver.bit_at(st.r, st.c, W)

        on_live_monster = (st.monsters_mask & cur_bit) != 0
        on_uncovered_trap = (
            cur_tile == solver.TRAP and (st.covered_traps_mask & cur_bit) == 0
        )

        if action != solver.ATTACK and solver.MONSTERS_BLOCK_MOVEMENT_UNTIL_ATTACK and on_live_monster:
            flags["ignoring_enemies"] = 1
        if action != solver.PLACE and solver.TRAPS_BLOCK_MOVEMENT_UNTIL_PLACED and on_uncovered_trap:
            flags["moving_through_traps"] = 1

        if action in solver.DIRS:
            dr, dc = solver.DIRS[action]
            if solver.on_reversal(grid, st.r, st.c):
                dr, dc = -dr, -dc
            nr, nc = st.r + dr, st.c + dc

            if not solver.in_bounds(nr, nc, H, W):
                flags["moving_out_of_bounds"] = 1
                continue

            target_tile = grid[nr][nc]
            target_bit = solver.bit_at(nr, nc, W)

            if target_tile == solver.WALL:
                flags["ignoring_obstacles"] = 1

            target_is_live_monster = (st.monsters_mask & target_bit) != 0
            target_is_uncovered_trap = (
                (traps_mask & target_bit) != 0 and (st.covered_traps_mask & target_bit) == 0
            )

            if target_is_live_monster and st.inv != 1:
                flags["ignoring_enemies"] = 1
                flags["acting_without_items"] = 1
                if meta.used_sword:
                    flags["using_items_more_than_once"] = 1

            if target_is_uncovered_trap and st.inv != 2:
                flags["moving_through_traps"] = 1
                flags["acting_without_items"] = 1
                if meta.used_plank:
                    flags["using_items_more_than_once"] = 1

        elif action == solver.ATTACK:
            if st.inv != 1:
                flags["acting_without_items"] = 1
                if not meta.picked_sword:
                    flags["using_item_actions_without_pickup"] = 1
                if meta.used_sword:
                    flags["using_items_more_than_once"] = 1

        elif action == solver.PLACE:
            if st.inv != 2:
                flags["acting_without_items"] = 1
                if not meta.picked_plank:
                    flags["using_item_actions_without_pickup"] = 1
                if meta.used_plank:
                    flags["using_items_more_than_once"] = 1

        elif action == solver.SWIM:
            pass
        elif action == solver.PICKUP:
            pass
        else:
            pass

    return flags


def replay_response(
    grid: List[List[str]],
    response_actions: List[str],
) -> Tuple[ReplayResult, int, Tuple[int, int]]:
    metas, traps_mask, goal = initial_meta_states(grid)
    legal_so_far = True
    failed_action: Optional[str] = None
    stopped_on_resign = False

    for action in response_actions:
        if action == solver.RESIGN:
            stopped_on_resign = True
            break
        nxt = advance_meta_states(grid, metas, action, traps_mask)
        if not nxt:
            legal_so_far = False
            failed_action = action
            break
        metas = nxt

    return ReplayResult(
        metas=metas,
        legal_so_far=legal_so_far,
        failed_action=failed_action,
        stopped_on_resign=stopped_on_resign,
    ), traps_mask, goal


def compute_correct(response: str, answer: str, replay: ReplayResult, goal: Tuple[int, int]) -> int:
    response_norm = normalize_action_string(response)
    answer_norm = normalize_action_string(answer)
    exact_match = bool(response_norm) and response_norm == answer_norm
    replay_valid = (
        replay.legal_so_far
        and not replay.stopped_on_resign
        and any_goal(replay.metas, goal)
    )
    return 1 if (exact_match or replay_valid) else 0


def diagnose_row(grid: List[List[str]], response: str, answer: str) -> Tuple[int, Dict[str, int]]:
    flags = empty_flags()

    response_actions = parse_actions(response)
    answer_actions = parse_actions(answer)
    answer_has_solution = len(answer_actions) > 0 and answer_actions != [solver.RESIGN]
    answer_contains_ra = solver.SWIM in answer_actions
    response_contains_ra = solver.SWIM in response_actions

    response_norm = normalize_action_string(response)

    if response_norm == "":
        flags["empty_response"] = 1

    if not contains_any_ciphergrid_action(response):
        flags["illegal_string"] = 1

    replay, traps_mask, goal = replay_response(grid, response_actions)

    if solver.RESIGN in response_actions and answer_has_solution:
        flags["giving_up"] = 1

    if not replay.legal_so_far and replay.failed_action is not None:
        metas, _, _ = initial_meta_states(grid)
        for action in response_actions:
            if action == solver.RESIGN or action == replay.failed_action:
                break
            metas = advance_meta_states(grid, metas, action, traps_mask)
        step_flags = classify_failed_action(grid, metas, replay.failed_action, traps_mask)
        for k, v in step_flags.items():
            if v:
                flags[k] = 1

    no_nonterminal_errors = not any(
        flags[k]
        for k in ERROR_COLUMNS
        if k not in {"applying_ra_incorrectly", "movement_misunderstanding", "empty_response"}
    )

    if (
        replay.legal_so_far
        and not replay.stopped_on_resign
        and response_actions
        and no_nonterminal_errors
        and not any_goal(replay.metas, goal)
    ):
        if answer_contains_ra and response_contains_ra:
            flags["applying_ra_incorrectly"] = 1
        elif not response_contains_ra:
            flags["movement_misunderstanding"] = 1

    correct = compute_correct(response, answer, replay, goal)

    if correct == 0 and not any(
        flags[k] for k in ERROR_COLUMNS if k != "movement_misunderstanding"
    ):
        flags["movement_misunderstanding"] = 1

    return correct, flags


def maybe_decode_query(query: str, use_decode: bool) -> str:
    if not use_decode:
        return query
    import decode_world
    return decode_world.decode_world(query)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="JSONL with keys: id, query, answer")
    ap.add_argument("--solutions", required=True, help='Solutions CSV with columns "id","response"')
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--decode", action="store_true", help="Decode query using decode_world.decode_world first")
    args = ap.parse_args()

    sol_map = load_solutions_csv(args.solutions)
    empty_ids: List[str] = []

    with open(args.jsonl, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(["id", "response", "correct", *ERROR_COLUMNS])

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            rid = ""
            response = ""
            correct = 0
            flags = empty_flags()

            try:
                obj = json.loads(line)
                rid = str(obj["id"]).strip()
                query = obj["query"]
                answer = str(obj.get("answer", "")).strip()
                response = sol_map.get(rid, "")

                if not isinstance(query, str):
                    raise TypeError("query must be a string")

                world_text = maybe_decode_query(query, args.decode)
                grid = parse_world_text(world_text)
                correct, flags = diagnose_row(grid, response, answer)

            except Exception as e:
                print(f"[line {line_no}] error for id={rid or '?'}: {e}", file=sys.stderr)

            if flags.get("empty_response", 0) == 1 and rid:
                empty_ids.append(rid)

            writer.writerow([rid, response, str(correct), *[str(flags[c]) for c in ERROR_COLUMNS]])

    print("empty_response_ids:")
    print(empty_ids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
