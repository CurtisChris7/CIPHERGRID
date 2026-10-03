#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import pathlib
import sys
import heapq
import itertools

# =========================
# Tiles
# =========================
WALL = "ca"
OPEN = "pa"
GOAL = "da"
START = "na"

MONSTER = "ea"
REVERSAL = "xa"
SWORD = "oa"
PLANK = "la"
WATER = "ha"
TRAP = "ka"

# =========================
# Actions
# =========================
UP = "ma"
RIGHT = "ga"
LEFT = "za"
DOWN = "va"
PICKUP = "ya"
ATTACK = "ba"
PLACE = "sa"
SWIM = "ra"
RESIGN = "ta"

DIRS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# =========================
# Rules (yours)
# =========================
# Monsters: cannot enter alive monster tile unless holding sword.
# If you are standing on a live monster tile, you must attack before moving away.
ATTACK_REQUIRES_SWORD = True
SWORD_CONSUMED_ON_ATTACK = True
MONSTERS_BLOCK_MOVEMENT_UNTIL_ATTACK = True

# Traps: you may ONLY step onto trap if holding plank, and must place before moving away.
TRAPS_REQUIRE_PLANK_TO_ENTER = True
TRAPS_BLOCK_MOVEMENT_UNTIL_PLACED = True

# Water: ra moves to one chosen adjacent legal tile, but transcript omits direction.
# ra is ONLY usable when standing on water.
RA_ONLY_FROM_WATER = True


# =========================
# State
# =========================
@dataclass(frozen=True)
class State:
    r: int
    c: int
    inv: int  # 0 none, 1 sword, 2 plank
    swords_mask: int
    planks_mask: int
    monsters_mask: int
    covered_traps_mask: int


def inv_name(inv: int) -> str:
    return {0: "none", 1: "sword", 2: "plank"}[inv]


def idx(r: int, c: int, W: int) -> int:
    return r * W + c


def bit_at(r: int, c: int, W: int) -> int:
    return 1 << idx(r, c, W)


def in_bounds(r: int, c: int, H: int, W: int) -> bool:
    return 0 <= r < H and 0 <= c < W


# =========================
# Parsing
# =========================
def parse_world_file(path: pathlib.Path) -> List[List[str]]:
    text = path.read_text().strip()
    if not text:
        raise ValueError("Empty dungeon file.")

    # Supports comma-separated or newline-separated wa: rows
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
        raise ValueError("No rows found. Each row must start with 'wa:'.")

    # Rectangularize by padding with walls
    W = max(len(r) for r in rows)
    for r in rows:
        if len(r) < W:
            r.extend([WALL] * (W - len(r)))

    return rows


# =========================
# Masks / world entities
# =========================
def build_masks(grid: List[List[str]]):
    H, W = len(grid), len(grid[0])
    swords = planks = monsters = traps = 0
    starts: List[Tuple[int, int]] = []
    goals: List[Tuple[int, int]] = []

    for r in range(H):
        for c in range(W):
            t = grid[r][c]
            b = bit_at(r, c, W)
            if t == SWORD:
                swords |= b
            elif t == PLANK:
                planks |= b
            elif t == MONSTER:
                monsters |= b
            elif t == TRAP:
                traps |= b
            elif t == START:
                starts.append((r, c))
            elif t == GOAL:
                goals.append((r, c))

    if len(starts) != 1:
        raise ValueError(f"Expected exactly one start tile 'na', found {len(starts)}.")
    if len(goals) != 1:
        raise ValueError(f"Expected exactly one goal tile 'da', found {len(goals)}.")

    goal = goals[0]
    return swords, planks, monsters, traps, starts, goal


def on_reversal(grid: List[List[str]], r: int, c: int) -> bool:
    return grid[r][c] == REVERSAL


# =========================
# Movement legality
# =========================
def passable_to_enter(grid: List[List[str]], st: State, nr: int, nc: int, traps_mask: int) -> bool:
    H, W = len(grid), len(grid[0])
    if not in_bounds(nr, nc, H, W):
        return False

    t = grid[nr][nc]
    if t == WALL:
        return False

    b = bit_at(nr, nc, W)

    # Trap entering rules: uncovered trap requires holding plank
    if (traps_mask & b) != 0 and (st.covered_traps_mask & b) == 0:
        if TRAPS_REQUIRE_PLANK_TO_ENTER:
            return st.inv == 2
        return False

    # Monster entering rules: alive monster requires sword
    if (st.monsters_mask & b) != 0:
        return st.inv == 1

    return True


# =========================
# Successors
# =========================
def successors(grid: List[List[str]], st: State, traps_mask: int) -> Iterable[Tuple[str, State]]:
    H, W = len(grid), len(grid[0])
    cur_tile = grid[st.r][st.c]
    cur_bit = bit_at(st.r, st.c, W)

    # --- Commitment tile: live monster => MUST attack before anything else ---
    if MONSTERS_BLOCK_MOVEMENT_UNTIL_ATTACK and (st.monsters_mask & cur_bit):
        if (not ATTACK_REQUIRES_SWORD) or (st.inv == 1):
            new_inv = st.inv
            if st.inv == 1 and SWORD_CONSUMED_ON_ATTACK:
                new_inv = 0
            yield ATTACK, State(
                st.r, st.c, new_inv,
                st.swords_mask,
                st.planks_mask,
                st.monsters_mask & ~cur_bit,
                st.covered_traps_mask,
            )
        return

    # --- Commitment tile: uncovered trap => MUST place before moving away ---
    if TRAPS_BLOCK_MOVEMENT_UNTIL_PLACED and cur_tile == TRAP and (st.covered_traps_mask & cur_bit) == 0:
        if st.inv == 2:
            yield PLACE, State(
                st.r, st.c, 0,  # plank consumed
                st.swords_mask,
                st.planks_mask,
                st.monsters_mask,
                st.covered_traps_mask | cur_bit,
            )
        return

    # --- Optional pickup (only if empty-handed) ---
    if st.inv == 0:
        if (st.swords_mask & cur_bit) != 0:
            yield PICKUP, State(
                st.r, st.c, 1,
                st.swords_mask & ~cur_bit,
                st.planks_mask,
                st.monsters_mask,
                st.covered_traps_mask,
            )
        if (st.planks_mask & cur_bit) != 0:
            yield PICKUP, State(
                st.r, st.c, 2,
                st.swords_mask,
                st.planks_mask & ~cur_bit,
                st.monsters_mask,
                st.covered_traps_mask,
            )

    # --- Movement ---
    if cur_tile == WATER:
        if RA_ONLY_FROM_WATER:
            # ra: move to any adjacent legal tile; transcript doesn't specify direction
            for dr, dc in DIRS.values():
                nr, nc = st.r + dr, st.c + dc
                if not passable_to_enter(grid, st, nr, nc, traps_mask):
                    continue
                yield SWIM, State(
                    nr, nc, st.inv,
                    st.swords_mask,
                    st.planks_mask,
                    st.monsters_mask,
                    st.covered_traps_mask,
                )
        return

    # Non-water: directional moves; reversal applies on CURRENT tile
    rev = on_reversal(grid, st.r, st.c)
    for act, (dr, dc) in DIRS.items():
        if rev:
            dr, dc = -dr, -dc
        nr, nc = st.r + dr, st.c + dc
        if not passable_to_enter(grid, st, nr, nc, traps_mask):
            continue
        yield act, State(
            nr, nc, st.inv,
            st.swords_mask,
            st.planks_mask,
            st.monsters_mask,
            st.covered_traps_mask,
        )


# =========================
# BFS + reconstruction
# =========================
def reconstruct(parent: Dict[State, Tuple[Optional[State], str]], goal_state: State) -> Tuple[List[str], List[State]]:
    actions: List[str] = []
    states: List[State] = [goal_state]
    st = goal_state
    while True:
        prev, act = parent[st]
        if prev is None:
            break
        actions.append(act)
        st = prev
        states.append(st)
    actions.reverse()
    states.reverse()
    return actions, states

def static_dist_to_goal(grid: List[List[str]], goal: Tuple[int, int]) -> List[int]:
    """
    Reverse BFS from goal on the *static* grid (treat everything except WALL as passable).
    Returns a flat distance array of length H*W. Unreachable cells have -1.
    """
    H, W = len(grid), len(grid[0])
    dist = [-1] * (H * W)
    gr, gc = goal
    if grid[gr][gc] == WALL:
        return dist

    q = deque()
    dist[idx(gr, gc, W)] = 0
    q.append((gr, gc))

    while q:
        r, c = q.popleft()
        d = dist[idx(r, c, W)]
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, H, W):
                continue
            if grid[nr][nc] == WALL:
                continue
            j = idx(nr, nc, W)
            if dist[j] != -1:
                continue
            dist[j] = d + 1
            q.append((nr, nc))

    return dist

def solve_with_trace(grid: List[List[str]]) -> Tuple[str, List[str], List[State]]:
    swords0, planks0, monsters0, traps_mask, starts, goal = build_masks(grid)
    H, W = len(grid), len(grid[0])
    gr, gc = goal

    # Strong admissible heuristic: exact static shortest-path distance ignoring monsters/traps/reversal.
    dist_static = static_dist_to_goal(grid, goal)

    # If ALL starts are statically unreachable from the goal (walls-only), there is no solution.
    any_reachable = any(dist_static[idx(sr, sc, W)] != -1 for sr, sc in starts)
    if not any_reachable:
        return RESIGN, [RESIGN], []

    def h(st: State) -> int:
        d = dist_static[idx(st.r, st.c, W)]
        # If statically unreachable, treat as infinite; A* will avoid these.
        return 10**9 if d == -1 else d

    parent: Dict[State, Tuple[Optional[State], str]] = {}
    gscore: Dict[State, int] = {}

    heap: List[Tuple[int, int, int, State]] = []
    tie = itertools.count()

    for sr, sc in starts:
        if dist_static[idx(sr, sc, W)] == -1:
            continue
        s0 = State(sr, sc, 0, swords0, planks0, monsters0, 0)
        if s0 in gscore:
            continue
        parent[s0] = (None, "")
        gscore[s0] = 0
        heapq.heappush(heap, (h(s0), 0, next(tie), s0))

    while heap:
        f, g, _, st = heapq.heappop(heap)
        if g != gscore.get(st):
            continue

        if (st.r, st.c) == goal:
            actions, states = reconstruct(parent, st)
            action_str = "-".join(actions) if actions else ""
            return action_str, actions, states

        ng = g + 1

        for act, nxt in successors(grid, st, traps_mask):
            # Optional extra pruning: if next position is statically unreachable, skip
            if dist_static[idx(nxt.r, nxt.c, W)] == -1:
                continue

            old = gscore.get(nxt)
            if old is not None and ng >= old:
                continue

            gscore[nxt] = ng
            parent[nxt] = (st, act)
            heapq.heappush(heap, (ng + h(nxt), ng, next(tie), nxt))

    return RESIGN, [RESIGN], []


# =========================
# Validation + printing
# =========================
def validate_plan(grid: List[List[str]], actions: List[str], states: List[State], traps_mask: int) -> None:
    """
    Validates that actions[i] is a legal transition from states[i] -> states[i+1].
    Raises AssertionError with a helpful message on first violation.
    """
    if not states:
        return
    H, W = len(grid), len(grid[0])

    def tile(r: int, c: int) -> str:
        return grid[r][c]

    for i, act in enumerate(actions):
        s = states[i]
        t = states[i + 1]
        s_tile = tile(s.r, s.c)

        # Basic adjacency / staying
        dr = t.r - s.r
        dc = t.c - s.c
        manhattan = abs(dr) + abs(dc)

        # Enforce commitment rules on the SOURCE state (what was allowed to do from there)
        s_bit = bit_at(s.r, s.c, W)
        if MONSTERS_BLOCK_MOVEMENT_UNTIL_ATTACK and (s.monsters_mask & s_bit):
            assert act == ATTACK, f"Step {i}: on live monster at {(s.r,s.c)} must 'ba', got '{act}'."
        if TRAPS_BLOCK_MOVEMENT_UNTIL_PLACED and s_tile == TRAP and (s.covered_traps_mask & s_bit) == 0:
            assert act == PLACE, f"Step {i}: on uncovered trap at {(s.r,s.c)} must 'sa', got '{act}'."

        if act == SWIM:
            assert s_tile == WATER, f"Step {i}: 'ra' only legal from water. At {(s.r,s.c)} tile={s_tile}."
            assert manhattan == 1, f"Step {i}: 'ra' must move to adjacent tile. From {(s.r,s.c)} to {(t.r,t.c)}."
            assert passable_to_enter(grid, s, t.r, t.c, traps_mask), f"Step {i}: 'ra' moved into illegal tile {(t.r,t.c)}."
        elif act in DIRS:
            assert s_tile != WATER, f"Step {i}: directional move '{act}' not legal from water at {(s.r,s.c)}."
            # Apply reversal on source tile
            ract = act
            if on_reversal(grid, s.r, s.c):
                # reverse direction
                if act == UP: ract = DOWN
                elif act == DOWN: ract = UP
                elif act == LEFT: ract = RIGHT
                elif act == RIGHT: ract = LEFT
            exp_dr, exp_dc = DIRS[ract]
            assert (dr, dc) == (exp_dr, exp_dc), (
                f"Step {i}: '{act}' from {(s.r,s.c)} expected delta {(exp_dr,exp_dc)} "
                f"(reversal={'yes' if on_reversal(grid,s.r,s.c) else 'no'}), got {(dr,dc)}."
            )
            assert passable_to_enter(grid, s, t.r, t.c, traps_mask), f"Step {i}: move entered illegal tile {(t.r,t.c)}."
        elif act == PICKUP:
            assert (s.r, s.c) == (t.r, t.c), f"Step {i}: 'ya' should not move, but moved to {(t.r,t.c)}."
            assert s.inv == 0, f"Step {i}: 'ya' requires empty hands."
            s_bit = bit_at(s.r, s.c, W)
            assert (s.swords_mask & s_bit) or (s.planks_mask & s_bit), f"Step {i}: 'ya' but no item at {(s.r,s.c)}."
            assert t.inv in (1, 2), f"Step {i}: 'ya' should result in holding sword/plank."
        elif act == PLACE:
            assert (s.r, s.c) == (t.r, t.c), f"Step {i}: 'sa' should not move."
            assert grid[s.r][s.c] == TRAP, f"Step {i}: 'sa' only on trap tile, but tile={grid[s.r][s.c]}."
            assert s.inv == 2, f"Step {i}: 'sa' requires holding plank."
        elif act == ATTACK:
            assert (s.r, s.c) == (t.r, t.c), f"Step {i}: 'ba' should not move."
            s_bit = bit_at(s.r, s.c, W)
            assert (s.monsters_mask & s_bit) != 0, f"Step {i}: 'ba' but no live monster at {(s.r,s.c)}."
            if ATTACK_REQUIRES_SWORD:
                assert s.inv == 1, f"Step {i}: 'ba' requires sword."
        else:
            raise AssertionError(f"Step {i}: unknown action '{act}'.")


def print_solution_with_actions(grid: List[List[str]], actions: List[str], states: List[State]) -> None:
    """
    Prints:
      0: (r,c) start tile=<tile> inv=<inv>
      i: (r,c) <action> tile=<tile> inv=<inv>
    Where action is the action taken to arrive at step i (so step 0 is 'start').
    """
    if not states:
        return

    def tile(r: int, c: int) -> str:
        return grid[r][c]

    s0 = states[0]
    print(f"0: ({s0.r},{s0.c}) start tile={tile(s0.r,s0.c)} inv={inv_name(s0.inv)}")

    for i, act in enumerate(actions, start=1):
        st = states[i]
        print(f"{i}: ({st.r},{st.c}) {act} tile={tile(st.r,st.c)} inv={inv_name(st.inv)}")


# =========================
# Main
# =========================
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 solver.py <dungeon_file>")
        sys.exit(1)

    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        raise FileNotFoundError(path)

    grid = parse_world_file(path)
    action_str, actions, states = solve_with_trace(grid)

    # First line: action sequence (or ta)
    print(action_str if action_str else "")

    if action_str == RESIGN or not states:
        return

    # Validate and print annotated trace
    _, _, _, traps_mask, _, _ = build_masks(grid)
    validate_plan(grid, actions, states, traps_mask)
    print_solution_with_actions(grid, actions, states)


if __name__ == "__main__":
    main()
