"""
Unified workflow: generate multiple dungeon "worlds", encode each world into a single string,
solve each world, and write two output files:

1) encoded_worlds.txt  -> one encoded world string per line
2) solutions.txt       -> one solution (action string) per line

This script composes the behavior of:
- map_generator.py (world row generation with global-once constraints)
- encode_script.py (remove delimiters/newlines -> continuous string)
- solver.py (BFS solver)

Example:
  python generate_worlds.py --worlds 50 --rows 8 --min-len 8 --max-len 8 --encoded-out encoded_worlds.txt --solutions-out solutions.txt

  --worlds 20 --rows 10 --min-len 5 --max-len 10 --encoded-out encoded_worlds_test.txt --solutions-out solutions_test.txt --require-solvable
    --seed 0 

Notes:
- Each "world" is represented as `--rows` lines, each line formatted `wa:a-b-c-...`.
- Exactly one START token ('na') and one GOAL token ('da') are placed per world by default.
- The solver expects at least one 'na' and exactly one 'da' per world.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

# --- Import from your existing scripts (must be in same directory / PYTHONPATH) ---
from map_generator import Config as GenConfig, generate_strings 
from encode_script import undelim_all  
from solver import solve_with_trace  


def parse_world_rows(rows: List[str]) -> List[List[str]]:
    """
    Convert a list of 'wa:' rows into a rectangular grid of tokens.
    Mirrors solver.parse_world_file() behavior but without file I/O. fileciteturn0file0L67-L110
    """
    parts: List[str] = []
    for line in rows:
        line = line.strip()
        if not line:
            continue
        # Allow comma-separated row blobs if someone passes "wa:... , wa:..."
        for seg in line.split(","):
            seg = seg.strip()
            if seg:
                parts.append(seg)

    parsed: List[List[str]] = []
    for p in parts:
        if not p.startswith("wa:"):
            continue
        tiles = [t.strip() for t in p[3:].split("-") if t.strip()]
        parsed.append(tiles)

    if not parsed:
        raise ValueError("No rows found. Each row must start with 'wa:'.")

    # Rectangularize by padding with wall tokens ('ca') to the max width. fileciteturn0file0L99-L109
    WALL = "ca"
    W = max(len(r) for r in parsed)
    for r in parsed:
        if len(r) < W:
            r.extend([WALL] * (W - len(r)))

    return parsed


def world_seed(base_seed: Optional[int], world_idx: int) -> Optional[int]:
    """Derive a per-world seed so the whole run is reproducible but worlds differ."""
    if base_seed is None:
        return None
    # simple, deterministic mixing
    return base_seed + world_idx


def main() -> None:
    p = argparse.ArgumentParser(description="Generate+encode+solve many worlds; write encoded strings and solutions.")
    p.add_argument("--worlds", type=int, required=True, help="Number of worlds to generate.")
    p.add_argument("--rows", type=int, required=True, help="Rows per world (each row is one wa:... string).")

    p.add_argument("--min-len", type=int, required=True, help="Min internal length per row.")
    p.add_argument("--max-len", type=int, required=True, help="Max internal length per row.")

    p.add_argument("--seed", type=int, default=None, help="Base RNG seed (per-world seeds are derived).")

    p.add_argument("--encoded-out", type=str, required=True, help="Path to write encoded world strings (one per line).")
    p.add_argument("--solutions-out", type=str, required=True, help="Path to write solutions (one per line).")

    p.add_argument(
        "--vocab",
        type=str,
        default="na,xa,ea,pa,oa,ka,ca,ha,la,da", #na,pa,ca,da - Nav Only, #na,pa,ca,da,ha,la,ka - Nav Only Complex
        help="Comma-separated vocabulary (defaults to your tile vocab).",
    )
    p.add_argument(
        "--global-once",
        nargs="*",
        default=["na", "da"],
        help="Tokens that must appear exactly once per world (default: na da).",
    )
    p.add_argument(
        "--forbid",
        nargs="*",
        default=[],
        help="Tokens that must never appear in generated rows.",
    )
    p.add_argument(
        "--require-solvable",
        action="store_true",
        help="If set, keep generating (with incremented seed) until each world is solvable.",
    )
    p.add_argument(
        "--max-tries",
        type=int,
        default=200,
        help="Max attempts per world when --require-solvable is set.",
    )

    args = p.parse_args()

    if args.worlds <= 0:
        raise SystemExit("--worlds must be > 0")
    if args.rows <= 0:
        raise SystemExit("--rows must be > 0")

    vocab = [t.strip() for t in args.vocab.split(",") if t.strip()]
    global_once = set(args.global_once or [])
    forbid = set(args.forbid or [])

    encoded_lines: List[str] = []
    solution_lines: List[str] = []

    for w in range(args.worlds):
        print("Solved: ", w)
        tries = 0
        while True:
            tries += 1
            seed = world_seed(args.seed, w * 10_000 + tries) if args.require_solvable else world_seed(args.seed, w)

            cfg = GenConfig(
                n=args.rows,
                min_len=args.min_len,
                max_len=args.max_len,
                vocab=vocab,
                seed=seed,
                global_once=global_once,
                forbid=forbid,
            )
            rows = generate_strings(cfg)

            # Encode: remove delimiters and row boundaries into one continuous string
            encoded = undelim_all("\n".join(rows))
            grid = parse_world_rows(rows)
            action_str, _, _ = solve_with_trace(grid)

            if args.require_solvable and action_str == "ta":
                if tries >= args.max_tries:
                    raise RuntimeError(
                        f"Failed to generate a solvable world {w} after {args.max_tries} attempts."
                    )
                continue

            encoded_lines.append(encoded)
            solution_lines.append(action_str if action_str is not None else "")
            break

    # Write outputs
    with open(args.encoded_out, "w", encoding="utf-8") as f:
        f.write("\n".join(encoded_lines) + ("\n" if encoded_lines else ""))

    with open(args.solutions_out, "w", encoding="utf-8") as f:
        f.write("\n".join(solution_lines) + ("\n" if solution_lines else ""))


if __name__ == "__main__":
    main()
