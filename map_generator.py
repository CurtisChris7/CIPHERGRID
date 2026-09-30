"""
Generate n strings of the form:

    wa:a-b-c-d-...

Internal terms are randomly drawn from a vocabulary, with:
- per-string min/max internal length
- GLOBAL constraint: 'na' and/or 'da' appear exactly once EACH across ALL n strings
  (i.e., at most one total occurrence across the entire dataset)
- optional forbidding of 'na'/'da' entirely
- reproducible with --seed

Examples:
  # exactly one 'na' and one 'da' somewhere across the whole batch:
  python map_generator.py --n 20 --min-len 3 --max-len 7 --global-once na da --seed 0

  # only 'na' appears exactly once globally; 'da' never appears:
  python map_generator.py --n 10 --min-len 2 --max-len 5 --global-once na --forbid da

  # neither appears at all:
  python map_generator.py --n 10 --min-len 2 --max-len 5 --forbid na da
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


DEFAULT_VOCAB = ["na", "xa", "ea", "pa", "oa", "ka", "ca", "ha", "la", "da"]


@dataclass(frozen=True)
class Config:
    n: int
    min_len: int
    max_len: int
    vocab: List[str]
    prefix: str = "wa:"
    sep: str = "-"
    seed: Optional[int] = None
    global_once: Set[str] = None  # tokens that must appear exactly once globally
    forbid: Set[str] = None       # tokens that must not appear at all


def _validate(cfg: Config) -> None:
    if cfg.n <= 0:
        raise ValueError("--n must be > 0")
    if cfg.min_len < 0 or cfg.max_len < 0:
        raise ValueError("--min-len/--max-len must be >= 0")
    if cfg.min_len > cfg.max_len:
        raise ValueError("--min-len cannot exceed --max-len")
    if not cfg.vocab:
        raise ValueError("vocab cannot be empty")

    global_once = cfg.global_once or set()
    forbid = cfg.forbid or set()

    overlap = global_once & forbid
    if overlap:
        raise ValueError(f"Token(s) cannot be both globally-once and forbidden: {sorted(overlap)}")

    vocab_set = set(cfg.vocab)
    missing = [t for t in global_once if t not in vocab_set]
    if missing:
        raise ValueError(f"Token(s) in --global-once not in vocab: {missing}")

    # Feasibility: we must have at least one slot somewhere to place each global_once token.
    # Total slots across all strings: sum(length_i). Minimal total slots: n * min_len.
    min_total_slots = cfg.n * cfg.min_len
    if min_total_slots < len(global_once):
        raise ValueError(
            f"Impossible: n*min_len = {cfg.n}*{cfg.min_len} = {min_total_slots} "
            f"slots, but need {len(global_once)} globally-once token placements: {sorted(global_once)}"
        )

    # Also if min_len==0, we can still be feasible as long as max_total_slots allows it,
    # but we don't know actual sum until we sample. We'll handle by constructing lengths.
    max_total_slots = cfg.n * cfg.max_len
    if max_total_slots < len(global_once):
        raise ValueError(
            f"Impossible: n*max_len = {cfg.n}*{cfg.max_len} = {max_total_slots} "
            f"slots, but need {len(global_once)} placements."
        )


def _choose_lengths(rng: random.Random, n: int, min_len: int, max_len: int, required_slots: int) -> List[int]:
    """
    Choose per-string internal lengths in [min_len, max_len] such that
    total slots >= required_slots. (We guarantee feasibility if possible.)
    """
    lengths = [rng.randint(min_len, max_len) for _ in range(n)]
    if sum(lengths) >= required_slots:
        return lengths

    # If random pick didn't give enough slots (can happen when min_len is low),
    # deterministically bump lengths up until enough.
    deficit = required_slots - sum(lengths)
    # Indices we can still increase
    inc_candidates = [i for i in range(n) if lengths[i] < max_len]
    rng.shuffle(inc_candidates)

    i = 0
    while deficit > 0 and inc_candidates:
        idx = inc_candidates[i % len(inc_candidates)]
        if lengths[idx] < max_len:
            lengths[idx] += 1
            deficit -= 1
        i += 1
        # refresh candidates occasionally
        if i > 10_000 and deficit > 0:
            inc_candidates = [j for j in range(n) if lengths[j] < max_len]
            rng.shuffle(inc_candidates)

    if deficit > 0:
        # Should not happen due to validation, but keep it safe.
        raise RuntimeError("Failed to allocate enough total length to place globally-once tokens.")
    return lengths


def generate_strings(cfg: Config) -> List[str]:
    _validate(cfg)
    rng = random.Random(cfg.seed)

    global_once = cfg.global_once or set()
    forbid = cfg.forbid or set()

    # Allowed vocab for random draws (excluding forbidden and excluding global_once for filler draws,
    # so we don't accidentally create extra occurrences).
    filler_vocab = [t for t in cfg.vocab if t not in forbid and t not in global_once]
    if not filler_vocab and cfg.max_len > 0:
        # Still feasible if all terms are global_once and lengths exactly match, but uncommon.
        # We'll allow empty filler_vocab as long as no extra slots beyond global_once placements exist.
        pass

    # Pick per-string lengths ensuring we have enough total slots to place each global_once token
    lengths = _choose_lengths(rng, cfg.n, cfg.min_len, cfg.max_len, required_slots=len(global_once))

    # Create empty slots for each string
    terms_per_string: List[List[Optional[str]]] = [[None] * L for L in lengths]

    # Build a list of all slot coordinates (string_idx, pos_idx), then pick distinct ones
    all_slots: List[Tuple[int, int]] = [(si, pi) for si, L in enumerate(lengths) for pi in range(L)]
    rng.shuffle(all_slots)

    # Assign each globally-once token to a unique slot
    # (shuffle tokens so distribution is random)
    tokens = list(global_once)
    rng.shuffle(tokens)

    for tok, (si, pi) in zip(tokens, all_slots[: len(tokens)]):
        terms_per_string[si][pi] = tok

    # Fill remaining None slots with filler vocab
    for si, slots in enumerate(terms_per_string):
        for pi, v in enumerate(slots):
            if v is None:
                if not filler_vocab:
                    raise ValueError(
                        "No filler vocabulary available to fill extra slots "
                        "(all vocab terms are forbidden or globally-once). "
                        "Reduce lengths or expand vocab."
                    )
                slots[pi] = rng.choice(filler_vocab)

    # Serialize
    out: List[str] = []
    for slots in terms_per_string:
        out.append(cfg.prefix + cfg.sep.join(slots))  # type: ignore[arg-type]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate wa:a-b-c strings with global-once token constraint.")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--min-len", type=int, required=True)
    p.add_argument("--max-len", type=int, required=True)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument(
        "--global-once",
        nargs="*",
        default=[],
        help="Tokens that must appear exactly once total across all strings (e.g., na da).",
    )
    p.add_argument(
        "--forbid",
        nargs="*",
        default=[],
        help="Tokens that must never appear (e.g., na da).",
    )

    p.add_argument("--prefix", type=str, default="wa:")
    p.add_argument("--sep", type=str, default="-")
    p.add_argument(
        "--vocab",
        type=str,
        default=",".join(DEFAULT_VOCAB),
        help=f"Comma-separated vocabulary (default: {','.join(DEFAULT_VOCAB)}).",
    )

    args = p.parse_args()

    vocab = [t.strip() for t in args.vocab.split(",") if t.strip()]
    cfg = Config(
        n=args.n,
        min_len=args.min_len,
        max_len=args.max_len,
        vocab=vocab,
        prefix=args.prefix,
        sep=args.sep,
        seed=args.seed,
        global_once=set(args.global_once),
        forbid=set(args.forbid),
    )

    strings = generate_strings(cfg)
    print("\n".join(strings))


if __name__ == "__main__":
    main()
