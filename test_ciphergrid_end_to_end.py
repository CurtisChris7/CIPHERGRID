#!/usr/bin/env python3
"""
End-to-end regression tests for CIPHERGRID generation, encoding/decoding,
rectangularization, solving, solver trace validation, and output validation.

Place this file in the repository root beside:
    map_generator.py
    generate_worlds.py
    encode_script.py
    decode_world.py
    solver.py
    validate_solutions.py

Run:
    python3 test_ciphergrid_end_to_end.py -v

Optional heavier randomized pass:
    CIPHERGRID_STRESS_WORLDS=500 python3 test_ciphergrid_end_to_end.py -v

This suite uses only Python's standard library.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the exact local project modules under test.
import decode_world
import encode_script
import generate_worlds
import map_generator
import solver
import validate_solutions


REQUIRED_FILES = (
    "map_generator.py",
    "generate_worlds.py",
    "encode_script.py",
    "decode_world.py",
    "solver.py",
    "validate_solutions.py",
)


def rows_to_tokens(rows: list[str]) -> list[str]:
    """Flatten serialized wa: rows into tile tokens without rectangular padding."""
    out: list[str] = []
    for row in rows:
        if not row.startswith("wa:"):
            raise AssertionError(f"Bad row prefix: {row!r}")
        out.extend(tok for tok in row[3:].split("-") if tok)
    return out


def grid_from_rows(rows: list[str]) -> list[list[str]]:
    return generate_worlds.parse_world_rows(rows)


def validate_solver_trace(grid: list[list[str]]) -> tuple[str, list[str], list[solver.State]]:
    """Solve a grid and independently run solver.validate_plan on non-ta traces."""
    action_str, actions, states = solver.solve_with_trace(grid)
    if action_str != solver.RESIGN:
        _, _, _, traps_mask, _, _ = solver.build_masks(grid)
        solver.validate_plan(grid, actions, states, traps_mask)
        if not states:
            raise AssertionError("Non-ta solution returned no states")
        _, _, _, _, _, goal = solver.build_masks(grid)
        if (states[-1].r, states[-1].c) != goal:
            raise AssertionError("Solver trace does not end at the goal")
    return action_str, actions, states


def run_python(script: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(ROOT / script), *map(str, args)]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


class Test00RepositoryLayout(unittest.TestCase):
    def test_required_files_are_present(self) -> None:
        missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
        self.assertEqual(missing, [], f"Missing required project files: {missing}")


class Test10WorldGeneration(unittest.TestCase):
    def test_generator_enforces_unique_start_and_goal(self) -> None:
        for seed in range(50):
            with self.subTest(seed=seed):
                cfg = map_generator.Config(
                    n=7,
                    min_len=4,
                    max_len=9,
                    vocab=list(map_generator.DEFAULT_VOCAB),
                    seed=seed,
                    global_once={"na", "da"},
                    forbid=set(),
                )
                rows = map_generator.generate_strings(cfg)
                tokens = rows_to_tokens(rows)
                self.assertEqual(len(rows), 7)
                self.assertEqual(tokens.count("na"), 1)
                self.assertEqual(tokens.count("da"), 1)
                for row in rows:
                    L = len([t for t in row[3:].split("-") if t])
                    self.assertGreaterEqual(L, 4)
                    self.assertLessEqual(L, 9)

    def test_forbidden_tokens_are_never_generated(self) -> None:
        cfg = map_generator.Config(
            n=20,
            min_len=5,
            max_len=5,
            vocab=list(map_generator.DEFAULT_VOCAB),
            seed=123,
            global_once={"na", "da"},
            forbid={"ea", "ha", "ka"},
        )
        tokens = rows_to_tokens(map_generator.generate_strings(cfg))
        for tok in ("ea", "ha", "ka"):
            self.assertNotIn(tok, tokens)

    def test_same_seed_is_reproducible_in_process(self) -> None:
        cfg = map_generator.Config(
            n=10,
            min_len=5,
            max_len=8,
            vocab=list(map_generator.DEFAULT_VOCAB),
            seed=999,
            global_once={"na", "da"},
            forbid=set(),
        )
        self.assertEqual(
            map_generator.generate_strings(cfg),
            map_generator.generate_strings(cfg),
        )

    def test_same_seed_is_reproducible_across_python_hash_seeds(self) -> None:
        # Regression test for converting global_once set -> list without sorting first.
        snippet = r'''
from map_generator import Config, DEFAULT_VOCAB, generate_strings
cfg = Config(n=8, min_len=5, max_len=8, vocab=list(DEFAULT_VOCAB), seed=314159,
             global_once={"na", "da"}, forbid=set())
print("\n".join(generate_strings(cfg)))
'''
        outputs: list[str] = []
        for hash_seed in ("1", "2", "3", "12345", "random"):
            cp = subprocess.run(
                [sys.executable, "-c", snippet],
                cwd=ROOT,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)
            outputs.append(cp.stdout)
        self.assertTrue(all(x == outputs[0] for x in outputs[1:]))

    def test_minimum_capacity_validation_matches_documented_behavior(self) -> None:
        # Current documented behavior: configurations are rejected unless n*min_len
        # already provides enough slots for all global-once tokens.
        bad = map_generator.Config(
            n=2,
            min_len=0,
            max_len=2,
            vocab=list(map_generator.DEFAULT_VOCAB),
            seed=0,
            global_once={"na", "da"},
            forbid=set(),
        )
        with self.assertRaises(ValueError):
            map_generator.generate_strings(bad)

        good = map_generator.Config(
            n=2,
            min_len=1,
            max_len=2,
            vocab=list(map_generator.DEFAULT_VOCAB),
            seed=0,
            global_once={"na", "da"},
            forbid=set(),
        )
        rows = map_generator.generate_strings(good)
        toks = rows_to_tokens(rows)
        self.assertEqual(toks.count("na"), 1)
        self.assertEqual(toks.count("da"), 1)


class Test20EncodingAndParsing(unittest.TestCase):
    def test_encode_decode_round_trip(self) -> None:
        rows = [
            "wa:na-pa-ca",
            "wa:ha-xa",
            "wa:oa-ea-la-ka-da",
        ]
        structured = "\n".join(rows)
        encoded = encode_script.undelim_all(structured)
        decoded = decode_world.decode_world(encoded)
        self.assertEqual(decoded, structured)

    def test_all_rectangularizers_agree_on_ragged_world(self) -> None:
        rows = [
            "wa:na-pa-da",
            "wa:pa",
            "wa:pa-pa",
        ]
        expected = [
            ["na", "pa", "da"],
            ["pa", "ca", "ca"],
            ["pa", "pa", "ca"],
        ]

        generation_grid = generate_worlds.parse_world_rows(rows)
        validator_grid = validate_solutions.parse_world_text("\n".join(rows))

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "world.txt"
            p.write_text("\n".join(rows), encoding="utf-8")
            standalone_solver_grid = solver.parse_world_file(p)

        self.assertEqual(generation_grid, expected)
        self.assertEqual(validator_grid, expected)
        self.assertEqual(standalone_solver_grid, expected)

    def test_encoded_world_decodes_to_same_rectangular_grid(self) -> None:
        rows = ["wa:na-pa", "wa:pa-pa-da"]
        encoded = encode_script.undelim_all("\n".join(rows))
        decoded = decode_world.decode_world(encoded)
        from_generation = generate_worlds.parse_world_rows(rows)
        from_evaluation = validate_solutions.parse_world_text(decoded)
        self.assertEqual(from_generation, from_evaluation)


class Test30SolverStructuralValidation(unittest.TestCase):
    def test_exactly_one_start_is_required(self) -> None:
        cases = [
            [["pa", "da"]],
            [["na", "na", "da"]],
        ]
        for grid in cases:
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    solver.build_masks(grid)

    def test_exactly_one_goal_is_required(self) -> None:
        cases = [
            [["na", "pa"]],
            [["na", "da", "da"]],
        ]
        for grid in cases:
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    solver.build_masks(grid)

    def test_valid_single_start_single_goal(self) -> None:
        masks = solver.build_masks([["na", "pa", "da"]])
        starts, goal = masks[-2], masks[-1]
        self.assertEqual(starts, [(0, 0)])
        self.assertEqual(goal, (0, 2))


class Test40HandcraftedMechanics(unittest.TestCase):
    def assertSolution(self, grid: list[list[str]], expected: str) -> None:  # noqa: N802
        got, _, _ = validate_solver_trace(grid)
        self.assertEqual(got, expected)
        if got != "ta":
            self.assertTrue(validate_solutions.validate_ends_on_goal(grid, got))

    def test_basic_navigation(self) -> None:
        self.assertSolution([["na", "pa", "da"]], "ga-ga")

    def test_unsolvable_wall(self) -> None:
        self.assertSolution([["na", "ca", "da"]], "ta")

    def test_monster_requires_sword_and_attack(self) -> None:
        self.assertSolution([["na", "oa", "ea", "da"]], "ga-ya-ga-ba-ga")

    def test_trap_requires_plank_and_place(self) -> None:
        self.assertSolution([["na", "la", "ka", "da"]], "ga-ya-ga-sa-ga")

    def test_water_can_exit_to_adjacent_land(self) -> None:
        # Important regression: ra from water is allowed to move to an adjacent
        # legal LAND cell; it is not restricted to water-to-water transitions.
        self.assertSolution([["na", "ha", "pa", "da"]], "ga-ra-ga")

    def test_reversal_applies_at_source_tile(self) -> None:
        self.assertSolution([["na", "xa", "da"]], "ga-za")

    def test_alternate_shortest_path_is_accepted_by_replay(self) -> None:
        grid = [
            ["na", "pa"],
            ["pa", "da"],
        ]
        reference, _, _ = validate_solver_trace(grid)
        self.assertIn(reference, {"ga-va", "va-ga"})
        alternate = "va-ga" if reference == "ga-va" else "ga-va"
        self.assertTrue(validate_solutions.validate_ends_on_goal(grid, reference))
        self.assertTrue(validate_solutions.validate_ends_on_goal(grid, alternate))

    def test_illegal_or_incomplete_sequences_are_rejected(self) -> None:
        grid = [["na", "pa", "da"]]
        self.assertFalse(validate_solutions.validate_ends_on_goal(grid, "ga"))
        self.assertFalse(validate_solutions.validate_ends_on_goal(grid, "ma"))
        self.assertFalse(validate_solutions.validate_ends_on_goal(grid, "ta"))


class Test50RandomizedSolverRegression(unittest.TestCase):
    def test_random_generated_worlds_solve_and_replay_consistently(self) -> None:
        n_worlds = int(os.environ.get("CIPHERGRID_STRESS_WORLDS", "100"))
        for seed in range(n_worlds):
            with self.subTest(seed=seed):
                cfg = map_generator.Config(
                    n=5,
                    min_len=4,
                    max_len=6,
                    vocab=list(map_generator.DEFAULT_VOCAB),
                    seed=seed,
                    global_once={"na", "da"},
                    forbid=set(),
                )
                rows = map_generator.generate_strings(cfg)
                raw_tokens = rows_to_tokens(rows)
                self.assertEqual(raw_tokens.count("na"), 1)
                self.assertEqual(raw_tokens.count("da"), 1)

                # Generation parser -> solver.
                grid = generate_worlds.parse_world_rows(rows)
                action_str, _, _ = validate_solver_trace(grid)

                # Encoder -> decoder -> evaluation parser must produce the same grid.
                encoded = encode_script.undelim_all("\n".join(rows))
                decoded = decode_world.decode_world(encoded)
                replay_grid = validate_solutions.parse_world_text(decoded)
                self.assertEqual(grid, replay_grid)

                if action_str == "ta":
                    # ta is a classification, not a replay transition. Re-solving the
                    # identical reconstructed grid must also classify it unsolvable.
                    replay_solution, _, _ = solver.solve_with_trace(replay_grid)
                    self.assertEqual(replay_solution, "ta")
                else:
                    self.assertTrue(
                        validate_solutions.validate_ends_on_goal(replay_grid, action_str),
                        f"Reference solution failed replay for seed {seed}: {action_str}",
                    )


class Test60CommandLinePipeline(unittest.TestCase):
    def test_generate_worlds_cli_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            worlds = td / "encoded.txt"
            solutions = td / "solutions.txt"
            cp = run_python(
                "generate_worlds.py",
                "--worlds", "25",
                "--rows", "5",
                "--min-len", "4",
                "--max-len", "6",
                "--seed", "42",
                "--encoded-out", str(worlds),
                "--solutions-out", str(solutions),
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)

            encoded_lines = worlds.read_text(encoding="utf-8").splitlines()
            solution_lines = solutions.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(encoded_lines), 25)
            self.assertEqual(len(solution_lines), 25)

            for i, (encoded, reference) in enumerate(zip(encoded_lines, solution_lines)):
                with self.subTest(i=i):
                    decoded = decode_world.decode_world(encoded)
                    grid = validate_solutions.parse_world_text(decoded)
                    # Structural validation runs here as part of build_masks/solve.
                    _, _, _, _, starts, goal = solver.build_masks(grid)
                    self.assertEqual(len(starts), 1)
                    self.assertIsInstance(goal, tuple)

                    re_solved, _, _ = solver.solve_with_trace(grid)
                    self.assertEqual(reference, re_solved)
                    if reference != "ta":
                        self.assertTrue(validate_solutions.validate_ends_on_goal(grid, reference))

    def test_require_solvable_cli_produces_only_solvable_records(self) -> None:
        # Use an open-only filler vocabulary so this is fast and deterministic while
        # still testing the --require-solvable output contract.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            worlds = td / "encoded.txt"
            solutions = td / "solutions.txt"
            cp = run_python(
                "generate_worlds.py",
                "--worlds", "10",
                "--rows", "5",
                "--min-len", "5",
                "--max-len", "5",
                "--vocab", "na,pa,da",
                "--seed", "7",
                "--require-solvable",
                "--max-tries", "20",
                "--encoded-out", str(worlds),
                "--solutions-out", str(solutions),
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)
            refs = solutions.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(refs), 10)
            self.assertTrue(all(x and x != "ta" for x in refs))

    def test_require_solvable_raises_on_retry_exhaustion(self) -> None:
        # Find a deterministic 1x3 seed with [na/da] separated by the single wall.
        # Under --require-solvable, attempt 1 uses base_seed + 1.
        attempt_seed = None
        for seed in range(1, 500):
            cfg = map_generator.Config(
                n=1,
                min_len=3,
                max_len=3,
                vocab=["na", "ca", "da"],
                seed=seed,
                global_once={"na", "da"},
                forbid=set(),
            )
            rows = map_generator.generate_strings(cfg)
            grid = generate_worlds.parse_world_rows(rows)
            result, _, _ = solver.solve_with_trace(grid)
            if result == "ta":
                attempt_seed = seed
                break
        self.assertIsNotNone(attempt_seed, "Could not find deterministic unsolvable seed")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cp = run_python(
                "generate_worlds.py",
                "--worlds", "1",
                "--rows", "1",
                "--min-len", "3",
                "--max-len", "3",
                "--vocab", "na,ca,da",
                "--seed", str(attempt_seed - 1),
                "--require-solvable",
                "--max-tries", "1",
                "--encoded-out", str(td / "encoded.txt"),
                "--solutions-out", str(td / "solutions.txt"),
            )
            self.assertNotEqual(cp.returncode, 0)
            self.assertIn("Failed to generate a solvable world", cp.stderr + cp.stdout)


class Test70ValidationCLI(unittest.TestCase):
    def test_validator_cli_accepts_reference_and_alternate_paths_and_rejects_bad_path(self) -> None:
        # Three records exercise replay-valid alternate path, ta/reference matching,
        # and an incomplete path that should fail.
        records: list[dict[str, str]] = []

        rows1 = ["wa:na-pa", "wa:pa-da"]
        grid1 = generate_worlds.parse_world_rows(rows1)
        ref1, _, _ = solver.solve_with_trace(grid1)
        alt1 = "va-ga" if ref1 == "ga-va" else "ga-va"
        self.assertTrue(validate_solutions.validate_ends_on_goal(grid1, alt1))
        records.append({
            "id": "alternate",
            "query": encode_script.undelim_all("\n".join(rows1)),
            "answer": ref1,
        })

        rows2 = ["wa:na-ca-da"]
        grid2 = generate_worlds.parse_world_rows(rows2)
        ref2, _, _ = solver.solve_with_trace(grid2)
        self.assertEqual(ref2, "ta")
        records.append({
            "id": "unsolvable",
            "query": encode_script.undelim_all("\n".join(rows2)),
            "answer": ref2,
        })

        rows3 = ["wa:na-pa-da"]
        grid3 = generate_worlds.parse_world_rows(rows3)
        ref3, _, _ = solver.solve_with_trace(grid3)
        self.assertEqual(ref3, "ga-ga")
        records.append({
            "id": "bad",
            "query": encode_script.undelim_all("\n".join(rows3)),
            "answer": ref3,
        })

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            jsonl = td / "records.jsonl"
            responses = td / "responses.csv"
            output = td / "validated.csv"

            with jsonl.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            with responses.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "response"])
                w.writerow(["alternate", alt1])       # replay-valid, not reference string
                w.writerow(["unsolvable", "ta"])     # exact reference for unsolvable
                w.writerow(["bad", "ga"])            # legal prefix, does not reach goal

            cp = run_python(
                "validate_solutions.py",
                "--decode",
                "--jsonl", str(jsonl),
                "--solutions", str(responses),
                "--out", str(output),
            )
            self.assertEqual(cp.returncode, 0, cp.stderr)

            with output.open("r", encoding="utf-8", newline="") as f:
                got = {row["id"]: row["valid"] for row in csv.DictReader(f)}
            self.assertEqual(got, {
                "alternate": "1",
                "unsolvable": "1",
                "bad": "0",
            })


if __name__ == "__main__":
    unittest.main(verbosity=2)
