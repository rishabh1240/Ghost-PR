"""Ghost-PR command-line interface.

Usage
-----
    ghost haunt --demo

Copies the bundled demo fixture into a temporary workspace, runs
baseline tests, applies a single ``<=`` → ``<`` AST mutation, re-runs
the tests, and reports whether the mutant survived or was killed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from ghost_pr.engine import Outcome, mutate_lte_to_lt, run_pytest
from ghost_pr.haunt import haunt_staged_changes


# ── Paths ───────────────────────────────────────────────────────────

_DEMO_DIR = Path(__file__).resolve().parent / "demo_fixtures"


# ── Pretty helpers ──────────────────────────────────────────────────

def _banner(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def _outcome_label(outcome: Outcome) -> str:
    return {
        Outcome.PASSED: "✅  PASSED",
        Outcome.FAILED: "❌  FAILED",
        Outcome.COLLECTION_ERROR: "⚠️   COLLECTION ERROR",
        Outcome.INTERNAL_ERROR: "💥  INTERNAL ERROR",
        Outcome.NO_TESTS: "🔍  NO TESTS COLLECTED",
        Outcome.TIMEOUT: "⏱️   TIMEOUT",
        Outcome.EXECUTION_ERROR: "🛑  EXECUTION ERROR",
    }.get(outcome, outcome.name)


# ── Core demo workflow ──────────────────────────────────────────────

def _run_demo() -> int:
    """Execute the full demo mutation-testing workflow.

    Returns an exit code (0 = success, 1 = error).
    """
    _banner("👻  Ghost-PR  ·  haunt --demo")

    # 1. Copy demo fixtures into a temporary workspace.
    tmp = tempfile.mkdtemp(prefix="ghost_pr_demo_")
    tmp_path = Path(tmp)
    print(f"\n📁  Temporary workspace: {tmp_path}")
    try:
        for src_file in _DEMO_DIR.iterdir():
            if src_file.name == "__init__.py" or src_file.name.startswith("__pycache__"):
                continue
            shutil.copy2(src_file, tmp_path / src_file.name)

        source_file = tmp_path / "bank.py"
        original_source = source_file.read_text()

        # 2. Run baseline tests.
        _banner("Step 1 · Running baseline tests")
        baseline = run_pytest(str(tmp_path))
        print(f"  Result: {_outcome_label(baseline.outcome)}")
        if baseline.stdout.strip():
            print(f"  stdout: {baseline.stdout.strip()}")

        # 3. Stop with a clear error if baseline fails.
        if baseline.outcome != Outcome.PASSED:
            print("\n🚫  Baseline tests did not pass — aborting mutation testing.")
            print(f"    (outcome: {baseline.outcome.name})")
            return 1

        # 4. Mutate: <= → <
        _banner("Step 2 · Applying AST mutation  (<=  →  <)")
        result = mutate_lte_to_lt(original_source)
        if result is None:
            print("  No '<=' comparison found in demo fixture — nothing to mutate.")
            return 1

        print(f"  Original : {result.original_expr}")
        print(f"  Mutated  : {result.mutated_expr}")

        # Write the mutant source into the temp workspace.
        source_file.write_text(result.mutated_source)

        # 5. Run tests against the mutant.
        _banner("Step 3 · Running tests against the mutant")
        mutant = run_pytest(str(tmp_path))
        print(f"  Result: {_outcome_label(mutant.outcome)}")
        if mutant.stdout.strip():
            print(f"  stdout: {mutant.stdout.strip()}")

        # 6/7. Report survival / kill.
        _banner("Result")
        if mutant.outcome == Outcome.PASSED:
            mutant_status = "SURVIVED"
            print("  🧟  MUTANT SURVIVED — the existing tests did NOT catch the change.")
            print("       ➜  [Example hint, not AI-generated] Consider adding a boundary test, e.g.  withdraw(100, 100) == 0")
        elif mutant.outcome == Outcome.FAILED:
            mutant_status = "KILLED"
            print("  🎯  MUTANT KILLED — the tests caught the mutation.  Nice coverage!")
        else:
            mutant_status = "INCONCLUSIVE"
            print(f"  ⚠️   INCONCLUSIVE — tests did not run cleanly: {mutant.outcome.name}")
            print("       Cannot determine if the mutant was killed or survived.")

        print(f"\n  Original expression : {result.original_expr}")
        print(f"  Mutated expression  : {result.mutated_expr}")
        print(f"  Mutant status       : {mutant_status}")

    finally:
        # 8. Clean up — restore nothing in the original, delete temp dir.
        shutil.rmtree(tmp_path, ignore_errors=True)
        print(f"\n🧹  Cleaned up temporary workspace.\n")

    return 0


# ── Argument parsing ────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ghost",
        description="Ghost-PR: mutation testing to find gaps in your tests.",
    )
    sub = parser.add_subparsers(dest="command")

    haunt_parser = sub.add_parser("haunt", help="Run mutation testing")
    haunt_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with the bundled demo fixture (withdraw function)",
    )
    haunt_parser.add_argument(
        "--repo",
        default=None,
        help="Path to repository root (defaults to searching from current working directory)",
    )

    args = parser.parse_args(argv)

    if args.command == "haunt":
        if args.demo:
            sys.exit(_run_demo())
        else:
            sys.exit(haunt_staged_changes(repo_dir=args.repo))
    else:
        parser.print_help()
        sys.exit(1)
