"""Ghost-PR command-line interface.

Usage
-----
    ghost haunt --demo
    ghost haunt --demo --suggest
    ghost haunt --demo --suggest --model qwen2.5-coder:7b
    ghost haunt

Copies the bundled demo fixture into a temporary workspace, runs
baseline tests, applies a single ``<=`` → ``<`` AST mutation, re-runs
the tests, reports whether the mutant survived or was killed, and
optionally requests an unverified AI test suggestion from local Ollama.
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

def _run_demo(
    suggest: bool = False,
    model: str | None = None,
    yes: bool = False,
) -> int:
    """Execute the full demo mutation-testing workflow.

    Parameters
    ----------
    suggest : bool
        If True, request an unverified AI test suggestion from local Ollama
        when a mutant survives.
    model : str, optional
        Ollama model to use for test suggestions (defaults to 'qwen2.5-coder:7b').
    yes : bool, optional
        If True, automatically accept verification without interactive confirmation.

    Returns an exit code (0 = success, 1 = error).
    """
    from ghost_pr.verify import reset_auto_yes_notice
    reset_auto_yes_notice()

    title = "👻  Ghost-PR  ·  haunt --demo"
    if suggest:
        title += " --suggest"
    if yes:
        title += " --yes"
    _banner(title)

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
        test_file = tmp_path / "test_bank.py"
        original_source = source_file.read_text(encoding="utf-8")

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
        source_file.write_text(result.mutated_source, encoding="utf-8")

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
            if not suggest:
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

        # 8. Local AI test suggestion (Milestone 3A) & Verification (Milestone 3B)
        if suggest:
            if mutant_status == "SURVIVED":
                from ghost_pr.suggest import (
                    display_suggestion,
                    extract_function_definition,
                    request_test_suggestion,
                )
                from ghost_pr.verify import (
                    VerificationVerdict,
                    run_verification_flow,
                )

                original_fn = extract_function_definition(original_source, "withdraw")
                mutated_fn = extract_function_definition(result.mutated_source, "withdraw")
                existing_tests = test_file.read_text(encoding="utf-8")

                suggestion = request_test_suggestion(
                    original_function=original_fn,
                    mutated_function=mutated_fn,
                    existing_tests=existing_tests,
                    import_path="from bank import withdraw",
                    model=model,
                )
                display_suggestion(suggestion, auto_yes=yes)

                # 9. Verification of reviewed test (Milestone 3B)
                if suggestion.is_success and suggestion.test_code:
                    v_result = run_verification_flow(
                        test_code=suggestion.test_code,
                        original_source=original_source,
                        mutated_source=result.mutated_source,
                        auto_yes=yes,
                    )

                    # At most ONE feedback-based retry if REJECTED or NON-DISTINGUISHING
                    if (
                        v_result is not None
                        and v_result.verdict in (
                            VerificationVerdict.REJECTED,
                            VerificationVerdict.NON_DISTINGUISHING,
                        )
                    ):
                        from ghost_pr.suggest import (
                            extract_failure_diagnostics,
                            request_revised_test_suggestion,
                        )

                        print(
                            f"\n🔄  Initial suggested test was {v_result.verdict.name}. "
                            "Requesting 1 feedback-based revision from Ollama..."
                        )
                        diagnostics = extract_failure_diagnostics(
                            verdict=v_result.verdict.name,
                            orig_detail=v_result.orig_detail,
                            mut_detail=v_result.mut_detail,
                        )
                        revised = request_revised_test_suggestion(
                            original_function=original_fn,
                            mutated_function=mutated_fn,
                            previous_test=suggestion.test_code,
                            verdict=v_result.verdict.name,
                            diagnostics=diagnostics,
                            import_path="from bank import withdraw",
                            model=model,
                        )
                        display_suggestion(revised, is_retry=True, auto_yes=yes)

                        if revised.is_success and revised.test_code:
                            run_verification_flow(
                                test_code=revised.test_code,
                                original_source=original_source,
                                mutated_source=result.mutated_source,
                                auto_yes=yes,
                            )
            else:
                print("\n  ℹ️   No surviving mutants — test suggestions are only requested when a mutant survives.")

    finally:
        # Clean up — restore nothing in the original, delete temp dir.
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
        "--suggest",
        action="store_true",
        help="Request a local AI test suggestion from Ollama for surviving mutants",
    )
    haunt_parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to use for AI test suggestions (defaults to 'qwen2.5-coder:7b')",
    )
    haunt_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Automatically accept verification without interactive confirmation (requires --suggest)",
    )
    haunt_parser.add_argument(
        "--repo",
        default=None,
        help="Path to repository root (defaults to searching from current working directory)",
    )

    args = parser.parse_args(argv)

    if args.command == "haunt":
        if args.yes and not args.suggest:
            print("\n❌  Usage error: --yes requires --suggest.")
            print("    Run 'ghost haunt --suggest --yes' or 'ghost haunt --demo --suggest --yes'.\n")
            sys.exit(1)

        suggest_enabled = args.suggest or (args.model is not None)
        if args.demo:
            sys.exit(_run_demo(suggest=suggest_enabled, model=args.model, yes=args.yes))
        else:
            sys.exit(
                haunt_staged_changes(
                    repo_dir=args.repo,
                    suggest=suggest_enabled,
                    model=args.model,
                    yes=args.yes,
                )
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

