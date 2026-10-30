"""Staged changes mutation testing workflow for Ghost-PR (Milestone 2).

This module coordinates:
  1. Finding the Git repository root.
  2. Rejecting unstaged changes to tracked files.
  3. Reporting and excluding untracked files.
  4. Reading staged changes relative to HEAD (including newly staged files).
  5. Building a pristine temporary snapshot from Git's index.
  6. Applying ``<=`` → ``<`` mutations strictly to changed lines in Python source.
     Explicitly skipping unsupported multiline comparisons, symlinks, submodules,
     and test files.
  7. Running baseline tests on the snapshot; stopping if they do not pass.
  8. Running at most three mutants, each starting from a pristine snapshot.
  9. Reporting file, line, expression diff, and outcome.
  10. Preserving user working files and Git index.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ghost_pr.engine import (
    Outcome,
    apply_mutation_to_source,
    find_mutation_candidates,
    run_pytest,
)
from ghost_pr.git_utils import (
    build_index_snapshot,
    find_repo_root,
    get_staged_entries,
    get_unstaged_tracked_files,
    get_untracked_files,
)


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


@dataclass
class MutantRunResult:
    file_path: str
    lineno: int
    original_expr: str
    mutated_expr: str
    status: str
    detail: str = ""


def haunt_staged_changes(repo_dir: Path | str | None = None) -> int:
    """Run mutation testing on staged Python changes in a Git repository.

    Returns exit code 0 on successful execution or clean skip, 1 on errors or failures.
    """
    _banner("👻  Ghost-PR  ·  haunt (staged changes)")

    # 1. Find the current Git repository root
    repo_root = find_repo_root(repo_dir)
    if repo_root is None:
        print("\n❌  Not inside a Git repository.")
        print("    Ghost-PR must be run from within a Git repository.")
        return 1

    print(f"\n📂  Git repository root: {repo_root}")

    # 2. Reject unstaged changes to tracked files for this MVP
    unstaged_tracked = get_unstaged_tracked_files(repo_root)
    if unstaged_tracked:
        print("\n❌  Unstaged changes detected in tracked file(s):")
        for f in unstaged_tracked:
            print(f"     • {f}")
        print("\n    Ghost-PR requires all tracked changes to be staged or resolved.")
        print("    Please stage them with 'git add <file>' or resolve them manually before running 'ghost haunt'.")
        return 1

    # 3. Exclude untracked files and report that they were excluded
    untracked = get_untracked_files(repo_root)
    if untracked:
        print("\nℹ️   Excluded untracked file(s) from snapshot:")
        for f in untracked:
            print(f"     • {f}")

    # 4. Read staged changes relative to HEAD
    staged_entries = get_staged_entries(repo_root)
    if not staged_entries:
        print("\nℹ️   No staged changes found relative to HEAD.")
        print("    Stage changes with 'git add <file>' before running 'ghost haunt'.")
        return 0

    # Filter staged files and report policy skips explicitly
    python_entries = []
    for entry in staged_entries:
        if entry.is_symlink:
            print(f"  ⚠️   Skipping staged file '{entry.path}': symlinks are not supported.")
            continue
        if entry.is_submodule:
            print(f"  ⚠️   Skipping staged file '{entry.path}': submodules are not supported.")
            continue
        if entry.is_test:
            print(f"  ℹ️   Skipping staged file '{entry.path}': test files are not mutated.")
            continue
        if not entry.is_python:
            # Non-python files (e.g. docs, configs)
            continue
        python_entries.append(entry)

    if not python_entries:
        print("\nℹ️   No supported mutations found in staged changes.")
        return 0

    # 5. Build temporary project snapshot from Git's index
    tmp_dir = tempfile.mkdtemp(prefix="ghost_pr_snapshot_")
    pristine_snapshot = Path(tmp_dir)
    print(f"\n📁  Temporary workspace: {pristine_snapshot}")

    try:
        build_index_snapshot(repo_root, pristine_snapshot)

        # Locate candidate mutations on changed lines
        all_candidates = []
        for entry in python_entries:
            staged_file = pristine_snapshot / entry.path
            if not staged_file.is_file():
                continue
            source = staged_file.read_text(encoding="utf-8")
            candidates, skipped = find_mutation_candidates(source, entry.path, entry.changed_lines)

            for skip in skipped:
                print(
                    f"  ⚠️   Skipping multiline comparison at {skip.file_path}:{skip.lineno} "
                    f"({skip.reason})."
                )

            all_candidates.extend(candidates)

        if not all_candidates:
            print("\nℹ️   No supported mutations found in staged changes.")
            return 0

        # 6 & 7. Run baseline tests on the snapshot; stop unless they pass
        _banner("Step 1 · Running baseline tests on staged snapshot")
        baseline = run_pytest(str(pristine_snapshot))
        print(f"  Result: {_outcome_label(baseline.outcome)}")
        if baseline.stdout.strip():
            print(f"  stdout: {baseline.stdout.strip()}")

        if baseline.outcome != Outcome.PASSED:
            print("\n🚫  Baseline tests did not pass on the staged snapshot — aborting mutation testing.")
            print(f"    (outcome: {baseline.outcome.name})")
            return 1

        # 8 & 9. Run at most three mutants, each starting from a pristine snapshot
        candidates_to_run = all_candidates[:3]
        if len(all_candidates) > 3:
            print(
                f"\nℹ️   Found {len(all_candidates)} mutation candidate(s). "
                f"Running at most 3 mutants."
            )
        else:
            print(f"\nℹ️   Running {len(candidates_to_run)} mutant(s).")

        results: list[MutantRunResult] = []

        for idx, candidate in enumerate(candidates_to_run, start=1):
            _banner(
                f"Mutant {idx}/{len(candidates_to_run)} · "
                f"{candidate.file_path}:{candidate.lineno}"
            )
            print(f"  Original : {candidate.original_expr}")
            print(f"  Mutated  : {candidate.mutated_expr}")

            # Create an isolated pristine snapshot copy for this mutant
            mutant_tmp = tempfile.mkdtemp(prefix=f"ghost_pr_mutant_{idx}_")
            mutant_path = Path(mutant_tmp)

            try:
                shutil.copytree(
                    pristine_snapshot,
                    mutant_path,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
                )
                mutant_file = mutant_path / candidate.file_path
                original_content = mutant_file.read_text(encoding="utf-8")
                mutated_content = apply_mutation_to_source(original_content, candidate)
                mutant_file.write_text(mutated_content, encoding="utf-8")

                mutant_run = run_pytest(str(mutant_path))
                print(f"  Result: {_outcome_label(mutant_run.outcome)}")
                if mutant_run.stdout.strip():
                    print(f"  stdout: {mutant_run.stdout.strip()}")

                if mutant_run.outcome == Outcome.PASSED:
                    status = "SURVIVED"
                    detail = "Existing tests did not catch this mutation."
                    print("  🧟  MUTANT SURVIVED — tests still passed with mutated code.")
                elif mutant_run.outcome == Outcome.FAILED:
                    status = "KILLED"
                    detail = "Tests failed, catching the mutation."
                    print("  🎯  MUTANT KILLED — tests caught the mutation.")
                else:
                    status = "INCONCLUSIVE"
                    detail = f"Tests failed to run cleanly: {mutant_run.outcome.name}"
                    print(f"  ⚠️   INCONCLUSIVE — {detail}")

                results.append(
                    MutantRunResult(
                        file_path=candidate.file_path,
                        lineno=candidate.lineno,
                        original_expr=candidate.original_expr,
                        mutated_expr=candidate.mutated_expr,
                        status=status,
                        detail=detail,
                    )
                )
            finally:
                shutil.rmtree(mutant_path, ignore_errors=True)

        # 10. Report summary
        _banner("Mutation Testing Summary")
        for res in results:
            icon = "🧟" if res.status == "SURVIVED" else ("🎯" if res.status == "KILLED" else "⚠️")
            print(f"  {icon}  {res.status:12}  {res.file_path}:{res.lineno}")
            print(f"      Original : {res.original_expr}")
            print(f"      Mutated  : {res.mutated_expr}")
            if res.detail:
                print(f"      Detail   : {res.detail}")

    finally:
        # Preserve working files and Git index: cleanup temp directory
        shutil.rmtree(pristine_snapshot, ignore_errors=True)
        print("\n🧹  Cleaned up temporary workspace.\n")

    return 0
