"""AST-based mutation engine for Ghost-PR.

This module provides:
  - `mutate_lte_to_lt`: rewrites *exactly one* ``<=`` → ``<`` comparison
    in a Python source string, using the `ast` module.
  - `run_pytest`: runs pytest in a subprocess with a timeout and returns
    a structured `TestResult`.
  - `find_mutation_candidates`: locates supported single-line ``<=`` comparisons
    on specific changed lines, explicitly flagging multiline comparisons.
  - `apply_mutation_to_source`: applies a specific mutation candidate to source code.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional


# ── Test-result classification ──────────────────────────────────────

class Outcome(Enum):
    """Possible outcomes when running a test suite."""
    PASSED = auto()           # all tests passed (exit 0)
    FAILED = auto()           # at least one test failed (exit 1)
    COLLECTION_ERROR = auto() # pytest could not collect tests (exit 2)
    INTERNAL_ERROR = auto()   # pytest hit an internal error (exit 3)
    NO_TESTS = auto()         # no tests were collected (exit 5)
    TIMEOUT = auto()          # subprocess exceeded the time limit
    EXECUTION_ERROR = auto()  # any other non-zero exit code


@dataclass
class TestResult:
    """Structured result from a single pytest invocation."""
    __test__ = False
    outcome: Outcome
    exit_code: Optional[int]
    stdout: str
    stderr: str



# Mapping of well-known pytest exit codes → Outcome.
_PYTEST_EXIT_MAP: dict[int, Outcome] = {
    0: Outcome.PASSED,
    1: Outcome.FAILED,
    2: Outcome.COLLECTION_ERROR,
    3: Outcome.INTERNAL_ERROR,
    5: Outcome.NO_TESTS,
}


def run_pytest(
    directory: str,
    *,
    timeout: int = 30,
    env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> TestResult:
    """Run ``pytest`` inside *directory* and return a `TestResult`.

    Uses the **active Python interpreter** (``sys.executable``) so that
    the correct virtual-environment pytest is invoked.

    Ensures tests import code from *directory* (and *directory/src* if present)
    by prepending them to ``PYTHONPATH``.

    Parameters
    ----------
    directory:
        Working directory that contains the tests.
    timeout:
        Maximum seconds before the subprocess is killed.
    env:
        Optional environment dictionary override.
    extra_args:
        Optional additional CLI arguments to pass to pytest.
    """
    run_env = dict(os.environ if env is None else env)
    abs_dir = str(Path(directory).resolve())
    paths = [abs_dir]
    src_dir = Path(directory) / "src"
    if src_dir.is_dir():
        paths.append(str(src_dir.resolve()))
    curr_pythonpath = run_env.get("PYTHONPATH", "")
    if curr_pythonpath:
        paths.append(curr_pythonpath)
    run_env["PYTHONPATH"] = os.pathsep.join(paths)
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"

    cmd = [sys.executable, "-B", "-m", "pytest", "-x", "-q", "--tb=short"]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        outcome = _PYTEST_EXIT_MAP.get(proc.returncode, Outcome.EXECUTION_ERROR)
        return TestResult(
            outcome=outcome,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            outcome=Outcome.TIMEOUT,
            exit_code=None,
            stdout="",
            stderr=f"Subprocess timed out after {timeout}s",
        )


# ── AST mutation ────────────────────────────────────────────────────

@dataclass
class MutationCandidate:
    """A specific AST mutation opportunity."""
    file_path: str
    lineno: int
    col_offset: int
    original_expr: str
    mutated_expr: str
    op_index: int = 0


@dataclass
class SkippedMutation:
    """A mutation candidate skipped due to policy or unsupported syntax."""
    file_path: str
    lineno: int
    reason: str


def find_mutation_candidates(
    source: str,
    file_path: str,
    changed_lines: set[int],
) -> tuple[list[MutationCandidate], list[SkippedMutation]]:
    """Scan *source* for ``<=`` comparisons on lines present in *changed_lines*.

    Skips unsupported multiline comparisons explicitly.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    candidates: list[MutationCandidate] = []
    skipped: list[SkippedMutation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue

        has_lte = any(isinstance(op, ast.LtE) for op in node.ops)
        if not has_lte:
            continue

        is_multiline = node.end_lineno is not None and node.end_lineno > node.lineno
        if is_multiline:
            # Check if any changed lines overlap with this multiline compare
            end_line = node.end_lineno if node.end_lineno is not None else node.lineno
            node_lines = set(range(node.lineno, end_line + 1))
            if node_lines & changed_lines:
                skipped.append(
                    SkippedMutation(
                        file_path=file_path,
                        lineno=node.lineno,
                        reason="multiline comparison (unsupported)",
                    )
                )
            continue

        if node.lineno not in changed_lines:
            continue

        for idx, op in enumerate(node.ops):
            if isinstance(op, ast.LtE):
                orig_expr = ast.unparse(node)
                mutated_node = ast.Compare(
                    left=node.left,
                    ops=[ast.Lt() if i == idx else o for i, o in enumerate(node.ops)],
                    comparators=node.comparators,
                )
                mut_expr = ast.unparse(mutated_node)
                candidates.append(
                    MutationCandidate(
                        file_path=file_path,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        original_expr=orig_expr,
                        mutated_expr=mut_expr,
                        op_index=idx,
                    )
                )

    # Deterministic sorting: by line number, then column offset, then op index
    candidates.sort(key=lambda c: (c.lineno, c.col_offset, c.op_index))
    skipped.sort(key=lambda s: s.lineno)

    return candidates, skipped


def apply_mutation_to_source(source: str, candidate: MutationCandidate) -> str:
    """Apply a single `MutationCandidate` to *source* and return rewritten code."""
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and node.lineno == candidate.lineno
            and node.col_offset == candidate.col_offset
        ):
            if candidate.op_index < len(node.ops) and isinstance(
                node.ops[candidate.op_index], ast.LtE
            ):
                node.ops[candidate.op_index] = ast.Lt()
                break

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ── Milestone 1 Legacy / Demo Helpers ───────────────────────────────

class _LtEToLt(ast.NodeTransformer):
    """Replace the *first* ``<=`` (LtE) comparison with ``<`` (Lt).

    Used by the Milestone 1 `--demo` workflow.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mutated: bool = False
        self.original_expr: Optional[str] = None
        self.mutated_expr: Optional[str] = None

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:  # noqa: N802
        if self.mutated:
            return node

        for idx, op in enumerate(node.ops):
            if isinstance(op, ast.LtE):
                self.original_expr = ast.unparse(node)
                node.ops[idx] = ast.Lt()
                self.mutated_expr = ast.unparse(node)
                self.mutated = True
                return node

        return node


@dataclass
class MutationResult:
    """Information returned by `mutate_lte_to_lt`."""
    mutated_source: str
    original_expr: str
    mutated_expr: str


def mutate_lte_to_lt(source: str) -> Optional[MutationResult]:
    """Parse *source*, mutate the first ``<=`` to ``<``, and return the
    rewritten source code together with expression details.

    Used by `--demo`. Returns ``None`` if no ``<=`` comparison was found.
    """
    tree = ast.parse(source)
    transformer = _LtEToLt()
    new_tree = transformer.visit(tree)

    if not transformer.mutated:
        return None

    ast.fix_missing_locations(new_tree)
    return MutationResult(
        mutated_source=ast.unparse(new_tree),
        original_expr=transformer.original_expr or "",
        mutated_expr=transformer.mutated_expr or "",
    )
