"""AST-based mutation engine for Ghost-PR.

This module provides:
  - `mutate_lte_to_lt`: rewrites *exactly one* ``<=`` → ``<`` comparison
    in a Python source string, using the `ast` module.
  - `run_pytest`: runs pytest in a subprocess with a timeout and returns
    a structured `TestResult`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
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


def run_pytest(directory: str, *, timeout: int = 30) -> TestResult:
    """Run ``pytest`` inside *directory* and return a `TestResult`.

    Uses the **active Python interpreter** (``sys.executable``) so that
    the correct virtual-environment pytest is invoked.

    Parameters
    ----------
    directory:
        Working directory that contains the tests.
    timeout:
        Maximum seconds before the subprocess is killed.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "--tb=short"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout,
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

class _LtEToLt(ast.NodeTransformer):
    """Replace the *first* ``<=`` (LtE) comparison with ``<`` (Lt).

    After the first replacement, the transformer sets ``self.mutated``
    to *True* and records the original & mutated expressions.  All
    subsequent ``<=`` nodes are left untouched.
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
                # Snapshot the original expression text.
                self.original_expr = ast.unparse(node)

                # Perform the mutation: LtE → Lt.
                node.ops[idx] = ast.Lt()

                # Snapshot the mutated expression text.
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

    Returns ``None`` if no ``<=`` comparison was found.
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
