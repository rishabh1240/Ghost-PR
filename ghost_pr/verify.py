"""Verification engine for AI-generated tests in Ghost-PR demo (Milestone 3B).

This module coordinates:
  1. Offering an explicit interactive choice to execute reviewed test code,
     with a security notice that temporary folders are not a security sandbox.
  2. Creating two isolated temporary workspaces: pristine original and mutated,
     managed via context managers to guarantee cleanup if creation of the
     second workspace fails.
  3. Writing the exact, unmodified suggested test into both workspaces.
  4. Running 'test_suggested.py' explicitly with a local pytest reporting plugin
     that records structured test node IDs, phase outcomes, and collection errors.
  5. Inspecting structured results rather than stdout regex as the source of truth:
     - Nonempty identical sets of collected test node IDs required.
     - Strict XPASS handled explicitly as inconclusive.
     - Collection/setup/teardown errors, timeouts, no tests, skips, and xfails
       are never counted as verified.
     - Missing, incomplete, or malformed reports are classified as INCONCLUSIVE.
     - Subprocess exit-code and timeout checks retained as additional checks.
  6. Classifying outcomes strictly:
     - Original passes + mutant fails during test call: VERIFIED.
     - Original fails: REJECTED.
     - Both pass: NON-DISTINGUISHING.
     - Any execution problem: INCONCLUSIVE with reason.
  7. Terminal output is reserved for human diagnostics only.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from ghost_pr.engine import (
    MutationCandidate,
    Outcome,
    TestResult,
    apply_mutation_to_source,
    run_pytest,
)


class VerificationVerdict(Enum):
    """Possible outcomes of the test verification process."""

    VERIFIED = auto()           # Original passes + mutant fails
    REJECTED = auto()           # Original fails during test-call phase
    NON_DISTINGUISHING = auto() # Both pass; mutation still survives
    INCONCLUSIVE = auto()       # Execution problems (setup/teardown/timeout/skip/etc.)


# ── Local Pytest Reporter Plugin Source ──────────────────────────────
# Injected into conftest.py in each temporary workspace to collect
# machine-readable structured results directly from pytest hooks.

_REPORTER_PLUGIN_SOURCE = """
import json

data = {
    "session_completed": False,
    "exitstatus": None,
    "collected_nodeids": [],
    "collection_errors": [],
    "phases": [],
}

def pytest_collection_modifyitems(session, config, items):
    data["collected_nodeids"] = [item.nodeid for item in items]

def pytest_collectreport(report):
    if report.failed:
        data["collection_errors"].append({
            "nodeid": report.nodeid,
            "message": str(report.longrepr) if report.longrepr else "",
        })

def pytest_runtest_logreport(report):
    longrepr_str = str(report.longrepr) if report.longrepr else ""
    is_strict_xpass = "[XPASS(strict)]" in longrepr_str or "[XPASS" in longrepr_str
    wasxfail = getattr(report, "wasxfail", None)
    if is_strict_xpass and wasxfail is None:
        wasxfail = "[XPASS(strict)]"

    data["phases"].append({
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
        "passed": report.passed,
        "failed": report.failed,
        "skipped": report.skipped,
        "wasxfail": wasxfail,
        "is_strict_xpass": is_strict_xpass,
        "longrepr": longrepr_str,
    })

def pytest_sessionfinish(session, exitstatus):
    data["session_completed"] = True
    data["exitstatus"] = exitstatus
    try:
        with open("_ghost_pr_report.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        with open("report.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
"""


@dataclass
class RunExecutionDetail:
    """Detailed execution information for a single test run."""

    outcome: Outcome
    exit_code: Optional[int]
    report: Optional[dict]
    stdout: str
    stderr: str
    status_label: str

    @property
    def passed_count(self) -> int:
        if not self.report:
            return 0
        return sum(
            1 for p in self.report.get("phases", [])
            if p.get("when") == "call" and p.get("outcome") == "passed" and p.get("wasxfail") is None
        )

    @property
    def failed_count(self) -> int:
        if not self.report:
            return 0
        return sum(
            1 for p in self.report.get("phases", [])
            if p.get("when") == "call" and p.get("outcome") == "failed" and p.get("wasxfail") is None
        )

    @property
    def error_count(self) -> int:
        if not self.report:
            return 1 if self.outcome != Outcome.PASSED else 0
        col_errs = len(self.report.get("collection_errors", []))
        setup_teardown_errs = sum(
            1 for p in self.report.get("phases", [])
            if p.get("when") in ("setup", "teardown") and p.get("outcome") != "passed"
        )
        return col_errs + setup_teardown_errs

    @property
    def skipped_count(self) -> int:
        if not self.report:
            return 0
        return sum(
            1 for p in self.report.get("phases", [])
            if p.get("outcome") == "skipped" and p.get("wasxfail") is None
        )

    @property
    def is_pure_pass(self) -> bool:
        if not self.report or not self.report.get("session_completed"):
            return False
        if self.outcome != Outcome.PASSED or self.exit_code != 0:
            return False
        if self.error_count > 0 or self.failed_count > 0 or self.skipped_count > 0:
            return False
        phases = self.report.get("phases", [])
        if any(p.get("wasxfail") is not None or p.get("is_strict_xpass") for p in phases):
            return False
        call_phases = [p for p in phases if p.get("when") == "call"]
        return len(call_phases) > 0 and all(p.get("outcome") == "passed" for p in call_phases)

    @property
    def is_pure_test_call_failure(self) -> bool:
        if not self.report or not self.report.get("session_completed"):
            return False
        if self.outcome != Outcome.FAILED or self.exit_code != 1:
            return False
        if self.error_count > 0 or self.skipped_count > 0:
            return False
        phases = self.report.get("phases", [])
        if any(p.get("wasxfail") is not None or p.get("is_strict_xpass") for p in phases):
            return False
        call_phases = [p for p in phases if p.get("when") == "call"]
        return any(p.get("outcome") == "failed" for p in call_phases)


@dataclass
class VerificationResult:
    """Structured result of verifying a suggested test on original and mutated code."""

    verdict: VerificationVerdict
    orig_detail: RunExecutionDetail
    mut_detail: RunExecutionDetail
    reason: str


def _compute_status_label(test_res: TestResult, report: dict | None) -> str:
    """Determine an accurate human diagnostic status label."""
    if test_res.outcome == Outcome.TIMEOUT:
        return "⏱️   TIMEOUT"

    if report is None or not report.get("session_completed"):
        if test_res.outcome == Outcome.INTERNAL_ERROR:
            return "💥  INTERNAL ERROR"
        return "💥  MALFORMED / MISSING REPORT"

    if report.get("collection_errors"):
        return "⚠️   COLLECTION ERROR"

    collected = report.get("collected_nodeids", [])
    if len(collected) == 0:
        return "🔍  NO TESTS COLLECTED"

    phases = report.get("phases", [])
    if any(p.get("when") == "setup" and p.get("outcome") != "passed" for p in phases):
        return "⚠️   SETUP ERROR"

    if any(p.get("when") == "teardown" and p.get("outcome") != "passed" for p in phases):
        return "⚠️   TEARDOWN ERROR"

    if any(p.get("is_strict_xpass") for p in phases):
        return "⚠️   STRICT XPASS"

    if any(p.get("outcome") == "skipped" and p.get("wasxfail") is None for p in phases):
        return "⏭️   SKIPPED"

    if any(p.get("wasxfail") is not None for p in phases):
        return "⚠️   XFAIL / XPASS"

    call_phases = [p for p in phases if p.get("when") == "call"]
    if any(p.get("outcome") == "failed" for p in call_phases):
        return "❌  FAILED"

    if all(p.get("outcome") == "passed" for p in call_phases) and len(call_phases) > 0:
        return "✅  PASSED"

    if test_res.outcome in (Outcome.INTERNAL_ERROR, Outcome.EXECUTION_ERROR):
        return "🛑  EXECUTION ERROR"

    return "⚠️   INCONCLUSIVE"


def parse_run_report(
    report_path: Path,
    test_res: TestResult,
) -> RunExecutionDetail:
    """Parse the JSON report produced by the pytest plugin."""
    report = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = None

    status_label = _compute_status_label(test_res, report)
    return RunExecutionDetail(
        outcome=test_res.outcome,
        exit_code=test_res.exit_code,
        report=report,
        stdout=test_res.stdout,
        stderr=test_res.stderr,
        status_label=status_label,
    )


def parse_pytest_summary(
    stdout: str,
    stderr: str,
    outcome: Outcome,
    exit_code: Optional[int],
) -> RunExecutionDetail:
    """Backwards-compatibility helper for synthetic test results."""
    passed_m = re.search(r"(\d+)\s+passed", stdout)
    failed_m = re.search(r"(\d+)\s+failed", stdout)
    errors_m = re.search(r"(\d+)\s+error", stdout)
    skipped_m = re.search(r"(\d+)\s+skipped", stdout)

    passed_count = int(passed_m.group(1)) if passed_m else 0
    failed_count = int(failed_m.group(1)) if failed_m else 0
    error_count = int(errors_m.group(1)) if errors_m else 0
    skipped_count = int(skipped_m.group(1)) if skipped_m else 0

    dummy_report = {
        "session_completed": outcome not in (Outcome.TIMEOUT, Outcome.INTERNAL_ERROR),
        "exitstatus": exit_code,
        "collected_nodeids": (
            ["test_suggested.py::test_placeholder"]
            if (passed_count + failed_count + skipped_count > 0)
            else []
        ),
        "collection_errors": (
            [{"nodeid": "test_suggested.py", "message": stderr}]
            if outcome == Outcome.COLLECTION_ERROR
            else []
        ),
        "phases": [],
    }
    for i in range(passed_count):
        dummy_report["phases"].append({
            "nodeid": f"test_{i}",
            "when": "call",
            "outcome": "passed",
            "wasxfail": None,
            "is_strict_xpass": False,
        })
    for i in range(failed_count):
        dummy_report["phases"].append({
            "nodeid": f"test_{i}",
            "when": "call",
            "outcome": "failed",
            "wasxfail": None,
            "is_strict_xpass": False,
        })
    for i in range(error_count):
        dummy_report["phases"].append({
            "nodeid": f"test_{i}",
            "when": "setup",
            "outcome": "failed",
            "wasxfail": None,
            "is_strict_xpass": False,
        })
    for i in range(skipped_count):
        dummy_report["phases"].append({
            "nodeid": f"test_{i}",
            "when": "call",
            "outcome": "skipped",
            "wasxfail": None,
            "is_strict_xpass": False,
        })

    status_label = _compute_status_label(
        TestResult(outcome, exit_code, stdout, stderr), dummy_report
    )
    return RunExecutionDetail(
        outcome=outcome,
        exit_code=exit_code,
        report=dummy_report,
        stdout=stdout,
        stderr=stderr,
        status_label=status_label,
    )


def _determine_verdict(
    orig: RunExecutionDetail,
    mut: RunExecutionDetail,
    timeout: int,
) -> tuple[VerificationVerdict, str]:
    """Classify verification outcome strictly according to Milestone 3B rules."""
    # 1. Subprocess timeout checks
    if orig.outcome == Outcome.TIMEOUT:
        return (
            VerificationVerdict.INCONCLUSIVE,
            f"Subprocess timed out executing original run after {timeout}s",
        )
    if mut.outcome == Outcome.TIMEOUT:
        return (
            VerificationVerdict.INCONCLUSIVE,
            f"Subprocess timed out executing mutant run after {timeout}s",
        )

    # 2. Missing, incomplete, or malformed structured report checks
    if orig.report is None or not orig.report.get("session_completed"):
        return (
            VerificationVerdict.INCONCLUSIVE,
            "Missing, incomplete, or malformed structured report on original run",
        )
    if mut.report is None or not mut.report.get("session_completed"):
        return (
            VerificationVerdict.INCONCLUSIVE,
            "Missing, incomplete, or malformed structured report on mutant run",
        )

    # 3. Collection error checks
    if orig.report.get("collection_errors"):
        err_msg = orig.report["collection_errors"][0].get("message", "").strip()
        first_line = err_msg.splitlines()[-1] if err_msg else "collection failure"
        return (
            VerificationVerdict.INCONCLUSIVE,
            f"Collection error on original run: {first_line}",
        )
    if mut.report.get("collection_errors"):
        err_msg = mut.report["collection_errors"][0].get("message", "").strip()
        first_line = err_msg.splitlines()[-1] if err_msg else "collection failure"
        return (
            VerificationVerdict.INCONCLUSIVE,
            f"Collection error on mutant run: {first_line}",
        )

    # 4. Collected test node ID checks (must be nonempty and identical)
    orig_nodeids = orig.report.get("collected_nodeids")
    mut_nodeids = mut.report.get("collected_nodeids")
    if not isinstance(orig_nodeids, list) or not isinstance(mut_nodeids, list):
        return (
            VerificationVerdict.INCONCLUSIVE,
            "Malformed collected test IDs in report",
        )
    if len(orig_nodeids) == 0:
        return (
            VerificationVerdict.INCONCLUSIVE,
            "No tests were collected on original run",
        )
    if len(mut_nodeids) == 0:
        return (
            VerificationVerdict.INCONCLUSIVE,
            "No tests were collected on mutant run",
        )
    if set(orig_nodeids) != set(mut_nodeids):
        return (
            VerificationVerdict.INCONCLUSIVE,
            f"Mismatched collected test IDs between original and mutant runs: original={sorted(orig_nodeids)}, mutant={sorted(mut_nodeids)}",
        )

    # 5. Analyze phases on original run: all tests must pass normally
    for phase in orig.report.get("phases", []):
        nodeid = phase.get("nodeid", "")
        when = phase.get("when")
        outcome = phase.get("outcome")
        wasxfail = phase.get("wasxfail")
        is_strict_xpass = phase.get("is_strict_xpass", False)

        if is_strict_xpass or wasxfail is not None:
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Test '{nodeid}' was marked xfail/xpass on original run",
            )
        if outcome == "skipped" or phase.get("skipped"):
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Test '{nodeid}' was skipped on original run",
            )
        if when == "setup" and outcome != "passed":
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Setup error on original run for '{nodeid}'",
            )
        if when == "teardown" and outcome != "passed":
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Teardown error on original run for '{nodeid}'",
            )
        if when == "call":
            if outcome == "failed":
                return (
                    VerificationVerdict.REJECTED,
                    "Original fails: REJECTED; incompatible with original behaviour.",
                )
            if outcome != "passed":
                return (
                    VerificationVerdict.INCONCLUSIVE,
                    f"Test '{nodeid}' did not pass cleanly on original run (outcome={outcome})",
                )

    # Subprocess exit-code check on original
    if orig.exit_code != 0:
        return (
            VerificationVerdict.INCONCLUSIVE,
            f"Subprocess exited with code {orig.exit_code} on original run",
        )

    # 6. Analyze phases on mutant run:
    # Setup/teardown/skip/xfail are inconclusive
    # At least one test must fail during call
    mut_call_failures: list[str] = []
    mut_call_passes: list[str] = []

    for phase in mut.report.get("phases", []):
        nodeid = phase.get("nodeid", "")
        when = phase.get("when")
        outcome = phase.get("outcome")
        wasxfail = phase.get("wasxfail")
        is_strict_xpass = phase.get("is_strict_xpass", False)

        if is_strict_xpass or wasxfail is not None:
            # Handle strict XPASS explicitly as inconclusive
            if is_strict_xpass or (when == "call" and outcome == "failed"):
                return (
                    VerificationVerdict.INCONCLUSIVE,
                    f"Test '{nodeid}' failed due to strict XPASS on mutant run",
                )
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Test '{nodeid}' was marked xfail/xpass on mutant run",
            )
        if outcome == "skipped" or phase.get("skipped"):
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Test '{nodeid}' was skipped on mutant run",
            )
        if when == "setup" and outcome != "passed":
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Setup error on mutant run for '{nodeid}'",
            )
        if when == "teardown" and outcome != "passed":
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Teardown error on mutant run for '{nodeid}'",
            )
        if when == "call":
            if outcome == "failed":
                mut_call_failures.append(nodeid)
            elif outcome == "passed":
                mut_call_passes.append(nodeid)
            else:
                return (
                    VerificationVerdict.INCONCLUSIVE,
                    f"Test '{nodeid}' had unexpected call phase outcome '{outcome}' on mutant run",
                )

    if len(mut_call_failures) == 0:
        if mut.exit_code != 0:
            return (
                VerificationVerdict.INCONCLUSIVE,
                f"Subprocess exited with code {mut.exit_code} on mutant run",
            )
        return (
            VerificationVerdict.NON_DISTINGUISHING,
            "Both pass: NON-DISTINGUISHING; mutation still survives.",
        )

    # At least one test failed during call phase!
    if mut.exit_code != 1:
        return (
            VerificationVerdict.INCONCLUSIVE,
            f"Expected exit code 1 for mutant test failure, got {mut.exit_code}",
        )

    return (
        VerificationVerdict.VERIFIED,
        "Original passes + mutant fails: VERIFIED for this mutation.",
    )


def prompt_verification_consent() -> bool:
    """Prompt the user for interactive consent to execute the suggested test.

    Explains that verification executes local Python code in temporary folders,
    which are NOT a security sandbox. Defaults to No ('N'). On EOF, declines cleanly.
    """
    print("  ⚠️   SECURITY NOTICE:")
    print("      Verifying this test will execute local Python code in a temporary directory.")
    print("      Temporary directories isolate file changes, but they are NOT a security sandbox.")
    try:
        response = input("  Execute and verify this suggested test? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        print("\n  Declined test verification — suggested test was not executed.\n")
        return False
    except (EOFError, OSError, KeyboardInterrupt):
        print("\n  Declined test verification — suggested test was not executed.\n")
        return False


def verify_suggested_test(
    test_code: str,
    original_source: str,
    mutated_source: str,
    *,
    timeout: int = 30,
) -> VerificationResult:
    """Run the exact suggested test against original and mutated demo copies.

    Uses context-managed temporary directories so failure while creating
    the second workspace still cleans up the first.
    Executes 'test_suggested.py' explicitly and collects structured pytest reports.
    """
    with tempfile.TemporaryDirectory(prefix="ghost_pr_verify_orig_") as orig_dir:
        with tempfile.TemporaryDirectory(prefix="ghost_pr_verify_mut_") as mut_dir:
            tmp_orig = Path(orig_dir)
            tmp_mut = Path(mut_dir)

            # 1. Setup original copy with bank.py, test_suggested.py, and reporter plugin
            (tmp_orig / "bank.py").write_text(original_source, encoding="utf-8")
            (tmp_orig / "test_suggested.py").write_text(test_code, encoding="utf-8")
            (tmp_orig / "_ghost_pr_reporter_plugin.py").write_text(_REPORTER_PLUGIN_SOURCE, encoding="utf-8")

            # 2. Setup mutated copy with bank.py, test_suggested.py, and reporter plugin
            (tmp_mut / "bank.py").write_text(mutated_source, encoding="utf-8")
            (tmp_mut / "test_suggested.py").write_text(test_code, encoding="utf-8")
            (tmp_mut / "_ghost_pr_reporter_plugin.py").write_text(_REPORTER_PLUGIN_SOURCE, encoding="utf-8")

            # 3. Run pytest explicitly on test_suggested.py in original copy
            orig_test_res = run_pytest(
                str(tmp_orig),
                timeout=timeout,
                extra_args=["-p", "_ghost_pr_reporter_plugin", "test_suggested.py"],
            )
            orig_detail = parse_run_report(tmp_orig / "report.json", orig_test_res)

            # 4. Run pytest explicitly on test_suggested.py in mutated copy
            mut_test_res = run_pytest(
                str(tmp_mut),
                timeout=timeout,
                extra_args=["-p", "_ghost_pr_reporter_plugin", "test_suggested.py"],
            )
            mut_detail = parse_run_report(tmp_mut / "report.json", mut_test_res)

            # 5. Determine verdict from structured reports
            verdict, reason = _determine_verdict(orig_detail, mut_detail, timeout)

            return VerificationResult(
                verdict=verdict,
                orig_detail=orig_detail,
                mut_detail=mut_detail,
                reason=reason,
            )


def display_verification_result(result: VerificationResult) -> None:
    """Display the verification results and verdict banner."""
    print(f"\n{'─' * 60}")
    print("  🔬  Test Verification Result")
    print(f"{'─' * 60}")
    print(f"  Original code : {result.orig_detail.status_label}")
    print(f"  Mutated code  : {result.mut_detail.status_label}")
    print()

    if result.verdict == VerificationVerdict.VERIFIED:
        print("  🎯  Original passes + mutant fails: VERIFIED for this mutation.")
        print("      The suggested test successfully catches the surviving mutant!\n")
    elif result.verdict == VerificationVerdict.REJECTED:
        print("  🚫  Original fails: REJECTED; incompatible with original behaviour.")
        print("      The suggested test does not pass on the original unmutated function.\n")
    elif result.verdict == VerificationVerdict.NON_DISTINGUISHING:
        print("  🧟  Both pass: NON-DISTINGUISHING; mutation still survives.")
        print("      The suggested test does not distinguish between original and mutated code.\n")
    else:  # INCONCLUSIVE
        print("  ⚠️   Execution problems: INCONCLUSIVE, with the reason.")
        print(f"      Reason: {result.reason}\n")


def run_verification_flow(
    test_code: str,
    original_source: str,
    mutated_source: str,
    *,
    ask_consent: bool = True,
    timeout: int = 30,
) -> Optional[VerificationResult]:
    """Coordinate the full interactive verification workflow."""
    if ask_consent:
        consent = prompt_verification_consent()
        if not consent:
            return None

    result = verify_suggested_test(
        test_code=test_code,
        original_source=original_source,
        mutated_source=mutated_source,
        timeout=timeout,
    )
    display_verification_result(result)
    return result


def _find_unique_test_path(workspace: Path) -> tuple[Path, str]:
    """Find an isolated target directory and unique test filename in workspace.

    If 'tests' directory exists, places test inside 'tests'.
    Else if 'test' directory exists, places test inside 'test'.
    Otherwise, places test in workspace root.

    Returns (full_path_in_workspace, relative_path_str).
    """
    if (workspace / "tests").is_dir():
        rel_dir = Path("tests")
    elif (workspace / "test").is_dir():
        rel_dir = Path("test")
    else:
        rel_dir = Path(".")

    base = "test_ghost_pr_suggested.py"
    target_path = workspace / rel_dir / base
    if not target_path.exists():
        return target_path, str((rel_dir / base).as_posix())

    idx = 1
    while (workspace / rel_dir / f"test_ghost_pr_suggested_{idx}.py").exists():
        idx += 1
    cand = f"test_ghost_pr_suggested_{idx}.py"
    return workspace / rel_dir / cand, str((rel_dir / cand).as_posix())


def _find_unique_plugin_module(workspace: Path) -> tuple[str, str]:
    """Find a unique module name for the reporter plugin in workspace.

    Returns (module_name, filename).
    """
    base = "_ghost_pr_reporter_plugin"
    if not (workspace / f"{base}.py").exists():
        return base, f"{base}.py"

    idx = 1
    while (workspace / f"{base}_{idx}.py").exists():
        idx += 1
    return f"{base}_{idx}", f"{base}_{idx}.py"


def verify_staged_test(
    pristine_snapshot: Path,
    candidate: MutationCandidate,
    test_code: str,
    *,
    timeout: int = 30,
) -> VerificationResult:
    """Run the exact suggested test against pristine and mutated staged snapshots.

    Creates two isolated temporary copies of the full staged snapshot, including
    project configuration, dependencies, and conftest fixtures. Never overwrites
    existing conftest.py files or project test files.
    """
    with tempfile.TemporaryDirectory(prefix="ghost_pr_staged_orig_") as orig_dir:
        with tempfile.TemporaryDirectory(prefix="ghost_pr_staged_mut_") as mut_dir:
            tmp_orig = Path(orig_dir)
            tmp_mut = Path(mut_dir)

            # 1. Copy pristine snapshot into both directories
            shutil.copytree(
                pristine_snapshot,
                tmp_orig,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
            )
            shutil.copytree(
                pristine_snapshot,
                tmp_mut,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
            )

            # 2. Mutate exactly the selected candidate in tmp_mut
            mut_target_path = tmp_mut / candidate.file_path
            orig_content = (tmp_orig / candidate.file_path).read_text(encoding="utf-8")
            mutated_content = apply_mutation_to_source(orig_content, candidate)
            mut_target_path.write_text(mutated_content, encoding="utf-8")

            # 3. Find unique test filename to never overwrite existing test files
            orig_test_path, rel_test_path = _find_unique_test_path(tmp_orig)
            mut_test_path = tmp_mut / rel_test_path

            orig_test_path.write_text(test_code, encoding="utf-8")
            mut_test_path.write_text(test_code, encoding="utf-8")

            # 4. Injected reporter plugin with a unique name to never overwrite conftest.py
            plugin_mod, plugin_file = _find_unique_plugin_module(tmp_orig)
            (tmp_orig / plugin_file).write_text(_REPORTER_PLUGIN_SOURCE, encoding="utf-8")
            (tmp_mut / plugin_file).write_text(_REPORTER_PLUGIN_SOURCE, encoding="utf-8")

            # 5. Run pytest explicitly on the suggested test file
            orig_test_res = run_pytest(
                str(tmp_orig),
                timeout=timeout,
                extra_args=["-p", plugin_mod, rel_test_path],
            )
            orig_detail = parse_run_report(tmp_orig / "_ghost_pr_report.json", orig_test_res)

            mut_test_res = run_pytest(
                str(tmp_mut),
                timeout=timeout,
                extra_args=["-p", plugin_mod, rel_test_path],
            )
            mut_detail = parse_run_report(tmp_mut / "_ghost_pr_report.json", mut_test_res)

            # 6. Determine verdict from structured reports
            verdict, reason = _determine_verdict(orig_detail, mut_detail, timeout)

            return VerificationResult(
                verdict=verdict,
                orig_detail=orig_detail,
                mut_detail=mut_detail,
                reason=reason,
            )


def run_staged_verification_flow(
    pristine_snapshot: Path,
    candidate: MutationCandidate,
    test_code: str,
    *,
    ask_consent: bool = True,
    timeout: int = 30,
) -> Optional[VerificationResult]:
    """Coordinate the interactive verification workflow for staged changes."""
    if ask_consent:
        consent = prompt_verification_consent()
        if not consent:
            return None

    result = verify_staged_test(
        pristine_snapshot=pristine_snapshot,
        candidate=candidate,
        test_code=test_code,
        timeout=timeout,
    )
    display_verification_result(result)
    return result
