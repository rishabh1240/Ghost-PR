"""Tests for Milestone 3B: Hardened verification of reviewed AI-generated tests.

Verifies:
  1. Correct boundary test is VERIFIED (passes on original, fails on mutant).
  2. Test that fails on both versions is REJECTED (incompatible with original).
  3. Test passing on both versions is NON-DISTINGUISHING (mutation survives).
  4. Execution problems are INCONCLUSIVE (never marked verified):
     - Collection error
     - Setup error
     - Teardown error
     - Skipped test
     - Xfail / Xpass test
     - Subprocess timeout
  5. Hardened regression checks:
     - Misleading summary-like text in stdout (does not fool structured reporter)
     - Strict XPASS on mutant handled explicitly as INCONCLUSIVE
     - Missing or malformed report handled as INCONCLUSIVE
     - Mismatched collected test IDs between original and mutant runs
     - Context-managed cleanup when second workspace creation fails
  6. User consent prompt:
     - Defaults to 'No' (Enter)
     - Declines cleanly on EOF (EOFError / OSError)
     - Explicit decline ('n') executes no generated code
  7. Original demo fixtures and git repository remain unchanged.
  8. Full CLI end-to-end workflow: 'ghost haunt --demo --suggest'.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_pr.cli import _DEMO_DIR, main
from ghost_pr.engine import Outcome, TestResult
from ghost_pr.verify import (
    RunExecutionDetail,
    VerificationVerdict,
    _determine_verdict,
    parse_pytest_summary,
    prompt_verification_consent,
    run_verification_flow,
    verify_suggested_test,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _make_mock_ollama_response(test_code: str) -> MagicMock:
    """Mock Ollama HTTP response containing given test code."""
    payload = {
        "model": "qwen2.5-coder:7b",
        "response": f"```python\n{test_code}\n```",
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


_ORIGINAL_BANK_SRC = """
def withdraw(balance, amount):
    if amount <= balance:
        return balance - amount
    raise ValueError("Insufficient funds")
""".strip()

_MUTATED_BANK_SRC = """
def withdraw(balance, amount):
    if amount < balance:
        return balance - amount
    raise ValueError("Insufficient funds")
""".strip()


# ── 1. Correct Boundary Test Is Verified ────────────────────────────

def test_correct_boundary_test_is_verified():
    """Suggested boundary test passes on original and fails on mutant -> VERIFIED."""
    test_code = (
        "from bank import withdraw\n\n"
        "def test_withdraw_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.VERIFIED
    assert result.orig_detail.is_pure_pass is True
    assert result.orig_detail.passed_count == 1
    assert result.orig_detail.error_count == 0

    assert result.mut_detail.is_pure_test_call_failure is True
    assert result.mut_detail.failed_count == 1
    assert result.mut_detail.error_count == 0

    assert "Original passes + mutant fails: VERIFIED for this mutation." in result.reason


# ── 2. Test Failing on Both Is Rejected ─────────────────────────────

def test_test_failing_on_both_is_rejected():
    """Suggested test that fails on original behavior -> REJECTED."""
    test_code = (
        "from bank import withdraw\n\n"
        "def test_bad_expectation():\n"
        "    assert withdraw(100, 40) == 999\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.REJECTED
    assert result.orig_detail.failed_count == 1
    assert "Original fails: REJECTED" in result.reason


# ── 3. Test Passing on Both Is Non-Distinguishing ───────────────────

def test_test_passing_on_both_is_non_distinguishing():
    """Suggested test that passes on both original and mutant -> NON-DISTINGUISHING."""
    test_code = (
        "from bank import withdraw\n\n"
        "def test_interior_point():\n"
        "    assert withdraw(100, 40) == 60\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.NON_DISTINGUISHING
    assert result.orig_detail.is_pure_pass is True
    assert result.mut_detail.is_pure_pass is True
    assert "Both pass: NON-DISTINGUISHING" in result.reason


# ── 4. Execution Problems Are Inconclusive (Never Verified) ─────────

def test_collection_error_is_inconclusive():
    """Collection error (e.g. broken import) -> INCONCLUSIVE."""
    test_code = (
        "import nonexistent_package_xyz_999\n\n"
        "def test_never_runs():\n"
        "    assert True\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.INCONCLUSIVE
    assert result.orig_detail.outcome == Outcome.COLLECTION_ERROR
    assert result.verdict != VerificationVerdict.VERIFIED
    assert "Collection error" in result.reason


def test_setup_error_is_inconclusive():
    """Fixture setup error -> INCONCLUSIVE (not counted as failure/verified)."""
    test_code = (
        "import pytest\n"
        "from bank import withdraw\n\n"
        "@pytest.fixture\n"
        "def broken_setup():\n"
        "    raise RuntimeError('Setup crashed')\n\n"
        "def test_with_broken_setup(broken_setup):\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.INCONCLUSIVE
    assert result.orig_detail.error_count >= 1
    assert result.verdict != VerificationVerdict.VERIFIED
    assert "Setup error" in result.reason or "Setup or teardown error" in result.reason


def test_teardown_error_is_inconclusive():
    """Fixture teardown error -> INCONCLUSIVE."""
    test_code = (
        "import pytest\n"
        "from bank import withdraw\n\n"
        "@pytest.fixture\n"
        "def broken_teardown():\n"
        "    yield\n"
        "    raise RuntimeError('Teardown crashed')\n\n"
        "def test_with_broken_teardown(broken_teardown):\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.INCONCLUSIVE
    assert result.orig_detail.error_count >= 1
    assert result.verdict != VerificationVerdict.VERIFIED
    assert "Teardown error" in result.reason or "Setup or teardown error" in result.reason


def test_skipped_test_is_inconclusive():
    """Skipped test -> INCONCLUSIVE (skips must never count as verified)."""
    test_code = (
        "import pytest\n"
        "from bank import withdraw\n\n"
        "def test_skipped_case():\n"
        "    pytest.skip('Skipping this test')\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.INCONCLUSIVE
    assert result.orig_detail.skipped_count >= 1
    assert result.verdict != VerificationVerdict.VERIFIED
    assert "skipped" in result.reason


def test_xfail_test_is_inconclusive():
    """Xfail test -> INCONCLUSIVE."""
    test_code = (
        "import pytest\n"
        "from bank import withdraw\n\n"
        "@pytest.mark.xfail\n"
        "def test_xfail_case():\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.INCONCLUSIVE
    assert result.verdict != VerificationVerdict.VERIFIED
    assert "xfail" in result.reason


def test_timeout_is_inconclusive():
    """Subprocess timeout -> INCONCLUSIVE."""
    test_code = (
        "import time\n"
        "from bank import withdraw\n\n"
        "def test_slow():\n"
        "    time.sleep(10)\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
        timeout=1,
    )

    assert result.verdict == VerificationVerdict.INCONCLUSIVE
    assert result.orig_detail.outcome == Outcome.TIMEOUT
    assert result.verdict != VerificationVerdict.VERIFIED
    assert "timed out" in result.reason.lower()


# ── 5. Hardened Regression Checks ───────────────────────────────────

def test_misleading_summary_text_in_stdout_does_not_fool_verifier():
    """Tests printing fake summary lines (e.g. '1 passed in 0.00s') are evaluated via structured report."""
    # Test prints fake passing summary but actually fails assertion
    test_code = (
        "from bank import withdraw\n\n"
        "def test_tricky():\n"
        "    print('\\n1 passed in 0.00s')\n"
        "    print('============================== 1 passed in 0.00s ==============================')\n"
        "    assert withdraw(100, 40) == 999\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    # Must be REJECTED based on structured call failure, not tricked by stdout into VERIFIED
    assert result.verdict == VerificationVerdict.REJECTED
    assert "Original fails: REJECTED" in result.reason


def test_strict_xpass_is_explicitly_inconclusive():
    """Strict XPASS (fails session when xfail test passes) must be marked INCONCLUSIVE, not VERIFIED."""
    test_code = (
        "import pytest\n"
        "from bank import withdraw\n\n"
        "@pytest.mark.xfail(strict=True)\n"
        "def test_strict_xpass_case():\n"
        "    # Passes on original and passes on mutant -> strict XPASS fails mutant run\n"
        "    assert withdraw(100, 40) == 60\n"
    )

    result = verify_suggested_test(
        test_code=test_code,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
    )

    assert result.verdict == VerificationVerdict.INCONCLUSIVE
    assert result.verdict != VerificationVerdict.VERIFIED
    assert "strict XPASS" in result.reason or "xfail" in result.reason


def test_missing_report_is_inconclusive():
    """Missing or corrupted report on either run produces INCONCLUSIVE."""
    dummy_res = TestResult(Outcome.PASSED, 0, "", "")
    dummy_detail = RunExecutionDetail(Outcome.PASSED, 0, None, "", "", "💥 MISSING")

    verdict, reason = _determine_verdict(dummy_detail, dummy_detail, timeout=30)
    assert verdict == VerificationVerdict.INCONCLUSIVE
    assert "Missing, incomplete, or malformed structured report" in reason


def test_mismatched_collected_tests_is_inconclusive():
    """Mismatched test IDs between original and mutant runs produces INCONCLUSIVE."""
    orig_rep = {
        "session_completed": True,
        "exitstatus": 0,
        "collected_nodeids": ["test_suggested.py::test_alpha"],
        "collection_errors": [],
        "phases": [{"nodeid": "test_suggested.py::test_alpha", "when": "call", "outcome": "passed", "wasxfail": None}],
    }
    mut_rep = {
        "session_completed": True,
        "exitstatus": 1,
        "collected_nodeids": ["test_suggested.py::test_beta"],
        "collection_errors": [],
        "phases": [{"nodeid": "test_suggested.py::test_beta", "when": "call", "outcome": "failed", "wasxfail": None}],
    }

    orig_detail = RunExecutionDetail(Outcome.PASSED, 0, orig_rep, "", "", "✅ PASSED")
    mut_detail = RunExecutionDetail(Outcome.FAILED, 1, mut_rep, "", "", "❌ FAILED")

    verdict, reason = _determine_verdict(orig_detail, mut_detail, timeout=30)
    assert verdict == VerificationVerdict.INCONCLUSIVE
    assert "Mismatched collected test IDs" in reason


def test_context_manager_cleanup_on_second_dir_failure(monkeypatch):
    """If creating the second temporary workspace fails, the first is still cleaned up."""
    created_dirs = []
    real_temp_dir = tempfile.TemporaryDirectory

    class MockTempDir:
        def __init__(self, *args, **kwargs):
            self._real = real_temp_dir(*args, **kwargs)
            created_dirs.append(Path(self._real.name))

        def __enter__(self):
            if len(created_dirs) == 2:
                raise OSError("Simulated disk failure on second workspace creation")
            return self._real.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._real.__exit__(exc_type, exc_val, exc_tb)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", MockTempDir)

    with pytest.raises(OSError, match="Simulated disk failure"):
        verify_suggested_test(
            test_code="def test_x(): pass",
            original_source=_ORIGINAL_BANK_SRC,
            mutated_source=_MUTATED_BANK_SRC,
        )

    # Verify that the first directory was cleaned up and does not linger
    assert len(created_dirs) >= 1
    assert not created_dirs[0].exists(), "First directory must be cleaned up even if second creation fails"


# ── 6. User Interactive Consent Prompt ──────────────────────────────

def test_prompt_consent_accepted(monkeypatch):
    """User entering 'y' or 'yes' accepts verification."""
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert prompt_verification_consent() is True

    monkeypatch.setattr("builtins.input", lambda _: "YES")
    assert prompt_verification_consent() is True


def test_prompt_consent_default_no_on_enter(monkeypatch, capsys):
    """Default choice on Enter is 'No' (declines)."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert prompt_verification_consent() is False
    out = capsys.readouterr().out
    assert "Verification skipped." in out


def test_prompt_consent_explicit_no(monkeypatch, capsys):
    """User entering 'n' declines verification."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert prompt_verification_consent() is False
    out = capsys.readouterr().out
    assert "Verification skipped." in out


def test_prompt_consent_eof_declines_cleanly(monkeypatch, capsys):
    """EOFError declines cleanly without crashing."""
    def raise_eof(_):
        raise EOFError()

    monkeypatch.setattr("builtins.input", raise_eof)
    assert prompt_verification_consent() is False
    out = capsys.readouterr().out
    assert "Verification skipped." in out


def test_declining_executes_no_generated_code(monkeypatch):
    """Declining verification runs no tests and executes no generated code."""
    marker = Path("/tmp/ghost_b_side_effect.txt")
    if marker.exists():
        marker.unlink()

    test_with_side_effect = (
        "from bank import withdraw\n\n"
        "def test_danger():\n"
        "    with open('/tmp/ghost_b_side_effect.txt', 'w') as f:\n"
        "        f.write('ran')\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    monkeypatch.setattr("builtins.input", lambda _: "n")
    result = run_verification_flow(
        test_code=test_with_side_effect,
        original_source=_ORIGINAL_BANK_SRC,
        mutated_source=_MUTATED_BANK_SRC,
        ask_consent=True,
    )

    assert result is None
    assert not marker.exists()


# ── 7. Original Files & Cleanups ────────────────────────────────────

def test_original_demo_files_remain_unchanged():
    """Demo fixtures on disk must never be modified by verification."""
    bank_path = _DEMO_DIR / "bank.py"
    test_bank_path = _DEMO_DIR / "test_bank.py"

    before_bank = bank_path.read_text(encoding="utf-8")
    before_test = test_bank_path.read_text(encoding="utf-8")

    # Run verification with real pytest subprocesses
    verify_suggested_test(
        test_code="from bank import withdraw\ndef test_b(): assert withdraw(100, 100) == 0",
        original_source=before_bank,
        mutated_source=_MUTATED_BANK_SRC,
    )

    after_bank = bank_path.read_text(encoding="utf-8")
    after_test = test_bank_path.read_text(encoding="utf-8")

    assert before_bank == after_bank
    assert before_test == after_test


# ── 8. Full CLI End-to-End Integration ──────────────────────────────

def test_cli_demo_suggest_with_verification_accepted(monkeypatch, capsys):
    """'ghost haunt --demo --suggest' with user accepting verification."""
    mock_resp = _make_mock_ollama_response(
        "from bank import withdraw\n\n"
        "def test_withdraw_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--demo", "--suggest"])
        assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "Step 1 · Running baseline tests" in out
    assert "Step 2 · Applying AST mutation" in out
    assert "Step 3 · Running tests against the mutant" in out
    assert "MUTANT SURVIVED" in out
    assert "AI-Suggested Test" in out
    assert "SECURITY NOTICE:" in out
    assert "Checking this test against original and mutated code…" in out
    assert "Test Verification Result" in out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out
    assert "Cleaned up temporary workspace." in out


def test_cli_demo_suggest_with_verification_declined(monkeypatch, capsys):
    """'ghost haunt --demo --suggest' with user declining verification."""
    mock_resp = _make_mock_ollama_response(
        "from bank import withdraw\n\n"
        "def test_withdraw_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
    )

    monkeypatch.setattr("builtins.input", lambda _: "n")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--demo", "--suggest"])
        assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "AI-Suggested Test" in out
    assert "This test has not been added to your project." in out
    assert "Not yet executed." in out
    assert "SECURITY NOTICE:" in out
    assert "Verification skipped." in out
    assert "Test Verification Result" not in out
