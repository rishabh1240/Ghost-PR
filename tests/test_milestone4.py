"""Tests for Milestone 4: AI test suggestions and verification for staged Python changes.

Verifies:
  1. Correct test suggestion verifies (passes on original, fails on mutant).
  2. Incorrect suggestion is rejected (fails on original).
  3. Non-distinguishing suggestion survives on mutant.
  4. User declining runs no generated code.
  5. Flat module layout imports from snapshot.
  6. src package layout imports from snapshot.
  7. Existing conftest.py fixtures remain usable and unchanged.
  8. Unsupported targets (async, class method, module-level) reported and skipped.
  9. No surviving mutants cleanly skips AI generation.
  10. Ollama failure preserves mutation report.
  11. Existing test files are never overwritten.
  12. Working files and Git index remain pristine throughout.
  13. CLI end-to-end dispatch: 'ghost haunt --suggest' and '--model'.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_pr.cli import main
from ghost_pr.engine import MutationCandidate
from ghost_pr.haunt import haunt_staged_changes
from ghost_pr.suggest import TargetFunctionInfo, analyze_mutation_target
from ghost_pr.verify import (
    VerificationVerdict,
    run_staged_verification_flow,
    verify_staged_test,
)


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo with required user config."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)


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


# ── 1. Correct Suggestion Verifies ───────────────────────────────────

def test_staged_suggest_correct_suggestion_verifies(tmp_path: Path, monkeypatch, capsys):
    """Correct suggestion passes on original and fails on mutant in staged workflow."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    # Function with <= boundary
    (repo / "discount.py").write_text(
        "def apply_discount(price: int, threshold: int) -> int:\n"
        "    if threshold <= price:\n"
        "        return price - 10\n"
        "    return price\n",
        encoding="utf-8",
    )
    # Existing weak test (tests interior point, mutant survives)
    (repo / "test_discount.py").write_text(
        "from discount import apply_discount\n\n"
        "def test_discount_interior():\n"
        "    assert apply_discount(150, 100) == 140\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "discount.py", "test_discount.py"], cwd=repo, check=True)

    suggested_test = (
        "from discount import apply_discount\n\n"
        "def test_discount_boundary():\n"
        "    assert apply_discount(100, 100) == 90\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    orig_discount_content = (repo / "discount.py").read_text(encoding="utf-8")
    orig_test_content = (repo / "test_discount.py").read_text(encoding="utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    assert "MUTANT SURVIVED" in out
    assert "AI Test Suggestion for discount.py:2" in out
    assert "AI-Suggested Test" in out
    assert "SECURITY NOTICE:" in out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out

    # Verify working files and Git index remain untouched
    assert (repo / "discount.py").read_text(encoding="utf-8") == orig_discount_content
    assert (repo / "test_discount.py").read_text(encoding="utf-8") == orig_test_content
    status_proc = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    # Only the staged files we staged originally are in status
    assert "test_suggested" not in status_proc.stdout
    assert "test_ghost_pr" not in status_proc.stdout


# ── 2. Incorrect Suggestion Is Rejected ───────────────────────────────

def test_staged_suggest_incorrect_suggestion_rejected(tmp_path: Path, monkeypatch, capsys):
    """Suggested test that asserts invalid original behavior is REJECTED."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text(
        "def is_valid(n: int, limit: int) -> bool:\n"
        "    if n <= limit:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import is_valid\n\n"
        "def test_below():\n"
        "    assert is_valid(5, 10) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "calc.py", "test_calc.py"], cwd=repo, check=True)

    # Test asserts wrong value on original
    suggested_test = (
        "from calc import is_valid\n\n"
        "def test_bad_expectation():\n"
        "    assert is_valid(10, 10) is False\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    assert "Original fails: REJECTED; incompatible with original behaviour." in out


# ── 3. Non-Distinguishing Suggestion Survives ────────────────────────

def test_staged_suggest_non_distinguishing_suggestion(tmp_path: Path, monkeypatch, capsys):
    """Suggested test that passes on both original and mutant is NON-DISTINGUISHING."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text(
        "def is_valid(n: int, limit: int) -> bool:\n"
        "    if n <= limit:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import is_valid\n\n"
        "def test_below():\n"
        "    assert is_valid(5, 10) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "calc.py", "test_calc.py"], cwd=repo, check=True)

    # Suggestion tests another interior point that passes on both
    suggested_test = (
        "from calc import is_valid\n\n"
        "def test_another_interior():\n"
        "    assert is_valid(4, 10) is True\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    assert "Both pass: NON-DISTINGUISHING; mutation still survives." in out


# ── 4. Declining Consent Executes No Generated Code ───────────────────

def test_staged_suggest_declining_consent_executes_no_generated_test(tmp_path: Path, monkeypatch, capsys):
    """Declining verification prompt executes no generated code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text(
        "def check(x: int) -> bool:\n"
        "    if x <= 100:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import check\n\n"
        "def test_check():\n"
        "    assert check(50) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "calc.py", "test_calc.py"], cwd=repo, check=True)

    side_effect_marker = tmp_path / "side_effect_must_not_exist.txt"
    suggested_test = (
        f"from calc import check\n\n"
        f"def test_danger():\n"
        f"    with open('{side_effect_marker}', 'w') as f:\n"
        f"        f.write('executed')\n"
        f"    assert check(100) is True\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    # Decline consent
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    assert "AI-Suggested Test" in out
    assert "SECURITY NOTICE:" in out
    assert "Verification skipped." in out
    assert "Test Verification Result" not in out
    assert not side_effect_marker.exists()


# ── 5. src Package Layout Imports From Snapshot ───────────────────────

def test_staged_suggest_src_package_layout(tmp_path: Path, monkeypatch, capsys):
    """src/package/module.py layout derives correct import and verifies from snapshot."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    src_pkg = repo / "src" / "my_pkg"
    src_pkg.mkdir(parents=True)
    tests_dir = repo / "tests"
    tests_dir.mkdir()

    (src_pkg / "__init__.py").write_text("", encoding="utf-8")
    (src_pkg / "rates.py").write_text(
        "def check_limit(rate: int, limit: int) -> bool:\n"
        "    if rate <= limit:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (tests_dir / "test_rates.py").write_text(
        "from my_pkg.rates import check_limit\n\n"
        "def test_below():\n"
        "    assert check_limit(5, 10) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "src", "tests"], cwd=repo, check=True)

    suggested_test = (
        "from my_pkg.rates import check_limit\n\n"
        "def test_boundary():\n"
        "    assert check_limit(10, 10) is True\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    assert "Mutant 1/1 · src/my_pkg/rates.py:2" in out
    assert "MUTANT SURVIVED" in out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out


# ── 6. Existing conftest.py Fixtures Preserved and Usable ────────────

def test_staged_suggest_existing_conftest_fixtures_preserved(tmp_path: Path, monkeypatch, capsys):
    """Existing conftest.py files are never overwritten and their fixtures remain usable."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    tests_dir = repo / "tests"
    tests_dir.mkdir()

    original_conftest = (
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def custom_threshold():\n"
        "    return 50\n"
    )
    (repo / "conftest.py").write_text(original_conftest, encoding="utf-8")
    (repo / "calc.py").write_text(
        "def qualify(score: int, cut: int) -> str:\n"
        "    if cut <= score:\n"
        "        return 'pass'\n"
        "    return 'fail'\n",
        encoding="utf-8",
    )
    (tests_dir / "test_calc.py").write_text(
        "from calc import qualify\n\n"
        "def test_qualify_high(custom_threshold):\n"
        "    assert qualify(100, custom_threshold) == 'pass'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "conftest.py", "calc.py", "tests"], cwd=repo, check=True)

    # Suggested test uses custom_threshold fixture from existing conftest.py
    suggested_test = (
        "from calc import qualify\n\n"
        "def test_qualify_boundary(custom_threshold):\n"
        "    assert qualify(custom_threshold, custom_threshold) == 'pass'\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out

    # Ensure conftest.py in user repo is identical
    assert (repo / "conftest.py").read_text(encoding="utf-8") == original_conftest


# ── 7. Unsupported Targets Handled and Skipped Gracefully ────────────

def test_staged_suggest_unsupported_targets_skipped(tmp_path: Path, capsys):
    """Async functions, class methods, and module-level comparisons are skipped with clear reasons."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "services.py").write_text(
        "class OrderService:\n"
        "    def check_amount(self, amt: int, limit: int) -> bool:\n"
        "        if amt <= limit:\n"
        "            return True\n"
        "        return False\n\n"
        "async def async_check(x: int, y: int) -> bool:\n"
        "    if x <= y:\n"
        "        return True\n"
        "    return False\n\n"
        "MODULE_COMPARE = (10 <= 20)\n",
        encoding="utf-8",
    )
    (repo / "test_services.py").write_text(
        "import pytest\n"
        "from services import OrderService, MODULE_COMPARE\n\n"
        "def test_order():\n"
        "    assert OrderService().check_amount(5, 10) is True\n"
        "    assert MODULE_COMPARE is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "services.py", "test_services.py"], cwd=repo, check=True)

    # Run without Ollama mock; Ollama should not even be called
    code = haunt_staged_changes(repo, suggest=True)
    assert code == 0

    out = capsys.readouterr().out
    assert "AI Test Suggestion [SKIPPED]" in out
    assert "No supported surviving mutants found for AI test suggestions." in out
    assert "Class methods" in out and "not supported" in out
    assert "Async functions" in out
    assert "module level" in out


# ── 8. No Surviving Mutants Skips AI Generation ──────────────────────

def test_staged_suggest_no_surviving_mutants(tmp_path: Path, capsys):
    """When existing tests kill all mutants, AI test suggestions are skipped."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text(
        "def clamp(x: int) -> int:\n"
        "    if x <= 10:\n"
        "        return x\n"
        "    return 999\n",
        encoding="utf-8",
    )
    # Thorough tests that kill the <= to < mutant
    (repo / "test_calc.py").write_text(
        "from calc import clamp\n\n"
        "def test_clamp_below():\n"
        "    assert clamp(5) == 5\n\n"
        "def test_clamp_boundary():\n"
        "    assert clamp(10) == 10\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "calc.py", "test_calc.py"], cwd=repo, check=True)

    code = haunt_staged_changes(repo, suggest=True)
    assert code == 0

    out = capsys.readouterr().out
    assert "MUTANT KILLED" in out
    assert "No surviving mutants in staged changes — test suggestions are only requested when a mutant survives." in out


# ── 9. Ollama Failure Preserves Mutation Report ──────────────────────

def test_staged_suggest_ollama_failure_preserves_mutation_report(tmp_path: Path, capsys):
    """When Ollama is unavailable, mutation report is preserved and exit code is 0."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text(
        "def check(x: int) -> bool:\n"
        "    if x <= 10:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import check\n\n"
        "def test_check():\n"
        "    assert check(5) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "calc.py", "test_calc.py"], cwd=repo, check=True)

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    assert "MUTANT SURVIVED" in out
    assert "Original : x <= 10" in out
    assert "Mutated  : x < 10" in out
    assert "Mutation Testing Summary" in out
    assert "Could not generate AI test suggestion:" in out
    assert "Ollama is unavailable" in out
    assert "Mutation testing results above were not affected" in out


# ── 10. Existing Test Files Never Overwritten ────────────────────────

def test_staged_verification_never_overwrites_existing_test_files(tmp_path: Path, monkeypatch):
    """Verifier creates uniquely named test file when default name is taken."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text(
        "def is_positive(x: int) -> bool:\n"
        "    if 0 <= x:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    # Create existing test_ghost_pr_suggested.py in the repository
    existing_content = "# Pre-existing test file that must not be overwritten\ndef test_existing(): pass\n"
    (repo / "test_ghost_pr_suggested.py").write_text(existing_content, encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "from calc import is_positive\ndef test_pos(): assert is_positive(5) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    cand = MutationCandidate(
        file_path="calc.py",
        lineno=2,
        col_offset=7,
        original_expr="0 <= x",
        mutated_expr="0 < x",
    )

    suggested_test = (
        "from calc import is_positive\n"
        "def test_zero_boundary():\n"
        "    assert is_positive(0) is True\n"
    )

    result = verify_staged_test(
        pristine_snapshot=repo,
        candidate=cand,
        test_code=suggested_test,
    )

    assert result.verdict == VerificationVerdict.VERIFIED
    # Existing file on disk remains unchanged
    assert (repo / "test_ghost_pr_suggested.py").read_text(encoding="utf-8") == existing_content


# ── 11. CLI End-to-End Workflow ──────────────────────────────────────

def test_cli_staged_suggest_end_to_end(tmp_path: Path, monkeypatch, capsys):
    """Full CLI dispatch: 'ghost haunt --suggest --repo <path>' with custom model."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "boundary.py").write_text(
        "def check_bound(val: int) -> bool:\n"
        "    if val <= 50:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "test_boundary.py").write_text(
        "from boundary import check_bound\n\n"
        "def test_low():\n"
        "    assert check_bound(25) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "boundary.py", "test_boundary.py"], cwd=repo, check=True)

    suggested_test = (
        "from boundary import check_bound\n\n"
        "def test_bound_exact():\n"
        "    assert check_bound(50) is True\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--suggest", "--model", "custom-model:14b", "--repo", str(repo)])
        assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "haunt (staged changes) --suggest" in out
    assert "MUTANT SURVIVED" in out
    assert "AI Test Suggestion for boundary.py:2" in out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out


# ── 12. Skip Unsupported Survivor and Target First Supported Survivor ─

def test_staged_suggest_skips_unsupported_survivor_and_targets_first_supported(tmp_path: Path, monkeypatch, capsys):
    """When the first surviving mutant is unsupported, selects the first supported surviving mutant."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "funcs.py").write_text(
        "class Helper:\n"
        "    def method(self, x: int) -> bool:\n"
        "        if x <= 10:\n"
        "            return True\n"
        "        return False\n\n"
        "def top_level(a: int) -> bool:\n"
        "    if a <= 20:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (repo / "test_funcs.py").write_text(
        "from funcs import Helper, top_level\n\n"
        "def test_interior():\n"
        "    assert Helper().method(5) is True\n"
        "    assert top_level(15) is True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "funcs.py", "test_funcs.py"], cwd=repo, check=True)

    suggested_test = (
        "from funcs import top_level\n\n"
        "def test_top_level_boundary():\n"
        "    assert top_level(20) is True\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0

    out = capsys.readouterr().out
    # Mutant 1 (line 3, method) survived
    assert "Mutant 1/2 · funcs.py:3" in out
    # Mutant 2 (line 8, top-level) survived
    assert "Mutant 2/2 · funcs.py:8" in out
    # AI suggestion specifically targeted the supported top_level function (line 8)
    assert "AI Test Suggestion for funcs.py:8" in out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out


# ── 13. Retry: Successful Recovery ────────────────────────────────────

def test_retry_successful_recovery(tmp_path: Path, monkeypatch, capsys):
    """When first suggestion is REJECTED, one feedback-based retry recovers and verifies."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "shipping.py").write_text(
        "def shipping_cost(weight: int) -> int:\n"
        "    if weight <= 5:\n"
        "        return 50\n"
        "    return 100\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_shipping.py").write_text(
        "from shipping import shipping_cost\n\n"
        "def test_light():\n"
        "    assert shipping_cost(2) == 50\n\n"
        "def test_heavy():\n"
        "    assert shipping_cost(8) == 100\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "shipping.py", "tests"], cwd=repo, check=True)

    # First attempt: Inverted assertion (fails on original)
    bad_suggestion = (
        "from shipping import shipping_cost\n\n"
        "def test_shipping_boundary():\n"
        "    assert shipping_cost(5) == 100\n"
    )
    # Retry attempt: Correct assertion of original behavior
    good_suggestion = (
        "from shipping import shipping_cost\n\n"
        "def test_shipping_boundary():\n"
        "    assert shipping_cost(5) == 50\n"
    )

    resp_1 = _make_mock_ollama_response(bad_suggestion)
    resp_2 = _make_mock_ollama_response(good_suggestion)

    inputs = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch("urllib.request.urlopen", side_effect=[resp_1, resp_2]) as mock_url:
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0
        assert mock_url.call_count == 2

    out = capsys.readouterr().out
    # First attempt was rejected
    assert "Original fails: REJECTED; incompatible with original behaviour." in out
    # Retry notice and revised generation
    assert "Initial suggested test was REJECTED. Requesting 1 feedback-based revision" in out
    assert "AI-Suggested Test (Revised)" in out
    # Second attempt succeeded and was verified
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out


# ── 14. Retry: Second Bad Suggestion Stops Cleanly ────────────────────

def test_retry_second_bad_suggestion_stops_cleanly(tmp_path: Path, monkeypatch, capsys):
    """When revised suggestion is also bad, Ghost-PR stops after 1 retry and preserves both outcomes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "shipping.py").write_text(
        "def shipping_cost(weight: int) -> int:\n"
        "    if weight <= 5:\n"
        "        return 50\n"
        "    return 100\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_shipping.py").write_text(
        "from shipping import shipping_cost\n\n"
        "def test_light():\n"
        "    assert shipping_cost(2) == 50\n\n"
        "def test_heavy():\n"
        "    assert shipping_cost(8) == 100\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "shipping.py", "tests"], cwd=repo, check=True)

    bad_1 = (
        "from shipping import shipping_cost\n\n"
        "def test_bad_1():\n"
        "    assert shipping_cost(5) == 100\n"
    )
    bad_2 = (
        "from shipping import shipping_cost\n\n"
        "def test_bad_2():\n"
        "    assert shipping_cost(5) == 999\n"
    )

    resp_1 = _make_mock_ollama_response(bad_1)
    resp_2 = _make_mock_ollama_response(bad_2)

    inputs = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch("urllib.request.urlopen", side_effect=[resp_1, resp_2]) as mock_url:
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0
        assert mock_url.call_count == 2

    out = capsys.readouterr().out
    # Both verification results are displayed
    assert out.count("Original fails: REJECTED") == 2
    assert "Initial suggested test was REJECTED. Requesting 1 feedback-based revision" in out


# ── 15. Retry: Declining Revised Test Executes No Code ────────────────

def test_retry_declining_revised_test_executes_no_code(tmp_path: Path, monkeypatch, capsys):
    """User can accept initial test verification but decline the revised test."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "shipping.py").write_text(
        "def shipping_cost(weight: int) -> int:\n"
        "    if weight <= 5:\n"
        "        return 50\n"
        "    return 100\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_shipping.py").write_text(
        "from shipping import shipping_cost\n\n"
        "def test_light():\n"
        "    assert shipping_cost(2) == 50\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "shipping.py", "tests"], cwd=repo, check=True)

    # First attempt: Non-distinguishing test
    test_1 = (
        "from shipping import shipping_cost\n\n"
        "def test_non_dist():\n"
        "    assert shipping_cost(1) == 50\n"
    )
    # Revised attempt: Side-effect test
    marker = tmp_path / "side_effect_must_never_exist.txt"
    test_2 = (
        f"from shipping import shipping_cost\n\n"
        f"def test_danger():\n"
        f"    with open('{marker}', 'w') as f: f.write('ran')\n"
        f"    assert shipping_cost(5) == 50\n"
    )

    resp_1 = _make_mock_ollama_response(test_1)
    resp_2 = _make_mock_ollama_response(test_2)

    # First accept ('y'), then decline ('n')
    inputs = iter(["y", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch("urllib.request.urlopen", side_effect=[resp_1, resp_2]) as mock_url:
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0
        assert mock_url.call_count == 2

    out = capsys.readouterr().out
    assert "Both pass: NON-DISTINGUISHING; mutation still survives." in out
    assert "Initial suggested test was NON_DISTINGUISHING. Requesting 1 feedback-based revision" in out
    assert "Verification skipped." in out
    assert not marker.exists()


# ── 16. Retry: Hard Ceiling of At Most One Retry Enforced ─────────────

def test_retry_limit_hard_ceiling_enforced(tmp_path: Path, monkeypatch, capsys):
    """Ollama is called at most twice (1 initial + 1 retry). No retry on initial success."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "shipping.py").write_text(
        "def shipping_cost(weight: int) -> int:\n"
        "    if weight <= 5:\n"
        "        return 50\n"
        "    return 100\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_shipping.py").write_text(
        "from shipping import shipping_cost\n\n"
        "def test_light():\n"
        "    assert shipping_cost(2) == 50\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "shipping.py", "tests"], cwd=repo, check=True)

    # Test that passes on original and fails on mutant right away
    good_test = (
        "from shipping import shipping_cost\n\n"
        "def test_shipping_boundary():\n"
        "    assert shipping_cost(5) == 50\n"
    )
    resp = _make_mock_ollama_response(good_test)

    monkeypatch.setattr("builtins.input", lambda _: "y")

    with patch("urllib.request.urlopen", return_value=resp) as mock_url:
        code = haunt_staged_changes(repo, suggest=True)
        assert code == 0
        # When first test verifies, urlopen is called exactly ONCE (no retry triggered)
        assert mock_url.call_count == 1

    out = capsys.readouterr().out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out
    assert "Requesting 1 feedback-based revision" not in out


# ── 17. --yes Non-Interactive Verification Tests ─────────────────────

def test_cli_yes_without_suggest_rejected(capsys):
    """'--yes' without --suggest in staged mode must exit 1 with clear usage error."""
    with pytest.raises(SystemExit) as exc1:
        main(["haunt", "--staged", "--yes"])
    assert exc1.value.code == 1

    out1 = capsys.readouterr().out
    assert "Usage error: --yes requires --suggest" in out1

    with pytest.raises(SystemExit) as exc2:
        main(["staged", "--yes"])
    assert exc2.value.code == 1

    out2 = capsys.readouterr().out
    assert "Usage error: --yes requires --suggest" in out2


def test_cli_haunt_default_runs_demo_suggest_yes(monkeypatch, capsys):
    """'ghost haunt' (with no flags) runs demo with suggest and auto-verification by default."""
    suggested_test = (
        "from bank import withdraw\n\n"
        "def test_withdraw_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)

    input_mock = MagicMock(side_effect=AssertionError("input() must not be called in default ghost haunt"))
    monkeypatch.setattr("builtins.input", input_mock)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt"])
        assert exc.value.code == 0

    assert input_mock.call_count == 0

    out = capsys.readouterr().out
    assert "haunt --demo --suggest --yes" in out
    assert "AI-Suggested Test" in out
    assert "NOTICE (--yes): Automatic verification executes generated Python code" in out
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out


def test_cli_staged_option_and_command(tmp_path: Path, capsys):
    """'ghost haunt --staged' and 'ghost staged' analyze staged changes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text("def is_ten(x):\n    return x <= 10\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_calc.py").write_text("from calc import is_ten\ndef test_t():\n    assert is_ten(5)\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    with pytest.raises(SystemExit) as exc1:
        main(["haunt", "--staged", "--repo", str(repo)])
    assert exc1.value.code == 0
    out1 = capsys.readouterr().out
    assert "Step 1 · Running baseline tests on staged snapshot" in out1

    with pytest.raises(SystemExit) as exc2:
        main(["staged", "--repo", str(repo)])
    assert exc2.value.code == 0
    out2 = capsys.readouterr().out
    assert "Step 1 · Running baseline tests on staged snapshot" in out2


def test_cli_demo_suggest_yes_never_calls_input(monkeypatch, capsys):
    """'ghost haunt --demo --suggest --yes' verifies without calling input()."""
    suggested_test = (
        "from bank import withdraw\n\n"
        "def test_withdraw_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
    )
    mock_resp = _make_mock_ollama_response(suggested_test)

    # If input() is called, raise an error to fail the test immediately
    input_mock = MagicMock(side_effect=AssertionError("input() must not be called when --yes is supplied"))
    monkeypatch.setattr("builtins.input", input_mock)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--demo", "--suggest", "--yes"])
        assert exc.value.code == 0

    assert input_mock.call_count == 0

    out = capsys.readouterr().out
    assert "haunt --demo --suggest --yes" in out
    assert "AI-Suggested Test" in out
    assert "Checking this test against original and mutated code…" in out
    # Must print brief notice that automatic verification executes generated Python
    assert "NOTICE (--yes): Automatic verification executes generated Python code" in out
    # Verification actually executes and marks VERIFIED
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out


def test_cli_staged_suggest_yes_never_calls_input_including_during_retry(tmp_path: Path, monkeypatch, capsys):
    """'ghost haunt --suggest --yes' never calls input(), even across an automatic retry."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "shipping.py").write_text(
        "def shipping_cost(weight: int) -> int:\n"
        "    if weight <= 5:\n"
        "        return 50\n"
        "    return 100\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_shipping.py").write_text(
        "from shipping import shipping_cost\n\n"
        "def test_light():\n"
        "    assert shipping_cost(2) == 50\n\n"
        "def test_heavy():\n"
        "    assert shipping_cost(8) == 100\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "shipping.py", "tests"], cwd=repo, check=True)

    # 1st attempt: Inverted assertion (fails on original -> REJECTED)
    bad_test = "from shipping import shipping_cost\ndef test_bound(): assert shipping_cost(5) == 100\n"
    # 2nd attempt: Correct assertion -> VERIFIED
    good_test = "from shipping import shipping_cost\ndef test_bound(): assert shipping_cost(5) == 50\n"

    resp_1 = _make_mock_ollama_response(bad_test)
    resp_2 = _make_mock_ollama_response(good_test)

    # input() must never be called on either attempt
    input_mock = MagicMock(side_effect=AssertionError("input() must not be called when --yes is supplied"))
    monkeypatch.setattr("builtins.input", input_mock)

    with patch("urllib.request.urlopen", side_effect=[resp_1, resp_2]) as mock_url:
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--suggest", "--yes", "--repo", str(repo)])
        assert exc.value.code == 0
        assert mock_url.call_count == 2

    assert input_mock.call_count == 0

    out = capsys.readouterr().out
    assert "haunt (staged changes) --suggest --yes" in out
    # Notice printed exactly once
    assert out.count("NOTICE (--yes)") == 1
    # First attempt rejected, retry triggered automatically without prompting
    assert "Original fails: REJECTED" in out
    assert "Initial suggested test was REJECTED. Requesting 1 feedback-based revision" in out
    assert "AI-Suggested Test (Revised)" in out
    # Final retry verified
    assert "Original passes + mutant fails: VERIFIED for this mutation." in out


def test_default_without_yes_still_prompts(tmp_path: Path, monkeypatch, capsys):
    """Without --yes, the default interactive behavior prompts the user for consent."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "shipping.py").write_text(
        "def shipping_cost(weight: int) -> int:\n"
        "    if weight <= 5:\n"
        "        return 50\n"
        "    return 100\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_shipping.py").write_text(
        "from shipping import shipping_cost\ndef test_l(): assert shipping_cost(2) == 50\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "shipping.py", "tests"], cwd=repo, check=True)

    good_test = "from shipping import shipping_cost\ndef test_b(): assert shipping_cost(5) == 50\n"
    resp = _make_mock_ollama_response(good_test)

    input_mock = MagicMock(return_value="y")
    monkeypatch.setattr("builtins.input", input_mock)

    with patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--suggest", "--repo", str(repo)])
        assert exc.value.code == 0

    # input() was called to prompt for consent
    assert input_mock.call_count == 1
    assert "Execute and verify this suggested test?" in input_mock.call_args[0][0]
