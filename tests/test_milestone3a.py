"""Tests for Milestone 3A: Local AI test suggestions for the Ghost-PR demo.

All tests use mocked model responses and mocked network calls so they
run deterministically and do NOT require a running Ollama service.

Covers:
  1. Default model ('qwen2.5-coder:7b') and configurable model mechanisms
     (argument, GHOST_MODEL, OLLAMA_MODEL).
  2. Prompt contains original function, mutated function, existing tests,
     and import path.
  3. General boundary value analysis concept in prompt without hardcoding
     the bank domain answer or inputs.
  4. Successful test suggestion parsing from code fence or plain code.
  5. Missing import automatically prepended to generated test code.
  6. Display formatting with mandatory UNVERIFIED label and disclaimer.
  7. Robust error handling without crashes:
     - Ollama unavailable (connection refused)
     - Missing model (HTTP 404 / 'not found' JSON)
     - Request timeout
     - Malformed JSON response
     - Empty response
     - Malformed Python syntax
     - Missing test function
  8. Full CLI integration: 'ghost haunt --demo --suggest' with mock responses.
  9. Preserving mutation testing results even when Ollama is unavailable.
  10. Rejection of '--suggest' when run without '--demo'.
  11. Generated code is never executed and never modifies project files.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_pr.cli import _DEMO_DIR, main
from ghost_pr.suggest import (
    BOUNDARY_ANALYSIS_CONCEPT,
    DEFAULT_MODEL,
    SuggestionResult,
    build_suggestion_prompt,
    display_suggestion,
    extract_function_definition,
    extract_test_code,
    request_test_suggestion,
)


# ── Helpers for Mocking urllib ──────────────────────────────────────

def _make_mock_response(body_dict: dict, status: int = 200) -> MagicMock:
    """Create a mock HTTP response context manager returning JSON bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(body_dict).encode("utf-8")
    mock_resp.status = status
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def _make_raw_mock_response(raw_bytes: bytes, status: int = 200) -> MagicMock:
    """Create a mock HTTP response context manager returning raw bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw_bytes
    mock_resp.status = status
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


# ── 1. Model Configuration ──────────────────────────────────────────

def test_default_model():
    """Default model must be qwen2.5-coder:7b."""
    assert DEFAULT_MODEL == "qwen2.5-coder:7b"


def test_model_selection_order(monkeypatch):
    """Model selection follows: explicit arg > GHOST_MODEL > OLLAMA_MODEL > default."""
    monkeypatch.delenv("GHOST_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    captured_payloads = []

    def mock_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(payload)
        return _make_mock_response({"response": "```python\ndef test_ok():\n    pass\n```"})

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        # 1. Default fallback
        request_test_suggestion("def f(): pass", "def f(): pass", "def test(): pass", "import f")
        assert captured_payloads[-1]["model"] == "qwen2.5-coder:7b"

        # 2. OLLAMA_MODEL env var
        monkeypatch.setenv("OLLAMA_MODEL", "env-ollama:latest")
        request_test_suggestion("def f(): pass", "def f(): pass", "def test(): pass", "import f")
        assert captured_payloads[-1]["model"] == "env-ollama:latest"

        # 3. GHOST_MODEL env var overrides OLLAMA_MODEL
        monkeypatch.setenv("GHOST_MODEL", "env-ghost:latest")
        request_test_suggestion("def f(): pass", "def f(): pass", "def test(): pass", "import f")
        assert captured_payloads[-1]["model"] == "env-ghost:latest"

        # 4. Explicit model argument overrides env vars
        request_test_suggestion(
            "def f(): pass",
            "def f(): pass",
            "def test(): pass",
            "import f",
            model="explicit-model:7b",
        )
        assert captured_payloads[-1]["model"] == "explicit-model:7b"


# ── 2. Prompt Construction ──────────────────────────────────────────

def test_build_suggestion_prompt_contains_all_requirements():
    """Prompt must include original fn, mutated fn, existing tests, and import path."""
    orig = "def compute(x):\n    return x <= 10"
    mut = "def compute(x):\n    return x < 10"
    tests = "def test_small():\n    assert compute(5) is True"
    imp = "from math_pkg import compute"

    prompt = build_suggestion_prompt(orig, mut, tests, imp)

    assert orig in prompt
    assert mut in prompt
    assert tests in prompt
    assert imp in prompt
    assert "It must PASS when run against the original function" in prompt
    assert "It must FAIL when run against the mutated function" in prompt
    assert "ONE new pytest test function" in prompt


def test_boundary_analysis_explanation_is_general():
    """Boundary value analysis must be explained generally without hardcoding bank domain answers."""
    concept = BOUNDARY_ANALYSIS_CONCEPT.lower()

    # Must explain boundaries conceptually
    assert "boundary" in concept
    assert "relational operators" in concept or "<=" in concept
    assert "diverges exclusively at the exact boundary value" in concept or "boundary equality" in concept

    # Must NOT mention domain-specific bank details or inputs
    assert "bank" not in concept
    assert "withdraw" not in concept
    assert "balance" not in concept
    assert "amount" not in concept
    assert "100" not in concept
    assert "40" not in concept


# ── 3. Test Code Extraction & AST Validation ────────────────────────

def test_extract_test_code_from_markdown_fence():
    """Extracts valid test from ```python ... ``` block."""
    response = (
        "Here is a suggested test that kills the mutant:\n\n"
        "```python\n"
        "from bank import withdraw\n\n"
        "def test_withdraw_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
        "```\n\n"
        "Hope this helps!"
    )
    extracted = extract_test_code(response, import_path="from bank import withdraw")
    assert extracted is not None
    assert "def test_withdraw_boundary():" in extracted
    assert "from bank import withdraw" in extracted


def test_extract_test_code_from_fence_without_language_tag():
    """Extracts test from generic ``` ... ``` block."""
    response = (
        "```\n"
        "from bank import withdraw\n\n"
        "def test_exact_balance():\n"
        "    assert withdraw(50, 50) == 0\n"
        "```"
    )
    extracted = extract_test_code(response)
    assert extracted is not None
    assert "def test_exact_balance():" in extracted


def test_extract_test_code_prepends_missing_import():
    """When the LLM omits the import, extract_test_code prepends the given import path."""
    response = (
        "```python\n"
        "def test_boundary_case():\n"
        "    assert withdraw(100, 100) == 0\n"
        "```"
    )
    extracted = extract_test_code(response, import_path="from bank import withdraw")
    assert extracted is not None
    assert extracted.startswith("from bank import withdraw")
    assert "def test_boundary_case():" in extracted


def test_extract_test_code_rejects_malformed_syntax():
    """Syntax errors in the generated code return None."""
    response = (
        "```python\n"
        "def test_syntax_error(\n"
        "    assert True\n"
        "```"
    )
    extracted = extract_test_code(response)
    assert extracted is None


def test_extract_test_code_rejects_no_test_function():
    """Explanations without a 'def test_' function return None."""
    response = "You should add a test with withdraw(100, 100) == 0 to catch the mutant."
    extracted = extract_test_code(response)
    assert extracted is None


def test_extract_function_definition():
    """Extracts function definition using AST."""
    source = (
        "# Module comment\n\n"
        "def withdraw(balance, amount):\n"
        "    if amount <= balance:\n"
        "        return balance - amount\n"
        "    raise ValueError()\n"
    )
    fn_def = extract_function_definition(source, "withdraw")
    assert fn_def.startswith("def withdraw(balance, amount):")
    assert "Module comment" not in fn_def


# ── 4. Error Handling (Without Crashing) ─────────────────────────────

def test_ollama_unavailable_handled_gracefully():
    """When Ollama is down (ConnectionRefused), returns an informative error result."""
    exc = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    with patch("urllib.request.urlopen", side_effect=exc):
        result = request_test_suggestion(
            original_function="def f(): pass",
            mutated_function="def f(): pass",
            existing_tests="def test(): pass",
            import_path="import f",
        )
    assert not result.is_success
    assert result.test_code is None
    assert result.unverified is True
    assert "Ollama is unavailable at" in result.error
    assert "Connection refused" in result.error


def test_ollama_missing_model_404_handled_gracefully():
    """When model is not found (HTTP 404), returns pull instructions."""
    fp = io.BytesIO(b'{"error": "model \'missing-model:7b\' not found"}')
    exc = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=fp,
    )
    with patch("urllib.request.urlopen", side_effect=exc):
        result = request_test_suggestion(
            original_function="def f(): pass",
            mutated_function="def f(): pass",
            existing_tests="def test(): pass",
            import_path="import f",
            model="missing-model:7b",
        )
    assert not result.is_success
    assert "Model 'missing-model:7b' not found in Ollama" in result.error
    assert "ollama pull missing-model:7b" in result.error


def test_ollama_missing_model_json_error_handled_gracefully():
    """When Ollama returns a JSON body with an error saying 'not found'."""
    fp = io.BytesIO(b'{"error": "model \'custom:latest\' not found, try pulling it first"}')
    exc = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=500,
        msg="Internal Error",
        hdrs={},
        fp=fp,
    )
    with patch("urllib.request.urlopen", side_effect=exc):
        result = request_test_suggestion(
            original_function="def f(): pass",
            mutated_function="def f(): pass",
            existing_tests="def test(): pass",
            import_path="import f",
            model="custom:latest",
        )
    assert not result.is_success
    assert "Model 'custom:latest' not found in Ollama" in result.error
    assert "ollama pull custom:latest" in result.error


def test_ollama_request_timeout_handled_gracefully():
    """When the request times out, returns timeout error message."""
    exc = socket.timeout("timed out")
    with patch("urllib.request.urlopen", side_effect=exc):
        result = request_test_suggestion(
            original_function="def f(): pass",
            mutated_function="def f(): pass",
            existing_tests="def test(): pass",
            import_path="import f",
            timeout=15,
        )
    assert not result.is_success
    assert "timed out after 15s" in result.error


def test_ollama_malformed_json_response_handled_gracefully():
    """Non-JSON response from proxy/host returns malformed error."""
    mock_resp = _make_raw_mock_response(b"<html>Bad Gateway 502</html>")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = request_test_suggestion(
            original_function="def f(): pass",
            mutated_function="def f(): pass",
            existing_tests="def test(): pass",
            import_path="import f",
        )
    assert not result.is_success
    assert "malformed response from Ollama (invalid JSON)" in result.error


def test_ollama_empty_response_handled_gracefully():
    """Empty response from Ollama returns empty response error."""
    mock_resp = _make_mock_response({"response": ""})
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = request_test_suggestion(
            original_function="def f(): pass",
            mutated_function="def f(): pass",
            existing_tests="def test(): pass",
            import_path="import f",
        )
    assert not result.is_success
    assert "Received empty response from Ollama" in result.error


def test_ollama_malformed_code_response_handled_gracefully():
    """Ollama output without a valid test returns malformed model response error."""
    mock_resp = _make_mock_response({"response": "I am not able to generate tests for this."})
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = request_test_suggestion(
            original_function="def f(): pass",
            mutated_function="def f(): pass",
            existing_tests="def test(): pass",
            import_path="import f",
        )
    assert not result.is_success
    assert "Model response was malformed" in result.error


# ── 5. UNVERIFIED Label and Disclaimer ──────────────────────────────

def test_display_suggestion_unverified_labels(capsys):
    """Output MUST be explicitly labeled UNVERIFIED and state that generation is not verification."""
    result = SuggestionResult(
        test_code="def test_sample():\n    assert True",
        error=None,
        model="qwen2.5-coder:7b",
        unverified=True,
    )
    display_suggestion(result)
    out = capsys.readouterr().out

    assert "AI Test Suggestion [UNVERIFIED]" in out
    assert "Status : UNVERIFIED (Generation alone is not verification)" in out
    assert "DISCLAIMER: This test is UNVERIFIED" in out
    assert "This test has NOT been executed or added to your project files" in out
    assert "def test_sample():" in out


def test_display_suggestion_error_labels(capsys):
    """When generation fails, displays error without losing mutation testing results."""
    result = SuggestionResult(
        test_code=None,
        error="Ollama is unavailable at http://localhost:11434 (Connection refused).",
        model="qwen2.5-coder:7b",
        unverified=True,
    )
    display_suggestion(result)
    out = capsys.readouterr().out

    assert "AI Test Suggestion [UNVERIFIED]" in out
    assert "Could not generate AI test suggestion:" in out
    assert "Mutation testing results above were not affected" in out


# ── 6. CLI Integration: ghost haunt --demo --suggest ─────────────────

def test_cli_demo_suggest_success(capsys):
    """'ghost haunt --demo --suggest' requests suggestion and reports results."""
    generated_test = (
        "```python\n"
        "from bank import withdraw\n\n"
        "def test_withdraw_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
        "```"
    )
    mock_resp = _make_mock_response({"response": generated_test})

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--demo", "--suggest"])
        assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "Ghost-PR  ·  haunt --demo --suggest" in out
    assert "MUTANT SURVIVED" in out
    assert "AI Test Suggestion [UNVERIFIED]" in out
    assert "def test_withdraw_boundary():" in out
    assert "assert withdraw(100, 100) == 0" in out
    assert "DISCLAIMER: This test is UNVERIFIED" in out


def test_cli_demo_suggest_with_custom_model(capsys):
    """'ghost haunt --demo --suggest --model my-model:14b' passes model to request."""
    captured_payloads = []

    def mock_urlopen(req, timeout=None):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return _make_mock_response({
            "response": "```python\nfrom bank import withdraw\ndef test_b():\n    assert withdraw(10, 10) == 0\n```"
        })

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--demo", "--suggest", "--model", "my-model:14b"])
        assert exc.value.code == 0

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["model"] == "my-model:14b"
    out = capsys.readouterr().out
    assert "Model  : my-model:14b" in out


def test_cli_demo_suggest_ollama_down_preserves_mutation_results(capsys):
    """When Ollama is unavailable during --suggest, mutation results are NOT lost."""
    exc = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    with patch("urllib.request.urlopen", side_effect=exc):
        with pytest.raises(SystemExit) as exc_exit:
            main(["haunt", "--demo", "--suggest"])
        assert exc_exit.value.code == 0

    out = capsys.readouterr().out
    # Mutation results must be present
    assert "Step 1 · Running baseline tests" in out
    assert "Step 2 · Applying AST mutation" in out
    assert "Step 3 · Running tests against the mutant" in out
    assert "MUTANT SURVIVED" in out
    assert "Original expression : amount <= balance" in out
    assert "Mutated expression  : amount < balance" in out
    # Suggestion error reported cleanly
    assert "Could not generate AI test suggestion:" in out
    assert "Ollama is unavailable" in out
    assert "Mutation testing results above were not affected" in out


def test_cli_suggest_without_demo_rejected(capsys):
    """'ghost haunt --suggest' without --demo must exit with code 1 and helpful message."""
    with pytest.raises(SystemExit) as exc:
        main(["haunt", "--suggest"])
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "only supported in demo mode (--demo)" in out


def test_generated_code_never_executed():
    """Generated test code must NEVER be executed automatically by Ghost-PR."""
    marker_file = Path("/tmp/ghost_pr_danger_marker.txt")
    if marker_file.exists():
        marker_file.unlink()

    # Craft a model response that would create a file if executed
    malicious_suggestion = (
        "```python\n"
        "from bank import withdraw\n\n"
        "def test_with_side_effect():\n"
        "    with open('/tmp/ghost_pr_danger_marker.txt', 'w') as f:\n"
        "        f.write('executed')\n"
        "    assert withdraw(100, 100) == 0\n"
        "```"
    )
    mock_resp = _make_mock_response({"response": malicious_suggestion})

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            main(["haunt", "--demo", "--suggest"])
        assert exc.value.code == 0

    # Ensure the marker file was NOT created (code was displayed only, never executed)
    assert not marker_file.exists()


def test_original_demo_fixtures_remain_untouched():
    """Original bank fixture files on disk must never be modified."""
    bank_py = _DEMO_DIR / "bank.py"
    test_bank_py = _DEMO_DIR / "test_bank.py"

    content_bank = bank_py.read_text(encoding="utf-8")
    content_test = test_bank_py.read_text(encoding="utf-8")

    assert "amount <= balance" in content_bank
    assert "def test_withdraw_basic():" in content_test
    assert "test_withdraw_boundary" not in content_test

