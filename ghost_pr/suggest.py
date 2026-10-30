"""Local AI test suggestion for Ghost-PR (Milestone 3A).

This module interacts with a local Ollama instance (defaulting to
``qwen2.5-coder:7b``) to request a targeted pytest test suggestion
when a mutant survives during the demo workflow.

All suggestions are strictly labeled UNVERIFIED. Generated code is
displayed for human review only and is never executed or written to disk.
"""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 30


@dataclass
class SuggestionResult:
    """Outcome of an AI test suggestion request."""

    test_code: Optional[str]
    error: Optional[str]
    model: str
    unverified: bool = True

    @property
    def is_success(self) -> bool:
        return self.test_code is not None and self.error is None


# ── General Boundary Analysis Concept ────────────────────────────────
# Explains boundary value analysis conceptually and generally, without
# hardcoding any domain answers, function names, or input values.

BOUNDARY_ANALYSIS_CONCEPT = (
    "Boundary Value Analysis Concept:\n"
    "Boundary value analysis is a software testing technique focused on evaluating "
    "inputs at the boundaries of equivalence partitions.\n"
    "Relational operators (such as <, <=, >, >=, ==, !=) establish strict decision boundaries.\n"
    "When a comparison operator is altered (for instance, changing <= to <), the behavior "
    "of the program remains identical for all interior values strictly above or strictly "
    "below the threshold.\n"
    "The behavior diverges exclusively at the exact boundary value where the two compared "
    "expressions are equal.\n"
    "To detect the mutation and kill the surviving mutant, a test case must probe this exact "
    "boundary equality condition, where the original operator and the mutated operator produce "
    "different outcomes."
)


def build_suggestion_prompt(
    original_function: str,
    mutated_function: str,
    existing_tests: str,
    import_path: str,
) -> str:
    """Construct the prompt sent to Ollama for generating a test suggestion.

    Explains boundary value analysis generally; does not hardcode any
    specific inputs, domain values, or answers.
    """
    return (
        "You are an expert automated testing assistant for Python and pytest.\n"
        "A mutation was applied to a Python function, and the mutation SURVIVED because "
        "the existing test suite does not test the boundary condition.\n"
        "Your task is to suggest exactly ONE new pytest test function that catches this mutation.\n\n"
        "The suggested test MUST satisfy the following requirements:\n"
        "1. It must PASS when run against the original function.\n"
        "2. It must FAIL when run against the mutated function.\n"
        f"3. It must import the target function using: {import_path}\n\n"
        f"{BOUNDARY_ANALYSIS_CONCEPT}\n\n"
        "--- Target Function (Original) ---\n"
        f"{original_function.strip()}\n\n"
        "--- Target Function (Mutated) ---\n"
        f"{mutated_function.strip()}\n\n"
        "--- Existing Tests ---\n"
        f"{existing_tests.strip()}\n\n"
        "Output Instructions:\n"
        "- Write exactly ONE standalone pytest test function.\n"
        f"- Include the required import statement: {import_path}\n"
        "- Provide the code inside a ```python ... ``` code block.\n"
        "- Do not include conversational commentary, explanations, or documentation outside the code block.\n"
    )


def extract_function_definition(source: str, function_name: str | None = None) -> str:
    """Extract a function definition from Python source code using AST.

    If function_name is provided, returns that function's definition.
    Otherwise returns the first function definition found.
    If no function is found or parsing fails, returns the stripped source.
    """
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if function_name is None or node.name == function_name:
                    return ast.unparse(node)
    except SyntaxError:
        pass
    return source.strip()


def extract_test_code(response_text: str, import_path: str | None = None) -> Optional[str]:
    """Extract and validate python test code from Ollama's response.

    Handles ```python ... ``` code blocks, generic ``` ... ``` blocks, or raw code.
    Ensures the extracted code contains a pytest test function and is valid syntax.
    """
    if not response_text or not response_text.strip():
        return None

    # Look for fenced code blocks: ```python ... ``` or ``` ... ```
    code_blocks = re.findall(
        r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\s*```",
        response_text,
        re.DOTALL,
    )

    candidate = None
    if code_blocks:
        # Prefer a code block that contains a test function
        for block in code_blocks:
            if "def test_" in block:
                candidate = block.strip()
                break
        if candidate is None:
            candidate = code_blocks[0].strip()
    else:
        # If no code fences, check if response contains def test_
        if "def test_" in response_text:
            lines = response_text.splitlines()
            code_lines = []
            capturing = False
            for line in lines:
                if line.startswith(("def test_", "import ", "from ", "@")):
                    capturing = True
                if capturing:
                    code_lines.append(line)
            if code_lines:
                candidate = "\n".join(code_lines).strip()
        else:
            candidate = response_text.strip()

    if not candidate or "def test_" not in candidate:
        return None

    # Ensure import is present if specified and not already imported
    if import_path and import_path not in candidate:
        # Check if the symbol or import is already present
        try:
            tree = ast.parse(candidate)
            has_import = any(
                isinstance(node, (ast.Import, ast.ImportFrom))
                for node in ast.walk(tree)
            )
            if not has_import:
                candidate = f"{import_path}\n\n{candidate}"
        except SyntaxError:
            pass

    # Verify extracted code is syntactically valid Python
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None

    return candidate


def request_test_suggestion(
    original_function: str,
    mutated_function: str,
    existing_tests: str,
    import_path: str,
    *,
    model: str | None = None,
    host: str | None = None,
    timeout: int | None = None,
) -> SuggestionResult:
    """Request a pytest test suggestion from local Ollama.

    Parameters
    ----------
    original_function : str
        Source of the original unmutated function.
    mutated_function : str
        Source of the mutated function.
    existing_tests : str
        Source of the existing test suite.
    import_path : str
        The import statement to include in the generated test.
    model : str, optional
        Ollama model name (defaults to env var or 'qwen2.5-coder:7b').
    host : str, optional
        Ollama host URL (defaults to env var or 'http://localhost:11434').
    timeout : int, optional
        Request timeout in seconds (defaults to env var or 30).
    """
    selected_model = (
        (model or "").strip()
        or os.environ.get("GHOST_MODEL", "").strip()
        or os.environ.get("OLLAMA_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    selected_host = (
        (host or "").strip()
        or os.environ.get("OLLAMA_HOST", "").strip()
        or DEFAULT_OLLAMA_HOST
    ).rstrip("/")

    req_timeout = timeout
    if req_timeout is None:
        try:
            req_timeout = int(os.environ.get("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT))
        except ValueError:
            req_timeout = DEFAULT_TIMEOUT

    prompt = build_suggestion_prompt(
        original_function=original_function,
        mutated_function=mutated_function,
        existing_tests=existing_tests,
        import_path=import_path,
    )

    payload = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    req = urllib.request.Request(
        f"{selected_host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=req_timeout) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError:
                return SuggestionResult(
                    test_code=None,
                    error="Received malformed response from Ollama (invalid JSON).",
                    model=selected_model,
                )

            if not isinstance(data, dict):
                return SuggestionResult(
                    test_code=None,
                    error="Received malformed response from Ollama (expected JSON object).",
                    model=selected_model,
                )

            # Check if Ollama returned an error field in the JSON body
            if "error" in data and not data.get("response"):
                err = str(data["error"])
                if "not found" in err.lower():
                    error_text = (
                        f"Model '{selected_model}' not found in Ollama. "
                        f"Run 'ollama pull {selected_model}' to download it."
                    )
                else:
                    error_text = f"Ollama error: {err}"
                return SuggestionResult(
                    test_code=None,
                    error=error_text,
                    model=selected_model,
                )

            raw_response = data.get("response")
            if not raw_response or not isinstance(raw_response, str) or not raw_response.strip():
                return SuggestionResult(
                    test_code=None,
                    error="Received empty response from Ollama.",
                    model=selected_model,
                )

            test_code = extract_test_code(raw_response, import_path=import_path)
            if test_code is None:
                return SuggestionResult(
                    test_code=None,
                    error="Model response was malformed: no valid pytest test function could be parsed.",
                    model=selected_model,
                )

            return SuggestionResult(
                test_code=test_code,
                error=None,
                model=selected_model,
            )

    except urllib.error.HTTPError as exc:
        err_msg = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
            err_json = json.loads(body)
            if isinstance(err_json, dict):
                err_msg = err_json.get("error", "")
        except Exception:
            err_msg = ""

        if exc.code == 404 or "not found" in str(err_msg).lower():
            error_text = (
                f"Model '{selected_model}' not found in Ollama. "
                f"Run 'ollama pull {selected_model}' to download it."
            )
        else:
            detail = err_msg or exc.reason
            error_text = f"Ollama HTTP error {exc.code}: {detail}"

        return SuggestionResult(
            test_code=None,
            error=error_text,
            model=selected_model,
        )

    except urllib.error.URLError as exc:
        reason_str = str(exc.reason)
        if (
            isinstance(exc.reason, (socket.timeout, TimeoutError))
            or "timed out" in reason_str.lower()
        ):
            error_text = f"Request to Ollama timed out after {req_timeout}s."
        else:
            error_text = (
                f"Ollama is unavailable at {selected_host} ({exc.reason}). "
                "Please ensure Ollama is installed and running."
            )
        return SuggestionResult(
            test_code=None,
            error=error_text,
            model=selected_model,
        )

    except (socket.timeout, TimeoutError):
        return SuggestionResult(
            test_code=None,
            error=f"Request to Ollama timed out after {req_timeout}s.",
            model=selected_model,
        )

    except Exception as exc:
        return SuggestionResult(
            test_code=None,
            error=f"Unexpected error communicating with Ollama: {exc}",
            model=selected_model,
        )


def display_suggestion(result: SuggestionResult) -> None:
    """Print the AI suggestion with mandatory UNVERIFIED label and disclaimer.

    Generated code is displayed for review only and is never automatically
    executed or written to disk.
    """
    print(f"\n{'─' * 60}")
    print("  💡  AI Test Suggestion [UNVERIFIED]")
    print(f"{'─' * 60}")
    print(f"  Model  : {result.model}")
    print("  Status : UNVERIFIED (Generation alone is not verification)")

    if result.error:
        print(f"\n  ⚠️   Could not generate AI test suggestion:")
        print(f"      {result.error}")
        print("      (Mutation testing results above were not affected)\n")
        return

    print("\n  Review the suggested test below before adding it to your test suite:")
    print("  (This test has NOT been executed or added to your project files)\n")
    lines = (result.test_code or "").splitlines()
    max_len = max([len(line) for line in lines] + [56])
    border = "─" * (max_len + 4)
    print(f"  ┌{border}┐")
    for line in lines:
        print(f"  │  {line:<{max_len}}  │")
    print(f"  └{border}┘")
    print("\n  ⚠️   DISCLAIMER: This test is UNVERIFIED.")
    print("      AI-generated tests can be inaccurate, incomplete, or flawed.")
    print("      Inspect inputs, boundary values, and assertions carefully")
    print("      before adopting any suggested code.\n")
