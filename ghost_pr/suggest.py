"""Local AI test suggestion for Ghost-PR (Milestones 3A & 4).

This module interacts with a local Ollama instance (defaulting to
``qwen2.5-coder:7b``) to request a targeted pytest test suggestion
when a mutant survives during the demo or staged workflow.

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
from pathlib import Path
from typing import Any, Optional

from ghost_pr.engine import MutationCandidate, apply_mutation_to_source

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


@dataclass
class TargetFunctionInfo:
    """Information about a supported top-level function target for AI test suggestion."""

    func_name: str
    import_path: str
    original_func: str
    mutated_func: str
    existing_tests: str


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
    specific inputs, domain values, or answers. Bounded in size.
    """
    return (
        "You are an expert automated testing assistant for Python and pytest.\n"
        "A mutation was applied to a Python function, and the mutation SURVIVED because "
        "the existing test suite does not test the boundary condition.\n"
        "Your task is to suggest exactly ONE new pytest test function that catches this mutation.\n\n"
        "Core Rule:\n"
        "The original implementation defines expected behaviour for this mutation experiment. "
        "Choose inputs that distinguish the versions, then assert the ORIGINAL result. Never assert the mutant's result.\n\n"
        "The suggested test MUST satisfy the following requirements:\n"
        "1. It must PASS when run against the original function.\n"
        "2. It must FAIL when run against the mutated function.\n"
        f"3. It must import the target function using: {import_path}\n"
        "4. Critical assertion rule: Assert the expected return value or behavior of the ORIGINAL function. "
        "Under no circumstances should the assertion check for the mutated function's return value.\n\n"
        f"{BOUNDARY_ANALYSIS_CONCEPT}\n\n"
        "--- Target Function (Original - Defines Expected Behavior) ---\n"
        f"{original_function.strip()}\n\n"
        "--- Target Function (Mutated - Defective Version) ---\n"
        f"{mutated_function.strip()}\n\n"
        "--- Existing Tests ---\n"
        f"{existing_tests.strip()}\n\n"
        "Output Instructions:\n"
        "- Write exactly ONE standalone pytest test function.\n"
        f"- Include the required import statement: {import_path}\n"
        "- Provide the code inside a ```python ... ``` code block.\n"
        "- Do not include conversational commentary, explanations, or documentation outside the code block.\n"
    )


def build_retry_prompt(
    original_function: str,
    mutated_function: str,
    previous_test: str,
    verdict: str,
    diagnostics: str,
    import_path: str,
) -> str:
    """Construct the prompt sent to Ollama for revising a test suggestion.

    Supplies the original and mutated function, the previous suggested test,
    the structured verdict, and bounded failure diagnostics.
    """
    if verdict == "REJECTED":
        feedback_explanation = (
            "The previous test suggestion was REJECTED because it failed when executed against the original function.\n"
            "This typically occurs when the test asserts the MUTATED function's return value rather than the ORIGINAL "
            "function's return value, or chooses an invalid expected result.\n"
            "Remember: The original implementation defines expected behaviour for this mutation experiment. "
            "Choose inputs that distinguish the versions, then assert the ORIGINAL result. Never assert the mutant's result."
        )
    elif verdict == "NON_DISTINGUISHING":
        feedback_explanation = (
            "The previous test suggestion was NON-DISTINGUISHING because it passed on both the original and mutated function.\n"
            "The mutation still survived. The chosen test inputs did not probe the exact boundary condition where the "
            "original operator and the mutated operator diverge.\n"
            "Choose inputs that distinguish the versions, then assert the ORIGINAL result. Never assert the mutant's result."
        )
    else:
        feedback_explanation = (
            f"The previous test suggestion did not verify cleanly (verdict: {verdict}).\n"
            "Remember: The original implementation defines expected behaviour for this mutation experiment. "
            "Choose inputs that distinguish the versions, then assert the ORIGINAL result. Never assert the mutant's result."
        )

    return (
        "You are an expert automated testing assistant for Python and pytest.\n"
        "Your previous test suggestion was executed against both the original and mutated code but failed verification.\n"
        "Your task is to provide exactly ONE revised pytest test function that corrects the issue.\n\n"
        f"Verification Feedback:\n{feedback_explanation}\n\n"
        "--- Previous Suggested Test (Failed Verification) ---\n"
        f"{previous_test.strip()}\n\n"
        "--- Verification Diagnostics ---\n"
        f"{diagnostics.strip()}\n\n"
        "--- Target Function (Original - Defines Expected Behavior) ---\n"
        f"{original_function.strip()}\n\n"
        "--- Target Function (Mutated - Defective Version) ---\n"
        f"{mutated_function.strip()}\n\n"
        f"{BOUNDARY_ANALYSIS_CONCEPT}\n\n"
        "Revised Requirements:\n"
        "1. The original implementation defines expected behaviour for this mutation experiment.\n"
        "2. Choose inputs that distinguish the versions, then assert the ORIGINAL result. Never assert the mutant's result.\n"
        "3. It must PASS when run against the original function.\n"
        "4. It must FAIL when run against the mutated function.\n"
        f"5. It must import the target function using: {import_path}\n\n"
        "Output Instructions:\n"
        "- Write exactly ONE standalone pytest test function.\n"
        f"- Include the required import statement: {import_path}\n"
        "- Provide the code inside a ```python ... ``` code block.\n"
        "- Do not include conversational commentary, explanations, or documentation outside the code block.\n"
    )


def extract_failure_diagnostics(
    verdict: str,
    orig_detail: Any,
    mut_detail: Any,
    max_chars: int = 1500,
) -> str:
    """Extract compact, bounded failure diagnostics from the verification run."""
    lines: list[str] = [
        f"Verdict: {verdict}",
        f"Original Run Status: {getattr(orig_detail, 'status_label', 'UNKNOWN')}",
        f"Mutated Run Status: {getattr(mut_detail, 'status_label', 'UNKNOWN')}",
    ]

    if verdict == "REJECTED":
        stdout = getattr(orig_detail, "stdout", "") or ""
        stderr = getattr(orig_detail, "stderr", "") or ""
        combined = stdout + "\n" + stderr
        # Extract pytest failure lines (assertion errors)
        fail_lines = [
            l for l in combined.splitlines()
            if any(k in l for k in ("FAILURES", "assert", "AssertionError", "E   ", "FAILED", "Error"))
        ]
        if fail_lines:
            lines.append("Original Run Failure Diagnostic:")
            lines.extend(fail_lines[-12:])
        elif combined.strip():
            lines.append("Original Run Output:")
            lines.extend(combined.strip().splitlines()[-10:])
    elif verdict == "NON_DISTINGUISHING":
        lines.append(
            "Diagnostics: The suggested test passed on both original and mutated code. "
            "It failed to kill the mutant because the chosen input did not trigger a divergence."
        )

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [truncated diagnostics]"
    return result


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


def derive_import_path(file_path: str, func_name: str) -> tuple[bool, str, str]:
    """Derive the import statement for func_name based on project layout.

    Supports:
      - Flat modules: e.g. calc.py -> from calc import func_name
      - Regular packages: e.g. my_pkg/calc.py -> from my_pkg.calc import func_name
      - Package __init__.py: e.g. my_pkg/__init__.py -> from my_pkg import func_name
      - src layouts: e.g. src/my_pkg/calc.py -> from my_pkg.calc import func_name
                     e.g. src/calc.py -> from calc import func_name
                     e.g. src/my_pkg/__init__.py -> from my_pkg import func_name

    Returns:
      (is_supported, import_statement, reason_if_unsupported)
    """
    p = Path(file_path)
    parts = list(p.parts)
    if not parts or not p.name.endswith(".py"):
        return False, "", f"File '{file_path}' is not a Python source file."

    # Strip src prefix if present
    if parts[0] == "src":
        parts = parts[1:]
        if not parts:
            return False, "", "File is in src directory without a module name."

    # Handle __init__.py vs regular modules
    if parts[-1] == "__init__.py":
        package_parts = parts[:-1]
        if not package_parts:
            return False, "", "Root __init__.py is not inside a regular package."
        module_parts = list(package_parts)
    else:
        stem = parts[-1][:-3]
        module_parts = list(parts[:-1]) + [stem]

    # Verify all parts are valid Python identifiers
    for part in module_parts:
        if not part.isidentifier():
            return False, "", f"Path component '{part}' is not a valid Python identifier."

    module_name = ".".join(module_parts)
    import_path = f"from {module_name} import {func_name}"
    return True, import_path, ""


def find_enclosing_function(
    source: str,
    lineno: int,
) -> tuple[bool, ast.FunctionDef | None, str]:
    """Locate the top-level synchronous FunctionDef containing lineno.

    Reports unsupported cases clearly (async functions, class methods,
    nested functions, module-level expressions).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, None, f"Syntax error in source file: {exc}"

    # First, check if lineno is inside an async function or a class definition
    for node in ast.walk(tree):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            end_line = node.end_lineno if node.end_lineno is not None else node.lineno
            if node.lineno <= lineno <= end_line:
                if isinstance(node, ast.AsyncFunctionDef):
                    return False, None, f"Async functions ('{node.name}') are not supported for AI test suggestions."
                if isinstance(node, ast.ClassDef):
                    for sub in ast.walk(node):
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sub_end = sub.end_lineno if sub.end_lineno is not None else sub.lineno
                            if sub.lineno <= lineno <= sub_end:
                                return False, None, f"Class methods ('{node.name}.{sub.name}') are not supported. Only synchronous top-level functions are supported."
                    return False, None, f"Mutant is inside class '{node.name}'. Only synchronous top-level functions are supported."

    # Next, search for top-level FunctionDef in tree.body
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            stmt_end = stmt.end_lineno if stmt.end_lineno is not None else stmt.lineno
            if stmt.lineno <= lineno <= stmt_end:
                # Check for nested functions inside stmt
                for child in ast.walk(stmt):
                    if child is not stmt and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        child_end = child.end_lineno if child.end_lineno is not None else child.lineno
                        if child.lineno <= lineno <= child_end:
                            return False, None, f"Nested functions ('{child.name}' inside '{stmt.name}') are not supported. Only synchronous top-level functions are supported."
                return True, stmt, ""

    return False, None, "Mutant is at module level (not inside a top-level function). Only synchronous top-level functions are supported."


def extract_relevant_tests(
    snapshot_dir: Path,
    target_func_name: str,
    target_file_path: str,
    max_chars: int = 3000,
) -> str:
    """Extract relevant existing tests from snapshot_dir with a strict size bound.

    Searches for test functions referencing target_func_name or matching test files.
    """
    from ghost_pr.git_utils import is_test_file

    test_files: list[Path] = []
    for p in snapshot_dir.rglob("*.py"):
        rel_str = str(p.relative_to(snapshot_dir))
        if is_test_file(rel_str):
            test_files.append(p)

    matched_tests: list[str] = []
    name_matched_tests: list[str] = []
    other_tests: list[str] = []

    target_stem = Path(target_file_path).stem

    for tf in test_files:
        try:
            content = tf.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except Exception:
            continue

        is_name_match = target_stem in tf.stem

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                fn_text = ast.unparse(node)
                if target_func_name in fn_text:
                    matched_tests.append(fn_text)
                elif is_name_match:
                    name_matched_tests.append(fn_text)
                else:
                    other_tests.append(fn_text)

    selected = matched_tests or name_matched_tests or other_tests
    if not selected:
        return "# [No existing tests found targeting this function]"

    result_parts: list[str] = []
    total_len = 0
    for t in selected:
        if total_len + len(t) + 2 > max_chars:
            result_parts.append("# [Remaining existing tests omitted to bound prompt size]")
            break
        result_parts.append(t)
        total_len += len(t) + 2

    return "\n\n".join(result_parts)


def analyze_mutation_target(
    snapshot_dir: Path,
    candidate: MutationCandidate,
    max_test_chars: int = 3000,
) -> tuple[bool, TargetFunctionInfo | None, str]:
    """Analyze a surviving candidate mutant to determine if it is a supported target.

    Extracts original function, mutated function, relevant tests, and import path.
    """
    target_path = snapshot_dir / candidate.file_path
    if not target_path.is_file():
        return False, None, f"Target file '{candidate.file_path}' does not exist in snapshot."

    source = target_path.read_text(encoding="utf-8")

    # 1. Find enclosing top-level function
    supported_fn, func_node, fn_reason = find_enclosing_function(source, candidate.lineno)
    if not supported_fn or func_node is None:
        return False, None, fn_reason

    func_name = func_node.name

    # 2. Derive import path from project layout
    supported_layout, import_path, layout_reason = derive_import_path(candidate.file_path, func_name)
    if not supported_layout:
        return False, None, layout_reason

    # 3. Extract original function AST
    orig_fn_code = ast.unparse(func_node)

    # 4. Generate mutated source and extract mutated function AST
    mutated_source = apply_mutation_to_source(source, candidate)
    try:
        mut_tree = ast.parse(mutated_source)
        mut_func_node = None
        for stmt in mut_tree.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == func_name:
                mut_func_node = stmt
                break
        if mut_func_node is None:
            mut_fn_code = orig_fn_code
        else:
            mut_fn_code = ast.unparse(mut_func_node)
    except SyntaxError:
        mut_fn_code = orig_fn_code

    # 5. Extract relevant tests from snapshot
    existing_tests = extract_relevant_tests(
        snapshot_dir=snapshot_dir,
        target_func_name=func_name,
        target_file_path=candidate.file_path,
        max_chars=max_test_chars,
    )

    return True, TargetFunctionInfo(
        func_name=func_name,
        import_path=import_path,
        original_func=orig_fn_code,
        mutated_func=mut_fn_code,
        existing_tests=existing_tests,
    ), ""


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


def _send_ollama_request(
    prompt: str,
    import_path: str,
    *,
    model: str | None = None,
    host: str | None = None,
    timeout: int | None = None,
) -> SuggestionResult:
    """Send a prompt to local Ollama and return structured SuggestionResult."""
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
    prompt = build_suggestion_prompt(
        original_function=original_function,
        mutated_function=mutated_function,
        existing_tests=existing_tests,
        import_path=import_path,
    )
    return _send_ollama_request(
        prompt=prompt,
        import_path=import_path,
        model=model,
        host=host,
        timeout=timeout,
    )


def request_revised_test_suggestion(
    original_function: str,
    mutated_function: str,
    previous_test: str,
    verdict: str,
    diagnostics: str,
    import_path: str,
    *,
    model: str | None = None,
    host: str | None = None,
    timeout: int | None = None,
) -> SuggestionResult:
    """Request a revised pytest test suggestion from local Ollama after failed verification.

    Parameters
    ----------
    original_function : str
        Source of the original unmutated function.
    mutated_function : str
        Source of the mutated function.
    previous_test : str
        Source of the previously rejected or non-distinguishing test.
    verdict : str
        The structured verdict of the previous test ('REJECTED' or 'NON_DISTINGUISHING').
    diagnostics : str
        Compact failure diagnostics from the actual test runs.
    import_path : str
        The import statement to include in the generated test.
    model : str, optional
        Ollama model name (defaults to env var or 'qwen2.5-coder:7b').
    host : str, optional
        Ollama host URL (defaults to env var or 'http://localhost:11434').
    timeout : int, optional
        Request timeout in seconds (defaults to env var or 30).
    """
    prompt = build_retry_prompt(
        original_function=original_function,
        mutated_function=mutated_function,
        previous_test=previous_test,
        verdict=verdict,
        diagnostics=diagnostics,
        import_path=import_path,
    )
    return _send_ollama_request(
        prompt=prompt,
        import_path=import_path,
        model=model,
        host=host,
        timeout=timeout,
    )


def display_suggestion(
    result: SuggestionResult,
    *,
    is_retry: bool = False,
    auto_yes: bool = False,
) -> None:
    """Print the AI-suggested test for user review.

    Generated code is displayed for review only and is never automatically
    added to the project files.
    """
    banner_title = (
        "  💡  AI-Suggested Test (Revised)"
        if is_retry
        else "  💡  AI-Suggested Test"
    )
    print(f"\n{'─' * 60}")
    print(banner_title)
    print(f"{'─' * 60}")
    print(f"  Model  : {result.model}")

    if result.error:
        print(f"\n  ⚠️   Could not generate AI test suggestion:")
        print(f"      {result.error}")
        print("      (Mutation testing results above were not affected)\n")
        return

    print("\n  This test has not been added to your project.")
    if auto_yes:
        print("  Checking this test against original and mutated code…\n")
    else:
        print("  Not yet executed.\n")

    lines = (result.test_code or "").splitlines()
    max_len = max([len(line) for line in lines] + [56])
    border = "─" * (max_len + 4)
    print(f"  ┌{border}┐")
    for line in lines:
        print(f"  │  {line:<{max_len}}  │")
    print(f"  └{border}┘\n")
