"""Git utilities for Ghost-PR.

This module provides helpers to:
  - Find the Git repository root.
  - List staged Python files (diff --cached).
  - Detect unstaged changes to tracked files.
  - Detect untracked files.
  - Build a temporary project snapshot from Git's index.
  - Parse diff hunks to identify changed lines.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StagedFileInfo:
    """Metadata about a staged file relative to HEAD."""
    path: str
    status: str
    is_symlink: bool = False
    is_submodule: bool = False
    is_test: bool = False
    is_python: bool = False
    changed_lines: set[int] = field(default_factory=set)


def find_repo_root(start_dir: Path | str | None = None) -> Path | None:
    """Find the root directory of the current Git repository."""
    cwd = Path(start_dir).resolve() if start_dir else Path.cwd()
    res = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return None
    return Path(res.stdout.strip()).resolve()


def get_unstaged_tracked_files(repo_root: Path) -> list[str]:
    """Return relative paths of tracked files that have unstaged modifications/deletions."""
    res = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return []
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    return sorted(lines)


def get_untracked_files(repo_root: Path) -> list[str]:
    """Return relative paths of untracked files."""
    res = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return []
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    return sorted(lines)


def is_test_file(path_str: str) -> bool:
    """Check if a file path belongs to tests (e.g. tests/, test_*.py, *_test.py, conftest.py)."""
    p = Path(path_str)
    parts = p.parts
    # Any directory component is 'tests' or 'test'
    if any(part in ("tests", "test") for part in parts[:-1]):
        return True
    name = p.name
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    if name == "conftest.py":
        return True
    return False


def parse_diff_hunks_for_new_lines(diff_text: str) -> set[int]:
    """Parse unified diff text with -U0 to find 1-indexed lines added or modified."""
    changed: set[int] = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff_text, re.MULTILINE):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count > 0:
            changed.update(range(start, start + count))
    return changed


def get_staged_entries(repo_root: Path) -> list[StagedFileInfo]:
    """Inspect Git staging area and return list of staged files with changed line info."""
    head_check = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    head_exists = (head_check.returncode == 0)

    cmd = ["git", "diff", "--cached", "--name-status"]
    if head_exists:
        cmd.append("HEAD")
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if res.returncode != 0:
        return []

    entries: list[StagedFileInfo] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        status_code = parts[0]
        if len(parts) == 3 and status_code.startswith(("R", "C")):
            file_path = parts[2]
        else:
            file_path = parts[1]

        if status_code.startswith("D"):
            continue

        ls_res = subprocess.run(
            ["git", "ls-files", "-s", "--", file_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        mode = ""
        if ls_res.returncode == 0 and ls_res.stdout.strip():
            mode = ls_res.stdout.split()[0]

        is_symlink = mode.startswith("120") or (repo_root / file_path).is_symlink()
        is_submodule = mode.startswith("160")
        is_python = file_path.endswith(".py")
        is_test = is_test_file(file_path)

        changed_lines: set[int] = set()
        if is_python and not is_symlink and not is_submodule:
            diff_cmd = ["git", "diff", "--cached", "-U0"]
            if head_exists:
                diff_cmd.append("HEAD")
            diff_cmd.extend(["--", file_path])
            diff_res = subprocess.run(diff_cmd, cwd=repo_root, capture_output=True, text=True)
            if diff_res.returncode == 0:
                changed_lines = parse_diff_hunks_for_new_lines(diff_res.stdout)

        entries.append(
            StagedFileInfo(
                path=file_path,
                status=status_code,
                is_symlink=is_symlink,
                is_submodule=is_submodule,
                is_test=is_test,
                is_python=is_python,
                changed_lines=changed_lines,
            )
        )

    return entries


def build_index_snapshot(repo_root: Path, target_dir: Path) -> None:
    """Build a complete project snapshot from Git's index into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(target_dir.resolve())
    if not prefix.endswith(os.sep):
        prefix += os.sep
    res = subprocess.run(
        ["git", "checkout-index", "-a", "-f", f"--prefix={prefix}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Failed to build snapshot from Git index: {res.stderr.strip()}")
