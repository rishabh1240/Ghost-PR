"""Tests for Milestone 2: ghost haunt on staged Python changes.

Verifies:
  - Repository root detection
  - Unstaged-change rejection
  - Untracked-file exclusion and reporting
  - Staged-file detection (including newly staged files)
  - Unchanged-line exclusion
  - Surviving mutants
  - Killed mutants
  - Failing baseline aborts
  - Original working file and Git index preservation
  - Multiline comparison skipping
  - Test file exclusion from mutation
  - At most three mutants execution limit
  - Snapshot import resolution
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ghost_pr.engine import (
    MutationCandidate,
    apply_mutation_to_source,
    find_mutation_candidates,
)
from ghost_pr.git_utils import (
    build_index_snapshot,
    find_repo_root,
    get_staged_entries,
    get_unstaged_tracked_files,
    get_untracked_files,
    is_test_file,
    parse_diff_hunks_for_new_lines,
)
from ghost_pr.haunt import haunt_staged_changes


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo with required user config."""
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)


def test_find_repo_root(tmp_path: Path):
    """find_repo_root discovers git root from subdirectories and returns None outside."""
    outside = tmp_path / "outside"
    outside.mkdir()
    assert find_repo_root(outside) is None

    repo = tmp_path / "my_repo"
    repo.mkdir()
    _init_git_repo(repo)
    sub = repo / "sub" / "pkg"
    sub.mkdir(parents=True)

    assert find_repo_root(repo) == repo.resolve()
    assert find_repo_root(sub) == repo.resolve()


def test_unstaged_changes_rejected(tmp_path: Path, capsys):
    """Tracked files with unstaged changes are rejected with exit code 1."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    f = repo / "module.py"
    f.write_text("def foo():\n    return 1\n")
    subprocess.run(["git", "add", "module.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

    # Modify file without staging
    f.write_text("def foo():\n    return 2\n")

    code = haunt_staged_changes(repo)
    assert code == 1
    out = capsys.readouterr().out
    assert "Unstaged changes detected in tracked file(s)" in out
    assert "module.py" in out


def test_untracked_files_excluded_and_reported(tmp_path: Path, capsys):
    """Untracked files are reported as excluded and omitted from the snapshot."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    # Staged code and test
    code_file = repo / "bank.py"
    code_file.write_text("def withdraw(b, a):\n    if a <= b:\n        return b - a\n    raise ValueError()\n")
    test_file = repo / "test_bank.py"
    test_file.write_text("from bank import withdraw\ndef test_w():\n    assert withdraw(100, 40) == 60\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    # Untracked file
    untracked = repo / "notes.txt"
    untracked.write_text("secret notes\n")

    code = haunt_staged_changes(repo)
    assert code == 0
    out = capsys.readouterr().out
    assert "Excluded untracked file(s) from snapshot:" in out
    assert "notes.txt" in out


def test_no_staged_changes(tmp_path: Path, capsys):
    """Clean repo with no staged changes cleanly exits with 0 and informative message."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    f = repo / "init.txt"
    f.write_text("hello\n")
    subprocess.run(["git", "add", "init.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=repo, check=True)

    code = haunt_staged_changes(repo)
    assert code == 0
    out = capsys.readouterr().out
    assert "No staged changes found relative to HEAD" in out


def test_unchanged_line_exclusion(tmp_path: Path, capsys):
    """Only <= comparisons on staged changed lines are mutated. Unchanged lines are excluded."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    # Initial commit with an existing <= comparison on line 3
    orig_code = (
        "def unchanged_op(x, y):\n"
        "    if x <= y:\n"       # Line 2 (unchanged)
        "        return True\n"
        "    return False\n"
    )
    (repo / "calc.py").write_text(orig_code)
    (repo / "test_calc.py").write_text(
        "from calc import unchanged_op, new_op\n"
        "def test_new():\n"
        "    assert new_op(5, 10) == 1\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "commit 1"], cwd=repo, check=True)

    # Add a new function with <= on a new line and stage it
    updated_code = (
        "def unchanged_op(x, y):\n"
        "    if x <= y:\n"
        "        return True\n"
        "    return False\n\n"
        "def new_op(a, b):\n"
        "    if a <= b:\n"       # Line 7 (changed)
        "        return 1\n"
        "    return 0\n"
    )
    (repo / "calc.py").write_text(updated_code)
    subprocess.run(["git", "add", "calc.py"], cwd=repo, check=True)

    code = haunt_staged_changes(repo)
    assert code == 0
    out = capsys.readouterr().out
    # Mutant on line 7 was tested
    assert "calc.py:7" in out
    # Mutant on line 2 was NOT tested
    assert "calc.py:2" not in out


def test_surviving_mutant_staged(tmp_path: Path, capsys):
    """A mutant surviving weak tests is reported as SURVIVED."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "bank.py").write_text(
        "def withdraw(b, a):\n"
        "    if a <= b:\n"
        "        return b - a\n"
        "    raise ValueError()\n"
    )
    (repo / "test_bank.py").write_text(
        "from bank import withdraw\n"
        "def test_w():\n"
        "    assert withdraw(100, 40) == 60\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    code = haunt_staged_changes(repo)
    assert code == 0
    out = capsys.readouterr().out
    assert "MUTANT SURVIVED" in out
    assert "bank.py:2" in out
    assert "a <= b" in out
    assert "a < b" in out


def test_killed_mutant_staged(tmp_path: Path, capsys):
    """A mutant killed by boundary tests is reported as KILLED."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "bank.py").write_text(
        "def withdraw(b, a):\n"
        "    if a <= b:\n"
        "        return b - a\n"
        "    raise ValueError()\n"
    )
    (repo / "test_bank.py").write_text(
        "from bank import withdraw\n"
        "def test_w():\n"
        "    assert withdraw(100, 40) == 60\n"
        "def test_boundary():\n"
        "    assert withdraw(100, 100) == 0\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    code = haunt_staged_changes(repo)
    assert code == 0
    out = capsys.readouterr().out
    assert "MUTANT KILLED" in out
    assert "bank.py:2" in out


def test_failing_baseline_aborts(tmp_path: Path, capsys):
    """Failing baseline tests abort mutation testing immediately with exit code 1."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "code.py").write_text(
        "def check(x):\n"
        "    if x <= 10:\n"
        "        return True\n"
        "    return False\n"
    )
    (repo / "test_code.py").write_text(
        "from code import check\n"
        "def test_broken():\n"
        "    assert check(5) == 999\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    code = haunt_staged_changes(repo)
    assert code == 1
    out = capsys.readouterr().out
    assert "Baseline tests did not pass on the staged snapshot — aborting mutation testing." in out


def test_original_file_and_git_index_preserved(tmp_path: Path):
    """Working files and Git index remain identical after haunt completes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "bank.py").write_text(
        "def withdraw(b, a):\n"
        "    if a <= b:\n"
        "        return b - a\n"
        "    raise ValueError()\n"
    )
    (repo / "test_bank.py").write_text(
        "from bank import withdraw\n"
        "def test_w():\n"
        "    assert withdraw(100, 40) == 60\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    before_content = (repo / "bank.py").read_text()
    before_diff = subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True).stdout

    code = haunt_staged_changes(repo)
    assert code == 0

    after_content = (repo / "bank.py").read_text()
    after_diff = subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True).stdout

    assert before_content == after_content
    assert before_diff == after_diff


def test_multiline_comparison_skipped(tmp_path: Path, capsys):
    """Multiline comparisons spanning multiple lines are explicitly skipped."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    code = (
        "def check(x, y):\n"
        "    if (\n"
        "        x <=\n"
        "        y\n"
        "    ):\n"
        "        return True\n"
        "    return False\n"
    )
    (repo / "multi.py").write_text(code)
    (repo / "test_multi.py").write_text("from multi import check\ndef test_m(): assert check(1, 2) is True\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    code_exit = haunt_staged_changes(repo)
    assert code_exit == 0
    out = capsys.readouterr().out
    assert "Skipping multiline comparison at multi.py:3" in out
    assert "No supported mutations found in staged changes." in out


def test_test_files_not_mutated(tmp_path: Path, capsys):
    """Test files containing <= are not mutated."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "test_alone.py").write_text(
        "def test_cmp():\n"
        "    val = 5\n"
        "    assert val <= 10\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    code = haunt_staged_changes(repo)
    assert code == 0
    out = capsys.readouterr().out
    assert "Skipping staged file 'test_alone.py': test files are not mutated." in out
    assert "No supported mutations found in staged changes." in out


def test_at_most_three_mutants(tmp_path: Path, capsys):
    """When more than 3 candidates exist on changed lines, only 3 mutants are run."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    code = (
        "def f(a, b, c, d, e):\n"
        "    r1 = a <= 1\n"
        "    r2 = b <= 2\n"
        "    r3 = c <= 3\n"
        "    r4 = d <= 4\n"
        "    r5 = e <= 5\n"
        "    return r1 and r2 and r3 and r4 and r5\n"
    )
    (repo / "many.py").write_text(code)
    (repo / "test_many.py").write_text("from many import f\ndef test_f(): assert f(0,0,0,0,0) is True\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    exit_code = haunt_staged_changes(repo)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Found 5 mutation candidate(s). Running at most 3 mutants." in out
    assert "Mutant 1/3" in out
    assert "Mutant 2/3" in out
    assert "Mutant 3/3" in out
    assert "Mutant 4/3" not in out


def test_symlink_skipped(tmp_path: Path, capsys):
    """Symlinks are explicitly skipped with a clear message."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    target = repo / "target.py"
    target.write_text("x = 1\n")
    subprocess.run(["git", "add", "target.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

    link = repo / "link.py"
    os.symlink("target.py", link)
    subprocess.run(["git", "add", "link.py"], cwd=repo, check=True)

    exit_code = haunt_staged_changes(repo)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Skipping staged file 'link.py': symlinks are not supported." in out


def test_snapshot_includes_unchanged_tracked_files(tmp_path: Path):
    """Temporary snapshot includes unchanged tracked files from index."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "shared_config.txt").write_text("config_val=42\n")
    (repo / "source.py").write_text("def val(): return 42\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "commit 1"], cwd=repo, check=True)

    # Now stage a new file
    (repo / "staged.py").write_text("def is_ok(x): return x <= 10\n")
    subprocess.run(["git", "add", "staged.py"], cwd=repo, check=True)

    snapshot_dir = tmp_path / "snapshot"
    build_index_snapshot(repo, snapshot_dir)

    assert (snapshot_dir / "shared_config.txt").is_file()
    assert (snapshot_dir / "source.py").is_file()
    assert (snapshot_dir / "staged.py").is_file()
    assert (snapshot_dir / "shared_config.txt").read_text() == "config_val=42\n"


def test_tests_import_from_snapshot_flat_layout(tmp_path: Path):
    """Tests executing in the temporary snapshot import code from the snapshot in flat layouts."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "calc.py").write_text(
        "def check(x):\n"
        "    if x <= 5:\n"
        "        return True\n"
        "    return False\n"
    )
    repo_str = str(repo.resolve())
    (repo / "test_calc.py").write_text(
        f"import calc\n"
        f"def test_imported_snapshot_path():\n"
        f"    # Must NOT import from the original working repo\n"
        f"    assert not calc.__file__.startswith({repo_str!r})\n"
        f"    assert 'ghost_pr_' in calc.__file__\n"
        f"    assert calc.check(3) is True\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    exit_code = haunt_staged_changes(repo)
    assert exit_code == 0


def test_tests_import_from_snapshot_src_layout(tmp_path: Path):
    """Tests executing in the temporary snapshot import code from src/ in src layouts."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    pkg_dir = repo / "src" / "mypkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "logic.py").write_text(
        "def check(x):\n"
        "    if x <= 10:\n"
        "        return True\n"
        "    return False\n"
    )

    tests_dir = repo / "tests"
    tests_dir.mkdir()
    repo_str = str(repo.resolve())
    (tests_dir / "test_logic.py").write_text(
        f"from mypkg import logic\n"
        f"def test_src_import():\n"
        f"    assert not logic.__file__.startswith({repo_str!r})\n"
        f"    assert 'ghost_pr_' in logic.__file__\n"
        f"    assert logic.check(5) is True\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    exit_code = haunt_staged_changes(repo)
    assert exit_code == 0


def test_bytecode_cache_not_inherited_and_writes_disabled(tmp_path: Path):
    """Test subprocesses run with -B and PYTHONDONTWRITEBYTECODE=1, and mutant snapshots have no pyc files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "math_mod.py").write_text(
        "def compute(n):\n"
        "    if n <= 10:\n"
        "        return n * 2\n"
        "    return n\n"
    )
    (repo / "test_math_mod.py").write_text(
        "import sys, os, pathlib\n"
        "import math_mod\n"
        "def test_bytecode_settings():\n"
        "    # Bytecode writes must be disabled\n"
        "    assert sys.dont_write_bytecode is True\n"
        "    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'\n"
        "    # No .pyc files should exist in the workspace\n"
        "    root = pathlib.Path(math_mod.__file__).parent\n"
        "    pycs = list(root.glob('**/*.pyc'))\n"
        "    assert len(pycs) == 0, f'Found unexpected pyc files: {pycs}'\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    exit_code = haunt_staged_changes(repo)
    assert exit_code == 0


def test_chained_comparison_single_operator_mutation(tmp_path: Path, capsys):
    """Chained comparison 0 <= amount <= balance produces separate single-operator mutants."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "wallet.py").write_text(
        "def valid_transfer(amount, balance):\n"
        "    if 0 <= amount <= balance:\n"
        "        return True\n"
        "    return False\n"
    )
    # Test catches the lower boundary (amount < 0) but not upper boundary (amount == balance)
    (repo / "test_wallet.py").write_text(
        "from wallet import valid_transfer\n"
        "def test_lower_boundary():\n"
        "    # amount=0 is valid; if mutated to 0 < amount, this fails (KILLED)\n"
        "    assert valid_transfer(0, 100) is True\n"
        "def test_nominal():\n"
        "    assert valid_transfer(50, 100) is True\n"
        "    assert valid_transfer(-1, 100) is False\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    exit_code = haunt_staged_changes(repo)
    assert exit_code == 0
    out = capsys.readouterr().out

    # Both mutants are identified on line 2
    assert "wallet.py:2" in out
    # Mutant 1 changes ONLY the first operator
    assert "0 < amount <= balance" in out
    # Mutant 2 changes ONLY the second operator
    assert "0 <= amount < balance" in out

    # The lower-boundary mutant is KILLED by test_lower_boundary
    # The upper-boundary mutant SURVIVES because amount=balance is not asserted
    assert "MUTANT KILLED" in out
    assert "MUTANT SURVIVED" in out


def test_initial_commit_newly_staged_files(tmp_path: Path, capsys):
    """Works cleanly on a fresh repository before any commits exist (empty HEAD)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "math_ops.py").write_text(
        "def is_small(x):\n"
        "    if x <= 10:\n"
        "        return True\n"
        "    return False\n"
    )
    (repo / "test_math_ops.py").write_text(
        "from math_ops import is_small\n"
        "def test_small():\n"
        "    assert is_small(5) is True\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    code = haunt_staged_changes(repo)
    assert code == 0
    out = capsys.readouterr().out
    assert "math_ops.py:2" in out
    assert "MUTANT SURVIVED" in out


def test_cli_dispatch(tmp_path: Path):
    """Test CLI invocation using main() with --repo and --demo."""
    from ghost_pr.cli import main

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "bank.py").write_text(
        "def withdraw(b, a):\n"
        "    if a <= b:\n"
        "        return b - a\n"
        "    raise ValueError()\n"
    )
    (repo / "test_bank.py").write_text(
        "from bank import withdraw\n"
        "def test_w():\n"
        "    assert withdraw(100, 40) == 60\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    with pytest.raises(SystemExit) as exc:
        main(["haunt", "--repo", str(repo)])
    assert exc.value.code == 0

    with pytest.raises(SystemExit) as exc_demo:
        main(["haunt", "--demo"])
    assert exc_demo.value.code == 0
