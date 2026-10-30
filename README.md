# Ghost-PR

**Mutation testing CLI that finds gaps in your test suite.**

Ghost-PR applies small, targeted mutations to your source code and re-runs
your tests. If the tests still pass after a mutation, it means they didn't
cover that behaviour — a *surviving mutant* highlights a real gap.

---

## Installation

```bash
# Clone and install in editable mode (requires Python ≥ 3.10)
git clone <repo-url> && cd Ghost-PR
pip install -e .
```

---

## Usage

### 1. Haunt Staged Changes (Milestone 2)

Test your staged Git changes before committing:

```bash
git add <files-you-changed>
ghost haunt
```

Ghost-PR will:
1. Locate the current Git repository root.
2. Check for unstaged changes to tracked files. If any exist, it stops and prompts you to stage or resolve them.
3. Exclude untracked files and report them.
4. Export a pristine project snapshot directly from Git's index (the staged versions, including all unchanged tracked files and test suites).
5. Run baseline tests on the snapshot; if baseline tests do not pass, it halts immediately.
6. Locate single-line `<=` comparisons on the changed lines of staged Python source files.
7. Run up to 3 isolated mutants, each starting from a pristine snapshot.
8. Report the file, line number, original expression, mutated expression, and outcome (`SURVIVED`, `KILLED`, or `INCONCLUSIVE`).
9. Completely preserve your working files and Git index.

### 2. Demo Mode (Milestone 1)

Run the self-contained demo fixture without needing a Git repository:

```bash
ghost haunt --demo
```

This runs mutation testing against a bundled `withdraw` function with a weak boundary test to demonstrate a surviving mutant.

---

## Supported Project Layouts

Ghost-PR supports standard Python project layouts managed with Git:

1. **Flat Layout:**
   ```
   repo/
     module.py
     test_module.py
   ```
2. **Standard Package Layout:**
   ```
   repo/
     mypackage/
       __init__.py
       logic.py
     tests/
       test_logic.py
     pyproject.toml
   ```
3. **`src` Layout:**
   ```
   repo/
     src/
       mypackage/
         __init__.py
         logic.py
     tests/
       test_logic.py
     pyproject.toml
   ```

`ghost haunt` automatically prepends the temporary workspace (and `src/` if present) to `PYTHONPATH`. For standard flat and `src` layouts, test subprocesses import code directly from the temporary workspace.

> **Note on Editable Installs:** Prepending `PYTHONPATH` isolates standard uninstalled module lookups. However, `PYTHONPATH` does *not* universally guarantee isolation from all editable installations (such as tools using PEP 660 meta-path hooks). Isolated virtualenv execution is planned for future milestones.

---

## Limitations (Milestone 2)

- **Mutation Operator:** Currently targets only `<=` comparisons, transforming them to `<`.
- **Changed Lines Only:** Mutations are applied strictly to lines added or modified in the staged diff relative to `HEAD`.
- **Single-Line Comparisons:** Multiline comparisons (where the comparison expression spans across multiple lines) are explicitly skipped.
- **Test Files Excluded:** Test files (`test_*.py`, `*_test.py`, files in `tests/` or `test/`, and `conftest.py`) are never mutated.
- **Symlinks & Submodules:** Symlinks and Git submodules are explicitly skipped.
- **Unstaged Changes:** To prevent testing mismatched state, Ghost-PR requires all changes to tracked files to be staged before running.
- **Untracked Files:** Untracked files are excluded from the test snapshot.
- **Mutant Cap:** Runs at most 3 mutants per invocation to keep test cycles fast.
- **No AI Yet:** AI-generated test suggestions are planned for future milestones.

---

## Running the Test Suite

```bash
pip install -e .
pytest tests/ -v
```

---

## License

[MIT](LICENSE)