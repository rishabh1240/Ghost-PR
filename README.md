# Ghost-PR

**Mutation testing CLI that finds gaps in your test suite.**

Ghost-PR applies small, targeted mutations to your source code and re-runs
your tests. If the tests still pass after a mutation, it means they didn't
cover that behaviour — a *surviving mutant* highlights a real gap.

> **Milestone 1** — Demo-only, single mutation operator (`<=` → `<`).
> No AI, no GitHub integration, no dashboard.

---

## Installation

```bash
# Clone and install in editable mode (requires Python ≥ 3.10)
git clone <repo-url> && cd Ghost-PR
pip install -e .
```

## Quick Start — Demo

```bash
ghost haunt --demo
```

This command:

1. Copies a bundled demo fixture (`withdraw` function + one weak test)
   into a temporary workspace.
2. Runs the existing tests to confirm the baseline passes.
3. Uses Python's `ast` module to mutate `amount <= balance` → `amount < balance`.
4. Re-runs the tests against the mutant.
5. Reports that the mutant **survived** (the weak test doesn't catch the change).
6. Cleans up the temporary workspace — the original fixture is never modified.

### Expected output (abbreviated)

```
  👻  Ghost-PR  ·  haunt --demo
  Step 1 · Running baseline tests          ✅  PASSED
  Step 2 · Applying AST mutation            <= → <
  Step 3 · Running tests against the mutant ✅  PASSED
  Result  🧟  MUTANT SURVIVED
```

## Running the Tests

```bash
pip install -e .
pytest tests/ -v
```

## How It Works

| Step | What happens |
|------|-------------|
| Copy | Demo fixtures are copied to a `tempfile.mkdtemp()` workspace |
| Baseline | `pytest` runs via `subprocess` using `sys.executable` |
| Mutate | `ast.NodeTransformer` rewrites the first `<=` → `<` |
| Retest | `pytest` re-runs against the mutated source |
| Classify | Exit codes map to PASSED / FAILED / COLLECTION_ERROR / TIMEOUT / … |
| Cleanup | `shutil.rmtree` removes the temp workspace |

## Limitations (Milestone 1)

- Only one mutation operator: `<=` → `<`.
- Only the bundled demo fixture; no user-project support yet.
- No AI test suggestions (planned for Milestone 2).
- Temporary directory is not a security sandbox — only trusted code.

## License

[MIT](LICENSE)

---

*Built at a hackathon. AI assistant used: **Claude Opus 4.6** (Anthropic, via Antigravity IDE).*