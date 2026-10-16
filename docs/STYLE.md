# Code style

Plain, readable Python. The code should explain itself; comments are the exception.

## Comments and docstrings

- Default to no comment. Prefer a better name or a small function over a comment.
- Comment only what the code cannot say: a non-obvious reason, a constraint, a workaround,
  or a reference (`# CUAD's rule: min(left, right) + 0.01 * length`). One line, two at most.
- Docstrings: a one-line summary on public modules, classes and functions. Add arguments or
  return values only when the types and names do not make them clear.
- No narrative docstrings, measurement logs, dates, run numbers or design history in code.
  Those belong in `PLAN.md`, `IMPLEMENTATION.md` or the thesis.
- No banner comments (`# --- section ---`), no all-caps headings, no emoji, no
  "Note:" / "Important:" / "Why this exists".
- No commented-out code. No `TODO` without a concrete next step.

## Code

- Type hints on public functions. `dataclass` or Pydantic models for structured data.
- Small functions with one job. Early returns over nested conditionals.
- Only the options that are used. No speculative parameters, flags or abstraction layers.
- Specific exceptions with short, factual messages.
- Module-level constants in `UPPER_CASE`. No magic numbers in function bodies.
- Library code uses `logging`; only `cli.py` prints.
- Formatting and linting: `ruff format` and `ruff check` must be clean.

## Tests

- `pytest`, one behaviour per test, descriptive test names instead of comments.
- Slow tests (model download, inference) are marked `@pytest.mark.slow`.

## Commits

- Imperative subject line, at most ~60 characters (`Add page mapping for PDF input`).
- A body only when the reason is not obvious from the diff; a few lines at most.
