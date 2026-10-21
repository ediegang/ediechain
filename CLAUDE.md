# Agent Instructions

This file provides guidance for agent when working with code in this repository.

## Project Overview

**ediechain** is the Brno research team's ("ediegang" Seznam's R&D team) repository.

**Project structure:**
- `CLAUDE.md` — this file.
- `README.md` — what the package is, how to install it, where to start.
- `TODO.md` — what is left to do. Nothing else.
- `src/ediechain/` — the package. `core/` is the engine; `handlers/`, `encoders/` and
  `coordinators/` are the backends, imported on first use.
- `tests/` — the specification.
- `examples/simple.py`, `examples/demo/` — one file, and a repository laid out the right way.
- `docs/glossary.md` — every term used in `src/`.
- `docs/ediechain_vs_taskchain.md` — the deck for people coming from taskchain.
- `docs/dev/design_notes.md` — the decisions, the reasoning, the traps.
- `docs/dev/taskchain_evaluation.md` — the review the rewrite came from.
- `playground/` — local copies of the libraries being replaced. Gitignored.

**Working on it:**
- Check `playground/taskchain/` before designing. Half the decisions are "taskchain did X, here
  is why not", and three times the answer was "taskchain did not solve this at all".
- Check `docs/dev/design_notes.md` before reversing something. Decision 19 exists because a
  conflict with an earlier decision was surfaced instead of quietly resolved.
- Run every claim. Prototype in a scratch directory, paste the output, then decide.
- A failing new test file is finished work, not broken work. Say which tests cover what, and in
  which order to make them green.
- Artifacts that already exist do not matter. A change that alters every key, or orphans stored
  data, costs nothing during development of this library. Do not count it as a cost.

## Package Development (run inside sandbox or with uv active)

**UV** — package and environment manager.

External dependencies (`seznam-pypi`, `seznam-pypi-staging`) require access to Seznam's internal PyPI.

```bash
uv run ruff check ./src ./scripts ./tests    # Lint; CI checks ./src only
uv run ruff check --fix ./src                # Lint and auto-fix
uv run python3 -m pytest tests               # Tests
```

## Publishing

`makefile` contains a `publish` command that automates the publishing process to Seznam's internal
PyPI repository (`seznam-pypi-staging`):

Before publishing, ensure that the `version` field in `pyproject.toml` has been updated to a new
version number.

Also there is a GitLab CI job configured to publish the package when a version change is detected in
`pyproject.toml`.


## Python Package (`src/ediechain/`)

The implementation of the project lives in the `src/ediechain/` directory,
structured as a standard Python package. The `pyproject.toml` file defines the package metadata,
dependencies, and entry points.

## Language
This is applying to the written text in docstrings, markdown files, jupyter (marimo) notebooks:
- **Simple English.** Short sentences, common words, no idioms, no metaphors, no coined terms.
- **Say the thing directly.** No build-up, no "it is worth noting that", no repeating the question
  before answering it.
- **Bullet-point facts, not paragraphs.** One fact per bullet.
- **Every fact carries its data.** A number in the text must be an f-string reading a frame computed
  in the same notebook. If some number is carried over from another notebook, then cite.
- **Small.** Keep only the results that are worth something. Cut analysis that answers nothing, even
  if it was expensive to produce.
- **Strict and compact.** If a section can be one table, make it one table.

## Code Style
- Use the vocabulary of `docs/glossary.md`. A term that is not there yet must be agreed before it is
written down.
- Follow current python version modern practices (type hints, dataclasses, f-strings)
- Use `ruff` for linting and auto-formatting (configured in `pyproject.toml`).
- Write comments only for non-obvious, hidden, implicit, side-effects, or surprising code. Avoid comments that
    just restate what the code does.
- NEVER write obvious comments. No banner/section-divider comments (e.g. `# ---- Models ----`),
    no comments that narrate what the next line plainly does. The reader can read the code.
- NEVER write obvious one-liner docstrings that just restate the function name or signature.
    Omit the docstring entirely when the name and types already say it.
- Use descriptive variable and function names to make code self-documenting.
- Docstrings should be used only for abstract base classes with non-obvious behavior.
- NEVER reference a notebook, a report or a section number from a comment or docstring in `src/`.
    Notebooks are renamed, renumbered and rewritten, so the reference is stale within weeks and
    nothing fails when it goes stale. Say what the code does and why; if a measurement is the
    reason, the measurement belongs in the notebook, not in `src/`.
- NEVER put a measured number in a `src/` comment or docstring. Numbers age with the data. State
    the property that holds (an inequality, an invariant) and let the notebook carry the evidence.
- Ensure lint, formatting, pytests are passing at the end of each development session.

## Markdown tables
- **A table line is at most 150 characters**, pipes included. A cell that does not fit is split
    across more rows, or the table becomes a list. A wide table is unreadable in a diff and in a
    plain-text editor, which is where it is reviewed.
- **Align the columns.** Pad every cell so the pipes line up, so the table reads as a table before
    it is rendered. This applies to a table a script writes as much as to one written by hand: a
    generator pads each cell to the width of its column.

## Merge Requests
Follow `.gitlab/merge_request_templates/Default.md`, but keep it short — nobody reads a wall of
text. Target the whole description at ~15 lines.

- **Description**: 2 sentences. What changed, and what it unblocks. No motivation essay, no
    retelling of the problem the diff already shows.
- **Changes**: only major changes (most important for reviewers), one-line bullets. Name the symbol
    or the API that changed, not the story behind it. No test counts, no lint status, no file-by-file
    inventory.
- **Priority Review Areas**: `file::symbol` plus a few words on what to scrutinize.
- **Additional Information**: only non-obvious consequences (cache invalidation, renamed keys,
    behavior changes), one line each. Omit the section when there are none.
- Do not annotate the checklist with explanations; leave the boxes for the author.

## CI/CD (`.gitlab-ci.yml`)

Pipeline stages:
1. **test** (automatic): `pytest` + `ruff check ./src` + Semgrep SAST/SCA/secrets
2. **publish** (manual, main-only): publishes wheel to `seznam-pypi-staging` when `pyproject.toml` version changes
3. **ai** (automatic): runs AI code review on merge to the main branch, more info `.gitlab-ci-code-review.yml`
