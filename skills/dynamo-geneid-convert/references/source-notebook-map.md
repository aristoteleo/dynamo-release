# Source Notebook Map

Source notebook:
`docs/tutorials/notebooks/110_geneid_convert_tutorial.ipynb`

Use this file to trace how the notebook was turned into a reusable skill instead of a summary.

## Notebook Sections To Skill Responsibilities

### 1. Motivation and setup

Notebook role:

- explain why gene-symbol conversion matters
- establish the two demonstration scenarios

Preserved in the skill:

- `SKILL.md` Goal
- `SKILL.md` Conversion Path Selection
- `references/conversion-paths.md`

Dropped from the skill:

- plotting-style setup
- notebook-only exposition

### 2. Human example

Notebook role:

- show direct conversion on human Ensembl IDs
- establish the safe `query` / merge / `var_names` update pattern

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `SKILL.md` Validation
- `references/source-grounding.md`

### 3. Zebrafish example

Notebook role:

- show version-suffix stripping
- show explicit `ensembl_release=77`

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/conversion-paths.md`
- `references/compatibility.md`

Source-grounded addition beyond the notebook:

- `convert2symbol(adata)` does not auto-handle zebrafish without explicit `scopes`

### 4. Optional preprocessing follow-on

Notebook role:

- show that ID conversion should happen before preprocessing
- use a monocle-style `Preprocessor` path

Preserved in the skill:

- `SKILL.md` Optional Preprocess Integration
- `references/preprocess-handoff.md`

Source-grounded addition beyond the notebook:

- current source exposes five `recipe` branches, not just the single monocle path demonstrated here

## What Was Intentionally Not Carried Over

- long pedagogical explanation
- raw notebook outputs
- visualization-only cells
- any wording that implied the implementation is MyGene-backed instead of current vendored `pyensembl`

## When To Reopen The Notebook

Reopen the notebook only when:

- the user wants cell-by-cell historical reproduction
- the user wants figure parity
- a future notebook revision changes the demonstrated IDs or preprocessing handoff
- source behavior and notebook prose appear to disagree and the difference affects execution
