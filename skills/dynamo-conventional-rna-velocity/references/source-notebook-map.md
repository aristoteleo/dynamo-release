# Source Notebook Map

Source notebook:
`docs/tutorials/notebooks/200_zebrafish.ipynb`

Use this file to see how the notebook was converted into a reusable stage-based skill.

Conversion rule used here:

- the stable skill identity is conventional spliced/unspliced RNA velocity in `dynamo`
- zebrafish remains the worked example notebook, not the trigger surface

## Notebook Sections To Skill Responsibilities

### 1. Setup and sample-data loading

Notebook role:

- install / import `dynamo`
- set plotting style
- load `dyn.sample_data.zebrafish()`

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Input Contract
- `references/source-grounding.md`
- `references/compatibility.md`

Intentionally dropped:

- notebook magics
- presentation styling
- inline package-upgrade cell

### 2. Preprocessing

Notebook role:

- run the monocle preprocessing path on conventional spliced / unspliced data

Preserved in the skill:

- `SKILL.md` Stage Selection
- `SKILL.md` Minimal Execution Patterns
- `references/stage-selection.md`

Source-grounded addition:

- current `recipe` branch set includes more than the notebook’s single visible path

### 3. Kinetics and low-dimensional velocity

Notebook role:

- run `dynamics(model='stochastic')`
- compute UMAP
- project velocity with `cell_velocities(method='pearson')`

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `SKILL.md` Validation
- `references/stage-selection.md`

Source-grounded addition:

- `cell_velocities` supports multiple kernel methods beyond the single notebook call
- transition matrix storage is kernel-specific

### 4. Confidence diagnostics and correction

Notebook role:

- inspect gene-wise and cell-wise confidence
- optionally reproject confident cell velocities using lineage priors

Preserved in the skill:

- `SKILL.md` Stage Selection
- `SKILL.md` Minimal Execution Patterns
- `references/stage-selection.md`

Worked-example downgrade:

- zebrafish lineage labels remain as a worked example only
- generic `group_key` and `lineage_dict` are the main reusable interface

### 5. Vector field, topology, and potential

Notebook role:

- reconstruct UMAP vector field
- inspect fixed points and topology
- use ddHodge potential as pseudotime-like ordering

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/visualization-and-animation.md`

Worked-example downgrade:

- zebrafish progenitor labels remain only as an example selection
- the reusable interface uses dataset-specific `progenitor_labels`
- `references/source-grounding.md`

Source-grounded addition:

- `pot_curl_div=True` is the compact executable route to the notebook’s potential / curl / divergence stage

### 6. PCA vector calculus

Notebook role:

- switch to PCA basis
- compute speed, curl, divergence, acceleration, curvature

Preserved in the skill:

- `SKILL.md` Stage Selection
- `SKILL.md` Minimal Execution Patterns
- `references/stage-selection.md`

### 7. Fate prediction and animation

Notebook role:

- select progenitors
- run `pd.fate`
- animate with `StreamFuncAnim`, `animate_fates`, and a streamplot GIF path

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/visualization-and-animation.md`

## What Was Intentionally Not Carried Over

- extensive biological interpretation prose
- figure-by-figure walkthrough text
- repeated save / reload cells
- notebook-only HTML / JS embedding cells

## When To Reopen The Notebook

Reopen the notebook only when:

- the user wants figure parity or presentation styling
- the user wants the exact published gene panels and story-line interpretation
- a future notebook revision changes the demonstrated stage order or branch choices
