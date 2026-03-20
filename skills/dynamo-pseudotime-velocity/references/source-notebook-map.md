# Source Notebook Map

Source notebook:
`docs/tutorials/notebooks/201_dynamo_beyondvelo.ipynb`

Use this file to see how the notebook was converted into a reusable pseudotime-to-velocity skill.

Conversion rule used here:

- the stable skill identity is converting pseudotime into `dynamo` velocity and downstream vector-field outputs
- bone marrow remains the worked example notebook, not the trigger surface

## Notebook Sections To Skill Responsibilities

### 1. Setup And Sample-Data Loading

Notebook role:

- import `dynamo`
- set plotting style
- load `dyn.sample_data.bone_marrow()`

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Input Contract
- `references/source-grounding.md`
- `references/compatibility.md`

Intentionally dropped:

- notebook magics
- presentation styling
- dependency-version display cells

### 2. Precomputed Embedding Reuse

Notebook role:

- copy `X_tsne` into `X_umap` for demonstration

Preserved in the skill:

- `SKILL.md` Branch Selection
- `SKILL.md` Minimal Execution Patterns

Downgrade rule:

- this is kept as an optional parity branch, not the default reusable workflow

### 3. Preprocessing

Notebook role:

- reuse the monocle preprocessing path before downstream analysis

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `references/branch-selection.md`

Source-grounded addition:

- current `Preprocessor` source exposes more `recipe` branches than the notebook uses

### 4. Convert Pseudotime To Velocity

Notebook role:

- create `M_s`
- run `dyn.tl.pseudotime_velocity(adata, pseudotime='palantir_pseudotime')`

Preserved in the skill:

- `SKILL.md` Interface Summary
- `SKILL.md` Minimal Execution Patterns
- `SKILL.md` Validation

Source-grounded additions:

- explicit `method` branches: `hodge`, `gradient`, `naive`
- explicit `dynamics_info` and `unspliced_RNA` behavior
- explicit requirement that `adj_key` actually exist in `.obsp`

### 5. Velocity Visualization

Notebook role:

- `streamline_plot`
- `cell_wise_vectors`

Preserved in the skill:

- `references/visualization-and-animation.md`

### 6. Vector Field, Potential, And Topography

Notebook role:

- `VectorField(..., pot_curl_div=True)`
- `plot_energy`
- `topography`
- `umap(..., color='umap_ddhodge_potential')`

Preserved in the skill:

- `SKILL.md` Branch Selection
- `SKILL.md` Validation
- `references/visualization-and-animation.md`

Source-grounded addition:

- `pot_curl_div=True` is the compact executable branch that materializes potential, curl, and divergence outputs together

### 7. Fate And Animation

Notebook role:

- select progenitors
- run `pd.fate`
- build `StreamFuncAnim`
- create animated streamplots

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/visualization-and-animation.md`

## Source-Grounded Additions Beyond The Notebook

These details were added from live source inspection and empirical execution, not notebook prose alone:

- `reduceDimension(...)` branch coverage for `reduction_method`
- `pseudotime_velocity(...)` branch coverage for `method`
- the runtime trap where `reduceDimension(...)` can leave `uns['neighbors']` populated while `obsp['distances']` is still missing
- the requirement that `M_s` be sparse-backed or otherwise `.toarray()`-compatible
- the optional `dynamics_info=True` branch for downstream metadata compatibility

## What Was Intentionally Not Carried Over

- long biological interpretation prose
- notebook-only HTML / JS embedding cells
- repeated background-color toggles
- direct file-save examples whose only purpose is presentation parity

## When To Reopen The Notebook

Reopen the notebook only when:

- the user wants exact bone marrow figure parity
- the user wants the same embedded animation presentation style
- a future notebook revision changes the demonstrated pseudotime handoff or downstream branch order
