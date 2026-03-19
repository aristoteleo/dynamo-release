# Source Notebook Map

Source notebook:
`docs/tutorials/notebooks/301_tutorial_hsc_velocity.ipynb`

Use this file to see how the hematopoiesis scNT-seq tutorial was converted into a reusable one-shot total RNA velocity skill.

Conversion rule used here:

- the stable skill identity is one-shot total RNA velocity in `dynamo`
- hematopoiesis remains the worked example notebook and sample dataset, not the trigger surface

## Notebook Sections To Skill Responsibilities

### 1. Setup And Sample-Data Loading

Notebook role:

- import `dynamo`
- set plotting style
- load `dyn.sample_data.hematopoiesis_raw()`

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Minimal Execution Patterns
- `references/source-grounding.md`

Intentionally dropped:

- warning filters
- plotting-style boilerplate

### 2. Predefined Gene List

Notebook role:

- read `adata_hsc_raw.uns["genes_to_use"]`

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Branch Selection
- `references/compatibility.md`

Source-grounded addition:

- `anndata.concat(...)` can drop `uns["genes_to_use"]`, so the reusable skill calls out explicit preservation during subset reconstruction

### 3. Monocle Preprocessing

Notebook role:

- instantiate `Preprocessor(force_gene_list=selected_genes_to_use)`
- call `config_monocle_recipe(..., n_top_genes=len(selected_genes_to_use))`
- call `preprocess_adata_monocle(..., tkey="time", experiment_type="one-shot")`

Preserved in the skill:

- `SKILL.md` Interface Summary
- `SKILL.md` Minimal Execution Patterns
- `references/branch-selection.md`

Source-grounded addition:

- current `Preprocessor` source also exposes higher-level `preprocess_adata(..., recipe='monocle', ...)` and other recipe branches

### 4. Embedding And Grouped Moments

Notebook role:

- call `dyn.tl.reduceDimension(...)`
- call `dyn.tl.moments(..., group="time")`

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Validation
- `references/branch-selection.md`

Source-grounded addition:

- `moments(...)` is where `M_n`, `M_t`, `M_s`, and `moments_con` are created for this workflow

### 5. Force Model 2 And Fit Kinetics

Notebook role:

- set `adata_hsc_raw.uns["pp"]["has_splicing"] = False`
- run `dyn.tl.dynamics(..., group="time", one_shot_method="sci_fate", model="deterministic")`

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Branch Selection
- `references/compatibility.md`

Source-grounded additions:

- `one_shot_method='combined'` is a real alternate branch
- current runtime does not store the chosen `one_shot_method` in `adata.uns["dynamics"]`

### 6. Calculate Total RNA Velocity

Notebook role:

- run `dyn.tl.calculate_velocity_alpha_minus_gamma_s(adata_hsc_raw)`

Preserved in the skill:

- `SKILL.md` Interface Summary
- `SKILL.md` Validation
- `references/source-grounding.md`

Source-grounded additions:

- current source hard-codes `adata.obs["time"]`
- current source reads `M_n` and `M_s` and writes `velocity_alpha_minus_gamma_s`

### 7. Project To UMAP

Notebook role:

- run `dyn.tl.cell_velocities(..., X=M_t, V=velocity_alpha_minus_gamma_s, method="cosine")`

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/branch-selection.md`

Source-grounded addition:

- `cell_velocities` has more kernel branches than the notebook uses and stores kernel-specific transition-matrix keys

### 8. Visualization

Notebook role:

- `dyn.pl.streamline_plot(...)`
- markdown-only note about phase diagrams

Preserved in the skill:

- `references/visualization-and-phase-portraits.md`

Intentional downgrade:

- streamline plotting is retained as an optional post-analysis branch
- phase portraits are documented as an inference-backed adaptation because the executed code cell is not present in the extracted notebook content

## Source-Grounded Additions Beyond The Notebook

These details were added from live source inspection and empirical execution, not notebook prose alone:

- recipe branches exposed by `Preprocessor`
- `moments(...)` output keys and grouped-smoothing behavior
- `dynamics(...)` one-shot branch coverage for `sci_fate` and `combined`
- the fact that `velocity_T` exists after `dynamics(...)` but `velocity_S` does not on the checked Model-2 path
- the hard requirement that `calculate_velocity_alpha_minus_gamma_s(...)` sees `adata.obs["time"]`
- kernel-specific output keys for `cell_velocities(...)`

## What Was Intentionally Not Carried Over

- tutorial narration about published figures
- warning-filter boilerplate
- display-only notebook setup
- unsupported claims about the exact phase-portrait cell that was not present in the extracted notebook JSON

## When To Reopen The Notebook

Reopen the notebook only when:

- the user wants exact figure-text parity with the tutorial prose
- a newer notebook revision includes the missing phase-portrait code cell
- the tutorial starts using a different one-shot estimator or storage convention
