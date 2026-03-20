---
name: dynamo-one-shot-total-rna-velocity
description: Run or adapt a one-shot total RNA velocity workflow in `dynamo` for metabolic-labeling or scNT-seq `AnnData`, including monocle preprocessing with an optional curated gene list, grouped moments by labeling time, Model-2 `dynamics`, `calculate_velocity_alpha_minus_gamma_s`, low-dimensional projection with `cell_velocities`, and optional streamline or phase-portrait plotting. Use when converting tutorials such as `301_tutorial_hsc_velocity.ipynb`, or when choosing between `one_shot_method` branches like `sci_fate` and `combined` and projection `method` branches like `cosine` and `pearson`.
---

# Dynamo One-Shot Total RNA Velocity

## Goal

Run the stable execution spine behind the hematopoiesis scNT-seq tutorial: preprocess a one-shot labeling dataset, smooth by labeling time, force the Model-2 no-splicing branch for kinetics, compute total RNA velocity with `alpha - gamma * s`, then project that custom velocity to a low-dimensional embedding without treating the tutorial dataset as the workflow identity.

## Quick Workflow

1. Inspect the input `AnnData` for `new`, `total`, and `time`, plus any reusable helpers such as `uns["genes_to_use"]` and `obsm["X_umap"]`.
2. Preprocess with the monocle path. If a curated gene list exists, preserve it explicitly and use `Preprocessor(force_gene_list=...)` plus `config_monocle_recipe(..., n_top_genes=len(gene_list))`.
3. Materialize grouped moments with `dyn.tl.moments(..., group='time')` so the one-shot fit sees time-specific neighborhoods and creates `M_n`, `M_t`, and `M_s`.
4. Force the no-splicing Model-2 branch by setting `adata.uns["pp"]["has_splicing"] = False`, then run `dyn.tl.dynamics(..., group='time', one_shot_method='sci_fate', model='deterministic')` unless the user explicitly wants another branch.
5. Convert the fitted kinetic parameters into total RNA velocity with `dyn.tl.calculate_velocity_alpha_minus_gamma_s(...)`.
6. Project the custom total RNA velocity with `dyn.tl.cell_velocities(..., X=adata.layers["M_t"], V=adata.layers["velocity_alpha_minus_gamma_s"], method='cosine')`.
7. Treat `streamline_plot` and `phase_portraits` as optional visualization steps after `velocity_umap` already exists.

## Interface Summary

- `Preprocessor.config_monocle_recipe(adata, n_top_genes=2000)` configures the notebook-style monocle preprocessing path.
- `Preprocessor.preprocess_adata_monocle(adata, tkey=None, experiment_type=None)` is the exact notebook entrypoint; `Preprocessor.preprocess_adata(..., recipe='monocle', ...)` is the higher-level wrapper.
- `dyn.tl.moments(..., group=None, use_gaussian_kernel=False, use_mnn=False, layers='all')` creates smoothed first and second moments and stores `obsp["moments_con"]`.
- `dyn.tl.dynamics(..., group=None, model='auto', est_method='auto', one_shot_method='combined', ...)` estimates one-shot kinetic parameters and writes velocity-like layers such as `velocity_N` and `velocity_T`.
- `dyn.tl.calculate_velocity_alpha_minus_gamma_s(adata, gene_subset_key='use_for_pca', velocity_layer_name='velocity_alpha_minus_gamma_s')` converts fitted rates plus `M_n` and `M_s` into the total RNA velocity layer used in the notebook.
- `dyn.tl.cell_velocities(..., X=None, V=None, basis='umap', method='pearson', adj_key='distances', enforce=False)` projects the custom velocity to low-dimensional space and stores a kernel-specific transition matrix.
- `dyn.pl.streamline_plot(..., basis='umap', method='gaussian')` and `dyn.pl.phase_portraits(...)` are optional presentation entrypoints after the analytical path is complete.

Read `references/source-grounding.md` before documenting narrower parameter behavior than the live source currently supports.

## Branch Selection

- Use the curated-gene-list branch when `uns["genes_to_use"]` or a user-supplied marker/HVG list exists. Preserve that list explicitly before subsetting or concatenating because `anndata.concat(...)` can drop it.
- Use `recipe='monocle'` for this workflow family. Other `recipe` branches are real source-level options, but the total-RNA one-shot path here was empirically checked only on the monocle branch.
- Use `group='time'` on `moments(...)` and `dynamics(...)` unless the user explicitly wants to smooth and fit without time-specific neighborhoods.
- Use `one_shot_method='sci_fate'` for notebook parity and `one_shot_method='combined'` when the user explicitly wants the steady-state-plus-absolute-gamma branch. Both were empirically checked.
- Use `model='deterministic'` by default for this total-RNA workflow. If the user wants a splicing-aware or stochastic branch, stop and switch to a different workflow instead of silently mixing models.
- Use `cell_velocities(..., method='cosine')` for notebook-style total-RNA projection. Use `method='pearson'` when the user wants the alternate kernel with a different transition-matrix key. Treat `kmc`, `fp`, and `transform` as advanced branches with extra assumptions.
- Use `streamline_plot(..., method='gaussian')` for default streamline rendering. Switch to `method='SparseVFC'` only after a vector field has been reconstructed and the user explicitly wants that rendering path.

Read `references/branch-selection.md` before choosing non-default `recipe`, `one_shot_method`, `model`, `method`, or plotting branches.

## Input Contract

- Expect a metabolic-labeling `AnnData` with usable `new` and `total` layers and a labeling-time column.
- Expect the time column to be accessible as `adata.obs["time"]` before calling `calculate_velocity_alpha_minus_gamma_s(...)`, or rename / copy the user's time key to `time`.
- Treat `spliced` and `unspliced` as optional inputs for the raw object, not as the target model branch. The notebook-compatible total-RNA path forces `has_splicing=False` before `dynamics(...)`.
- Expect `M_n`, `M_t`, and `M_s` to be created by `moments(...)` before calling `calculate_velocity_alpha_minus_gamma_s(...)`.
- Expect a valid embedding such as `X_umap` plus a matching adjacency matrix under `obsp["distances"]` before calling `cell_velocities(...)`.
- Treat tutorial-specific fields such as `genes_to_use`, `cell_type`, and the hematopoiesis sample layout as worked examples, not universal defaults.

## Minimal Execution Patterns

For the notebook-compatible one-shot total RNA workflow:

```python
import dynamo as dyn

adata = dyn.sample_data.hematopoiesis_raw()
adata.obs_names_make_unique()

selected_genes = list(adata.uns["genes_to_use"])

pre = dyn.pp.Preprocessor(force_gene_list=selected_genes)
pre.config_monocle_recipe(adata, n_top_genes=len(selected_genes))
pre.preprocess_adata_monocle(adata, tkey="time", experiment_type="one-shot")

dyn.tl.reduceDimension(adata)
dyn.tl.moments(adata, group="time")

adata.uns["pp"]["has_splicing"] = False
dyn.tl.dynamics(
    adata,
    group="time",
    one_shot_method="sci_fate",
    model="deterministic",
    cores=1,
)

dyn.tl.calculate_velocity_alpha_minus_gamma_s(adata)
dyn.tl.cell_velocities(
    adata,
    enforce=True,
    X=adata.layers["M_t"],
    V=adata.layers["velocity_alpha_minus_gamma_s"],
    method="cosine",
)
```

For a generic one-shot labeling dataset without a baked-in gene list:

```python
pre = dyn.pp.Preprocessor()
pre.preprocess_adata(adata, recipe="monocle", tkey="time", experiment_type="one-shot")
```

For a non-default `combined` one-shot branch:

```python
dyn.tl.dynamics(
    adata,
    group="time",
    one_shot_method="combined",
    model="deterministic",
    cores=1,
)
```

For a non-default `pearson` projection branch:

```python
dyn.tl.cell_velocities(
    adata,
    enforce=True,
    X=adata.layers["M_t"],
    V=adata.layers["velocity_alpha_minus_gamma_s"],
    method="pearson",
)
```

## Validation

After preprocessing, check these items:

- `adata.uns["pp"]["experiment_type"] == "one-shot"`
- `adata.var["use_for_pca"]` exists
- `adata.obsm["X_pca"]` exists

After `moments(..., group='time')`, check these items:

- `adata.obsp["moments_con"]` exists
- `adata.layers["M_n"]`, `adata.layers["M_t"]`, and `adata.layers["M_s"]` exist

Before `dynamics(...)`, check these items:

- `adata.obs["time"]` exists
- `adata.uns["pp"]["has_splicing"] is False` for the Model-2 branch

After `dynamics(...)`, check these items:

- `adata.uns["dynamics"]["model"] == "deterministic"` for the default branch
- `adata.layers["velocity_T"]` exists
- `adata.var["use_for_dynamics"]` exists

After `calculate_velocity_alpha_minus_gamma_s(...)`, check these items:

- `adata.layers["velocity_alpha_minus_gamma_s"]` exists
- the function reports each unique time point it processed

After `cell_velocities(...)`, check these items:

- `adata.obsm["velocity_umap"]` exists for the default UMAP branch
- `adata.obsp["cosine_transition_matrix"]` exists for `method='cosine'`
- `adata.obsp["pearson_transition_matrix"]` exists for `method='pearson'`

## Constraints

- Do not treat `velocity_T` from `dynamics(...)` as the notebook endpoint. The tutorial's total-RNA visualization path depends on the extra `calculate_velocity_alpha_minus_gamma_s(...)` plus `cell_velocities(...)` steps.
- Do not forget to rename or copy the time column to `adata.obs["time"]` before `calculate_velocity_alpha_minus_gamma_s(...)`; current source hard-codes that key.
- Do not assume `uns["genes_to_use"]` survives `AnnData` concatenation or subsetting workflows. Preserve it explicitly when reconstructing subsets.
- Do not keep `has_splicing=True` if the goal is Model-2 total RNA velocity. Preprocessing auto-detects splicing and labeling together on the hematopoiesis raw object.
- Do not interpret `one_shot_method` from `adata.uns["dynamics"]`; current runtime does not store it there.
- Do not make small per-time groups the default. Reviewer runs with fewer than `50` cells per time group emitted NaN-velocity warnings and filtered genes before projection.

## Resource Map

- Read `references/branch-selection.md` for decision rules across `recipe`, grouped smoothing, `one_shot_method`, and projection `method` branches.
- Read `references/visualization-and-phase-portraits.md` when the user explicitly wants notebook-style streamlines or total-RNA phase-portrait adaptation.
- Read `references/source-grounding.md` for inspected signatures, source-level branch evidence, and empirical execution notes.
- Read `references/source-notebook-map.md` to see how `docs/tutorials/notebooks/301_tutorial_hsc_velocity.ipynb` was converted into this reusable skill.
- Read `references/compatibility.md` when notebook wording and current source behavior diverge.
