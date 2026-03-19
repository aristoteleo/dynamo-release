---
name: dynamo-conventional-rna-velocity
description: Run or adapt a conventional spliced/unspliced RNA velocity workflow in `dynamo`, including `Preprocessor` preprocessing, `dynamics`, low-dimensional `cell_velocities`, `VectorField`, topology / potential analysis, confidence-based correction, fate prediction, and optional animation. Use when analyzing conventional scRNA-seq `AnnData`, reproducing or adapting tutorial notebooks such as `200_zebrafish.ipynb`, or selecting between preprocessing, kinetics, vector-field, and fate stages for a reusable velocity pipeline.
---

# Dynamo Conventional RNA Velocity

## Goal

Run the reusable execution spine for conventional spliced / unspliced `dynamo` analysis: preprocess a compatible `AnnData`, estimate velocities with the current `dynamics` API, project them to a low-dimensional basis, reconstruct a vector field, then optionally branch into confidence diagnostics, topology / potential, PCA vector calculus, or fate animation without treating any one notebook or dataset as the workflow identity.

## Quick Workflow

1. Load a conventional spliced / unspliced dataset, make cell names unique, and confirm the grouping column you want to use for interpretation.
2. Preprocess with `Preprocessor.preprocess_adata(..., recipe='monocle')` unless the user explicitly wants another `recipe`.
3. Estimate kinetics with `dyn.tl.dynamics(...)`, using `model='stochastic'` by default for conventional velocity analysis unless speed or legacy reproduction requires another branch.
4. Run `dyn.tl.reduceDimension(...)`, then `dyn.tl.cell_velocities(...)` on the basis the user actually needs.
5. Reconstruct a vector field with `dyn.vf.VectorField(...)`, then choose one of the downstream branches:
   confidence / correction, topology / potential, PCA vector calculus, or fate prediction.
6. Treat animation and styling cells as optional presentation steps after the analytical outputs already exist.

## Interface Summary

- `Preprocessor.preprocess_adata(adata, recipe='monocle', tkey=None, experiment_type=None)` is the preprocessing wrapper.
- `dyn.tl.dynamics(...)` is the core kinetics and raw-velocity entrypoint.
- `dyn.tl.cell_velocities(..., basis='umap', method='pearson')` projects high-dimensional velocity to a low-dimensional basis and stores a kernel-specific transition matrix.
- `dyn.vf.VectorField(..., basis='umap' | 'pca', method='SparseVFC', pot_curl_div=False, **kwargs)` reconstructs the vector field.
- `dyn.pd.fate(..., direction='both', average=False, inverse_transform=False)` integrates trajectories from chosen initial cells.
- `dyn.mv.animate_fates(...)` and `dyn.mv.StreamFuncAnim(...)` are optional post-fate animation entrypoints, not core analysis prerequisites.

Read `references/source-grounding.md` before documenting parameter behavior more narrowly than the live source does.

## Stage Selection

- Use the full notebook spine only when the user wants an end-to-end tutorial reproduction.
- Stop after `dynamics` if the goal is only gene-wise kinetics or phase portraits.
- Stop after `cell_velocities` if the goal is low-dimensional velocity visualization.
- Use `VectorField(..., basis='umap')` for topology, potential, and fate visualization in the same embedding used for plots.
- Use `cell_velocities(..., basis='pca')` plus `VectorField(..., basis='pca')` for speed, divergence, acceleration, and curvature.
- Use `gene_wise_confidence`, `cell_wise_confidence`, and `confident_cell_velocities` only when the user is debugging suspect flow directions or wants lineage-prior correction.

Read `references/stage-selection.md` before choosing non-default `model`, `method`, `basis`, or `direction` branches.

## Input Contract

- Expect a conventional scRNA-seq `AnnData` with usable `spliced` and `unspliced` layers.
- Expect to compute a fresh low-dimensional basis unless the user already provides a validated embedding.
- Expect any grouping column, lineage hints, and marker genes to be dataset-specific inputs, not universal defaults.
- Treat zebrafish tutorial labels such as `Cell_type`, `Proliferating Progenitor`, and `Schwann Cell` as worked examples only.

## Minimal Execution Patterns

For the default conventional spliced / unspliced workflow:

```python
import dynamo as dyn

adata = ...  # Any conventional AnnData with spliced/unspliced layers
adata.obs_names_make_unique()

preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=False)
preprocessor.preprocess_adata(adata, recipe="monocle")

dyn.tl.dynamics(adata, model="stochastic", cores=1)
dyn.tl.reduceDimension(adata)
dyn.tl.cell_velocities(adata, basis="umap", method="pearson")
dyn.vf.VectorField(adata, basis="umap", M=1000, pot_curl_div=False)
```

For the worked example notebook path:

```python
adata = dyn.sample_data.zebrafish()
adata.obs_names_make_unique()
```

For confidence-based correction after the initial velocity run:

```python
dyn.tl.gene_wise_confidence(
    adata,
    group=group_key,
    lineage_dict=lineage_dict,
)
dyn.tl.cell_wise_confidence(adata, method="jaccard")
dyn.tl.confident_cell_velocities(
    adata,
    group=group_key,
    lineage_dict=lineage_dict,
)
```

For the zebrafish worked example only:

```python
group_key = "Cell_type"
lineage_dict = {"Proliferating Progenitor": ["Schwann Cell"]}
```

For topology / potential on UMAP:

```python
dyn.vf.VectorField(adata, basis="umap", M=1000, pot_curl_div=True)
# This populates keys such as:
# adata.obs["umap_ddhodge_potential"], adata.obs["curl_umap"], adata.obs["divergence_umap"]
```

For PCA vector calculus:

```python
dyn.tl.cell_velocities(adata, basis="pca", method="pearson")
dyn.vf.VectorField(adata, basis="pca")
dyn.vf.speed(adata, basis="pca")
dyn.vf.divergence(adata, basis="pca")
dyn.vf.acceleration(adata, basis="pca")
dyn.vf.curvature(adata, basis="pca")
```

For forward fate prediction from a dataset-specific progenitor selection:

```python
init_cells = adata.obs_names[adata.obs[group_key].isin(progenitor_labels)]

dyn.pd.fate(
    adata,
    basis="umap",
    init_cells=init_cells[:20],
    interpolation_num=100,
    direction="forward",
    inverse_transform=False,
    average=False,
    cores=1,
)
```

For the zebrafish worked example only:

```python
group_key = "Cell_type"
progenitor_labels = ["Proliferating Progenitor", "Pigment Progenitor"]
```

## Validation

After preprocessing, check these items:

- `adata.uns["pp"]["experiment_type"] == "conventional"` for the chosen dataset.
- `adata.obsm["X_pca"]` exists.
- `adata.var["use_for_pca"]` exists.

After `dynamics`, check these items:

- `adata.uns["dynamics"]["model"]` matches the chosen branch.
- `adata.uns["dynamics"]["experiment_type"] == "conventional"` on conventional data.
- `adata.layers["velocity_S"]` exists.
- `adata.var["use_for_dynamics"]` exists.

After `cell_velocities(..., method="pearson")`, check these items:

- `adata.obsm["velocity_umap"]` exists for the UMAP branch.
- the transition matrix is stored under the kernel-specific key `adata.obsp["pearson_transition_matrix"]`, not a generic `transition_matrix`.

After `VectorField(..., basis="umap")`, check these items:

- `adata.uns["VecFld_umap"]` exists.
- `adata.obs["obs_vf_angle_umap"]` exists.

After `VectorField(..., pot_curl_div=True)`, check these items:

- `adata.obs["umap_ddhodge_potential"]` exists.
- `adata.obs["curl_umap"]` and `adata.obs["divergence_umap"]` exist.

After `pd.fate(..., basis="umap")`, check these items:

- `adata.uns["fate_umap"]` exists.
- `adata.uns["fate_umap"]["t"]` and `adata.uns["fate_umap"]["prediction"]` are populated.

## Constraints

- Do not assume any tutorial dataset already has `adata.obsm["X_umap"]`; compute or validate the embedding before downstream UMAP-specific steps.
- Do not assume observation names are unique on load; call `adata.obs_names_make_unique()` before branching deeper into the workflow.
- Do not copy zebrafish-specific grouping columns, lineage labels, or marker genes into a different dataset without replacing them.
- Do not treat every notebook plot as analytically required. Most styling, save / reload, and presentation cells are optional.
- Do not document `VectorField(..., M=1000)` as if it were in the public signature; `M` is passed through `**kwargs` into SparseVFC defaults.
- Prefer `method='pearson'` only for notebook parity. Other `cell_velocities` kernels are real source-level branches and should not be silently omitted.

## Resource Map

- Read `references/stage-selection.md` for branch-by-branch defaults and decision rules.
- Read `references/source-grounding.md` for inspected signatures, source-level branch evidence, and empirical execution notes.
- Read `references/visualization-and-animation.md` when the user specifically wants `topography`, `StreamFuncAnim`, `animate_fates`, or `animate_streamplot`.
- Read `references/source-notebook-map.md` to trace `200_zebrafish.ipynb` into the generic velocity skill and its worked-example sections.
- Read `references/compatibility.md` when notebook wording and current source behavior appear to disagree.
