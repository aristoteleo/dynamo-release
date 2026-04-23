---
name: dynamo-pseudotime-velocity
description: Convert pseudotime into reusable `dynamo` RNA velocity outputs on an `AnnData` object, then optionally continue into vector field, topology / potential, fate, and animation without relying on measured spliced/unspliced kinetics. Use when adapting the `201_dynamo_beyondvelo.ipynb` tutorial, working from pseudotime plus a neighbor graph and embedding, or choosing between `pseudotime_velocity` method branches such as `hodge`, `gradient`, and `naive`.
---

# Dynamo Pseudotime Velocity

## Goal

Turn pseudotime plus a valid neighbor graph into low-dimensional and gene-wise velocity with the current `dynamo` API, then optionally branch into vector-field reconstruction, topology / potential analysis, fate prediction, and presentation-only animation without treating the bone marrow notebook as the workflow identity.

## Quick Workflow

1. Inspect the user's `AnnData` for a pseudotime column, a usable embedding, and whether preprocessing still needs to be run.
2. Preprocess with `Preprocessor.preprocess_adata(..., recipe='monocle')` unless the user explicitly wants another `recipe`.
3. Materialize the low-dimensional basis and neighbor graph you will actually use. Do not assume `reduceDimension(...)` already left `obsp['distances']` behind.
4. Create a sparse-backed `M_s` layer, then run `dyn.tl.pseudotime_velocity(...)` with the right `method` and `dynamics_info` choice.
5. Add `dyn.vf.VectorField(...)` only when the user needs topology, potential, curl / divergence, or downstream fate prediction.
6. Treat `topography`, `StreamFuncAnim`, and `animate_streamplot` as optional presentation steps after the analytical outputs already exist.

## Interface Summary

- `Preprocessor.preprocess_adata(adata, recipe='monocle', tkey=None, experiment_type=None)` is the preprocessing wrapper.
- `dyn.tl.reduceDimension(..., reduction_method='umap')` is the default low-dimensional embedding path, but current source also exposes `trimap`, `diffusion_map`, `tsne`, `psl`, and `sude`.
- `dyn.tl.pseudotime_velocity(..., pseudotime='pseudotime', basis='umap', adj_key='distances', ekey='M_s', method='hodge', dynamics_info=False, unspliced_RNA=False)` is the core pseudotime-to-velocity entrypoint.
- `dyn.vf.VectorField(..., basis='umap', method='SparseVFC', pot_curl_div=False, **kwargs)` reconstructs the low-dimensional vector field.
- `dyn.pd.fate(..., direction='both', average=False, sampling='arc_length', inverse_transform=False)` predicts trajectories from chosen initial cells after vector-field reconstruction.
- `dyn.pl.topography(...)`, `dyn.mv.StreamFuncAnim(...)`, `dyn.pl.compute_velocity_on_grid(...)`, and `dyn.pl.animate_streamplot(...)` are optional notebook-style visualization / animation entrypoints.

Read [source-grounding.md](./references/source-grounding.md) before documenting any narrower parameter behavior than the live source currently supports.

## Branch Selection

- Use the fresh-embedding branch by default: preprocess, run `reduceDimension(...)`, then explicitly run `dyn.tl.neighbors(...)` if `obsp['distances']` is missing.
- Use the existing-embedding parity branch only when the user already has a trusted 2D embedding such as `X_tsne` and wants notebook-like plotting parity more than a fresh UMAP.
- Use `method='hodge'` for the normal pseudotime-to-velocity path. It is the default and was empirically checked through vector field, potential, and fate.
- Use `method='gradient'` when the user explicitly wants the alternate graph-gradient construction. This branch was also empirically checked and still populates the same core velocity outputs.
- Use `method='naive'` only when the user explicitly asks for that older branch or you are debugging differences between graph-construction strategies.
- Use `dynamics_info=True` when downstream tools or checks need `adata.uns['dynamics']`; notebook parity leaves it at the default `False`, but the reusable skill should prefer the more complete branch when compatibility matters.
- Use `unspliced_RNA=True` only when the user explicitly wants the pseudo-unspliced output stored under `add_ukey` such as `M_u_pseudo`.
- Use `VectorField(..., method='SparseVFC')` by default. Switch to `method='dynode'` only when the environment has `dynode` and the user explicitly wants that branch.
- Use `pd.fate(..., direction='forward', average=False, inverse_transform=False)` for notebook-like future-state prediction from progenitor cells. Keep `sampling='arc_length'` unless the user explicitly wants another path.

Read [branch-selection.md](./references/branch-selection.md) before choosing non-default `recipe`, `reduction_method`, `method`, `direction`, or `sampling` branches.

## Input Contract

- Expect an `AnnData` with a pseudotime column such as `adata.obs['palantir_pseudotime']`.
- Expect either a valid embedding already stored under the basis-specific key such as `adata.obsm['X_umap']`, or enough information to compute one.
- Expect a materialized adjacency matrix in `.obsp` under the chosen `adj_key`. The default is `distances`.
- Expect the expression layer passed as `ekey` to support `.toarray()`. If you alias from dense `adata.X`, wrap it with `scipy.sparse.csr_matrix(...)` first.
- Treat notebook-specific columns such as `clusters`, `palantir_pseudotime`, and `X_tsne` as worked-example inputs, not universal defaults.

## Minimal Execution Patterns

For the default reusable path:

```python
import dynamo as dyn
from scipy import sparse

adata = dyn.sample_data.bone_marrow()
adata.obs_names_make_unique()

pre = dyn.pp.Preprocessor(cell_cycle_score_enable=False)
pre.preprocess_adata(adata, recipe="monocle")

dyn.tl.reduceDimension(adata)
if "distances" not in adata.obsp:
    dyn.tl.neighbors(adata, basis="pca")

adata.layers["M_s"] = adata.X.copy() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)

dyn.tl.pseudotime_velocity(
    adata,
    pseudotime="palantir_pseudotime",
    basis="umap",
    method="hodge",
    dynamics_info=True,
)
```

For notebook-style embedding reuse when an existing 2D embedding already exists:

```python
if "X_tsne" in adata.obsm and "X_umap" not in adata.obsm:
    adata.obsm["X_umap"] = adata.obsm["X_tsne"].copy()

if "distances" not in adata.obsp:
    dyn.tl.neighbors(adata, basis="pca")
```

For the optional vector-field, topology, and fate branch:

```python
dyn.vf.VectorField(adata, basis="umap", M=100, pot_curl_div=True, cores=1)

progenitor = adata.obs_names[adata.obs[group_key].isin(progenitor_labels)][:20]

dyn.pd.fate(
    adata,
    basis="umap",
    init_cells=progenitor,
    interpolation_num=30,
    direction="forward",
    inverse_transform=False,
    average=False,
    cores=1,
)
```

For the non-default `gradient` branch:

```python
dyn.tl.pseudotime_velocity(
    adata,
    pseudotime="palantir_pseudotime",
    basis="umap",
    method="gradient",
    dynamics_info=False,
)
```

## Validation

After preprocessing, check these items:

- `adata.obsm["X_pca"]` exists.
- `adata.var["use_for_pca"]` exists.
- `adata.uns["pp"]` exists.

Before `pseudotime_velocity(...)`, check these items:

- the chosen pseudotime key exists in `adata.obs`
- `adata.obsm["X_umap"]` exists for the default `basis='umap'` path
- `adata.obsp["distances"]` exists if `adj_key='distances'`
- `adata.layers["M_s"]` exists and is sparse-backed or otherwise `.toarray()`-compatible

After `pseudotime_velocity(...)`, check these items:

- `adata.obsm["velocity_umap"]` exists
- `adata.layers["velocity_S"]` exists
- `adata.obsp["pseudotime_transition_matrix"]` exists
- `adata.var["use_for_dynamics"]` and `adata.var["use_for_transition"]` exist
- `adata.uns["dynamics"]` exists only if `dynamics_info=True`

After `VectorField(..., basis='umap', pot_curl_div=True)`, check these items:

- `adata.uns["VecFld_umap"]` exists
- `adata.obs["obs_vf_angle_umap"]` exists
- `adata.obs["umap_ddhodge_potential"]` exists
- `adata.obs["curl_umap"]` and `adata.obs["divergence_umap"]` exist

After `pd.fate(..., basis='umap')`, check these items:

- `adata.uns["fate_umap"]` exists
- `adata.uns["fate_umap"]["t"]` exists
- `adata.uns["fate_umap"]["prediction"]` exists

## Constraints

- Do not assume `dyn.tl.reduceDimension(...)` alone leaves a usable `obsp['distances']` matrix in this runtime. Check and materialize it explicitly with `dyn.tl.neighbors(...)` when needed.
- Do not treat `adata.obsm['X_umap'] = adata.obsm['X_tsne']` as the default workflow. It is a demo-specific parity shortcut from the notebook.
- Do not set `adata.layers["M_s"] = adata.X` blindly when `adata.X` is dense. Current source calls `.toarray()` on `ekey`.
- Do not imply that `M=1000` is a public `VectorField` signature parameter. Current source forwards it through `**kwargs`.
- Do not let animation setup block the analytical path. `StreamFuncAnim` and GIF export may require external tooling such as `imagemagick`.

## Resource Map

- Read [branch-selection.md](./references/branch-selection.md) for decision rules across `recipe`, `reduction_method`, `method`, and `fate` branches.
- Read [visualization-and-animation.md](./references/visualization-and-animation.md) when the user explicitly wants notebook-style plots, topology figures, or animations.
- Read [source-grounding.md](./references/source-grounding.md) for inspected signatures, source-level branch evidence, and empirical execution notes.
- Read [source-notebook-map.md](./references/source-notebook-map.md) to trace `201_dynamo_beyondvelo.ipynb` into the reusable skill layout.
- Read [compatibility.md](./references/compatibility.md) when notebook prose and current source behavior diverge.
