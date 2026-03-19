# Source Grounding

This skill was generated from live code inspection and empirical execution on the bone marrow worked example, not from notebook prose alone.

## Inspection Notes

Source inspection and empirical checks were performed in a local Python runtime compatible with this repository's dependencies.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/201_dynamo_beyondvelo.ipynb`

Code inspected:

- `dynamo/preprocessing/Preprocessor.py`
- `dynamo/tools/dimension_reduction.py`
- `dynamo/tools/utils_reduceDimension.py`
- `dynamo/tools/connectivity.py`
- `dynamo/tools/pseudotime_velocity.py`
- `dynamo/vectorfield/VectorField.py`
- `dynamo/prediction/fate.py`
- `dynamo/plot/topography.py`
- `dynamo/movie/fate.py`
- `dynamo/plot/animation_lines.py`

## Inspected Interfaces

### `Preprocessor.preprocess_adata`

Observed signature:

```python
(self, adata, recipe='monocle', tkey=None, experiment_type=None) -> None
```

Observed `recipe` branches in current source:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

### `dyn.tl.reduceDimension`

Observed signature includes:

```python
(..., basis='pca', n_pca_components=30, reduction_method='umap', ...) -> Optional[AnnData]
```

Observed `reduction_method` branches in current source:

- `trimap`
- `diffusion_map`
- `tsne`
- `umap`
- `psl`
- `sude`

Important source detail:

- `reduceDimension(...)` may populate `uns['neighbors']` itself
- because of that, its fallback `neighbors(...)` call may be skipped even when `obsp['distances']` is still absent

### `dyn.tl.pseudotime_velocity`

Observed signature:

```python
(adata, pseudotime='pseudotime', basis='umap', adj_key='distances', ekey='M_s', vkey='velocity_S', add_tkey='pseudotime_transition_matrix', add_ukey='M_u_pseudo', method='hodge', dynamics_info=False, unspliced_RNA=False) -> None
```

Observed `method` branches:

- `hodge`
- `gradient`
- `naive`

Important source details:

- `adata.obsp[adj_key]` must exist before the call
- `adata.layers[ekey].toarray()` is called internally, so a dense `numpy.ndarray` layer alias will fail
- `dynamics_info=True` adds `adata.uns['dynamics']` with `experiment_type='conventional'`, `has_splicing=True`, `use_smoothed=True`, and `est_method='conventional'`
- `unspliced_RNA=True` stores pseudo-unspliced output under `add_ukey`, default `M_u_pseudo`, and adds `pseudotime_vel_params`

### `dyn.vf.VectorField`

Observed signature:

```python
(adata, basis=None, layer=None, ..., method='SparseVFC', pot_curl_div=False, **kwargs)
```

Observed `method` branches:

- `SparseVFC`
- `dynode`

Important source details:

- notebook argument `M=1000` is not part of the public signature
- current source forwards `M` through `**kwargs`
- `pot_curl_div=True` runs `ddhodge(...)`, then curl for 2D bases, then divergence

### `dyn.pd.fate`

Observed signature includes these branch-heavy parameters:

- `direction`: `forward`, `backward`, `both`
- `average`: `False`, `True`, `origin`, `trajectory`
- `sampling`: `arc_length`, `logspace`, `uniform_indices`
- `inverse_transform`: `False`, `True`

Observed runtime storage rule:

- `basis='umap'` stores fate outputs under `adata.uns['fate_umap']`

### Visualization / Animation Entry Points

Observed signatures:

- `dyn.pl.topography(...)`
- `dyn.mv.StreamFuncAnim(...)`
- `dyn.pl.compute_velocity_on_grid(...)`
- `dyn.pl.animate_streamplot(...)`

Important preconditions:

- `topography(...)` is most useful after `VectorField(..., pot_curl_div=True)`
- `StreamFuncAnim(...)` expects prior `fate(...)` output
- `animate_streamplot(...)` operates on precomputed `X_grid`, `V_grid`

## Empirical Checks Run

### Sample Data Load

Observed on the current bone marrow worked example:

- shape is `5780 x 27876`
- `.obs` includes `clusters`, `palantir_pseudotime`, `palantir_diff_potential`
- `.obsm` includes `X_tsne`
- `.layers` includes `spliced`, `unspliced`
- `adata.X` is a `csr_matrix`
- observation names are already unique

Worked-example note:

- these metadata names are useful for reproducing `201_dynamo_beyondvelo.ipynb`
- they are not part of the generic input contract for the skill

### Neighbor Graph Materialization Check

Checked on a `160`-cell subset across `HSC_1`, `HSC_2`, `Ery_1`, and `Mono_1`:

1. `preprocess_adata(..., recipe='monocle')`
2. `reduceDimension(...)`

Observed result:

- `adata.uns['neighbors']` existed
- `adata.obsp` was still empty
- explicit `dyn.tl.neighbors(adata, basis='pca')` then created `obsp['distances']` and `obsp['connectivities']`

### Core Pseudotime Path

Checked on a `160`-cell subset after the neighbor-graph fix:

1. `preprocess_adata(..., recipe='monocle')`
2. `reduceDimension(...)`
3. `neighbors(adata, basis='pca')`
4. `adata.layers['M_s'] = csr_matrix-backed expression`
5. `pseudotime_velocity(..., pseudotime='palantir_pseudotime', method='hodge', dynamics_info=True)`

Observed outputs:

- `X_pca`
- `X_umap`
- `velocity_umap`
- `velocity_S`
- `pseudotime_transition_matrix`
- `use_for_dynamics`
- `use_for_transition`
- `dynamics`

### Vector Field, Potential, And Fate Branch

Checked on the same `160`-cell subset after the core path:

1. `VectorField(..., basis='umap', M=100, pot_curl_div=True, cores=1)`
2. `pd.fate(..., basis='umap', direction='forward', average=False, inverse_transform=False, cores=1)`

Observed outputs:

- `adata.uns['VecFld_umap']`
- `adata.obs['obs_vf_angle_umap']`
- `adata.obs['umap_ddhodge_potential']`
- `adata.obs['curl_umap']`
- `adata.obs['divergence_umap']`
- `adata.uns['fate_umap']`
- `adata.uns['fate_umap']` includes `t` and `prediction`

### Alternate `gradient` Branch

Checked on a `90`-cell subset across `HSC_1`, `HSC_2`, and `Ery_1`:

1. `preprocess_adata(..., recipe='monocle')`
2. `reduceDimension(...)`
3. `neighbors(adata, basis='pca')`
4. `pseudotime_velocity(..., method='gradient', dynamics_info=False)`

Observed outputs:

- `velocity_umap`
- `velocity_S`
- `pseudotime_transition_matrix`
- `dynamics` remained absent, matching `dynamics_info=False`

## Human Review Notes

- The stable reusable job is pseudotime-to-velocity conversion, not "run the bone marrow notebook."
- The strongest compatibility trap here is the missing `obsp['distances']` after `reduceDimension(...)`, not the pseudotime method call itself.
- The second strongest trap is that `ekey` must be sparse-backed or `.toarray()`-compatible.
