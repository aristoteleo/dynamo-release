# Source Grounding

This skill was generated from live code inspection and empirical execution on the zebrafish worked example, not from notebook prose alone.

## Inspection Notes

Source inspection and empirical checks were performed in a local Python runtime compatible with this repository's dependencies.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/200_zebrafish.ipynb`

Code inspected:

- `dynamo/preprocessing/Preprocessor.py`
- `dynamo/tools/dynamics.py`
- `dynamo/tools/cell_velocities.py`
- `dynamo/vectorfield/VectorField.py`
- `dynamo/prediction/fate.py`
- `dynamo/movie/fate.py`
- `dynamo/plot/topography.py`
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

### `dyn.tl.dynamics`

Observed signature includes these branch-heavy parameters:

- `model`: `auto`, `deterministic`, `stochastic`
- `est_method`: `ols`, `rlm`, `ransac`, `gmm`, `negbin`, `auto`, `twostep`, `direct`
- `one_shot_method`: `combined`, `sci-fate`, `sci_fate`

Observed runtime on the zebrafish worked example after preprocessing:

- `experiment_type='conventional'`
- `model='stochastic'`
- `est_method='gmm'`
- `has_splicing=True`
- `has_labeling=False`

### `dyn.tl.cell_velocities`

Observed signature:

```python
(..., basis='umap', method='pearson', ...) -> AnnData
```

Observed `method` branches:

- `kmc`
- `fp`
- `cosine`
- `pearson`
- `transform`

Notebook-important runtime finding:

- after `method='pearson'`, the projected velocity is written to `adata.obsm["velocity_umap"]`
- the transition matrix is written to `adata.obsp["pearson_transition_matrix"]`
- there is no generic `adata.obsp["transition_matrix"]` in this branch by default

### `dyn.vf.VectorField`

Observed signature:

```python
(adata, basis=None, layer=None, ..., method='SparseVFC', pot_curl_div=False, **kwargs)
```

Observed `method` branches:

- `SparseVFC`
- `dynode`

Important source detail:

- notebook argument `M=1000` is not part of the public signature
- current source forwards `M` through `**kwargs`
- SparseVFC defaults are assembled in `_get_svc_default_arguments(...)`, where `M` defaults to `None`

Observed `pot_curl_div=True` behavior:

- runs `ddhodge(adata, basis=basis, cores=cores)`
- computes curl automatically for 2D bases
- computes divergence automatically

### `dyn.pd.fate`

Observed signature includes these branch-heavy parameters:

- `direction`: `forward`, `backward`, `both`
- `average`: `False`, `True`, `origin`, `trajectory`
- `sampling`: `arc_length`, `logspace`, `uniform_indices`
- `inverse_transform`: `False`, `True`

Observed runtime:

- `basis='umap'`, `direction='forward'`, `average=False` stores results under `adata.uns["fate_umap"]`

### Visualization / Animation Entry Points

Observed signatures:

- `dyn.pl.topography(...)`
- `dyn.mv.StreamFuncAnim(...)`
- `dyn.mv.animate_fates(...)`
- `dyn.pl.compute_velocity_on_grid(...)`
- `dyn.pl.animate_streamplot(...)`

Important preconditions:

- `animate_fates` expects prior `fate(...)` output
- `animate_streamplot` operates on precomputed `X_grid`, `V_grid`

## Empirical Checks Run

### Sample Data Load

Observed on the current zebrafish worked example:

- dataset downloads to `./data/zebrafish.h5ad` on first use
- shape is `4181 x 16940`
- raw sample contains `layers["spliced"]`, `layers["unspliced"]`
- raw sample exposes `Cell_type` in `.obs`
- raw sample does not expose `obsm["X_umap"]`
- observation names are not unique until fixed

Worked-example note:

- these metadata names are useful for reproducing `200_zebrafish.ipynb`
- they are not part of the generic input contract for the skill

### Core UMAP Workflow

Checked on a `300`-cell subset:

1. `obs_names_make_unique()`
2. `preprocess_adata(..., recipe='monocle')`
3. `dynamics(..., model='stochastic', cores=1)`
4. `reduceDimension(...)`
5. `cell_velocities(..., method='pearson')`
6. `VectorField(..., basis='umap', M=100, pot_curl_div=False)`
7. `fate(..., basis='umap', direction='forward', average=False, inverse_transform=False)`

Observed outputs:

- `X_pca`
- `velocity_S`
- `X_umap`
- `velocity_umap`
- `pearson_transition_matrix`
- `VecFld_umap`
- `fate_umap`

### Topology / Potential Branch

Checked on a `200`-cell subset with `pot_curl_div=True`:

Observed outputs:

- `adata.obs["umap_ddhodge_potential"]`
- `adata.obs["curl_umap"]`
- `adata.obs["divergence_umap"]`
- `adata.uns["VecFld_umap"]`

## Human Review Notes

- The notebook is end-to-end, but the stable reusable job is stage-based, not “run every cell”.
- The strongest compatibility traps here are data-shape assumptions, basis-dependent outputs, and kernel-specific storage names, not just method defaults.
