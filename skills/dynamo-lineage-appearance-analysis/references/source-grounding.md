# Source Grounding

This skill was generated from notebook inspection, live interface inspection, source-code reads, and bounded empirical execution on real `dynamo` sample data.

## Inspection Notes

Live inspection used the repository's current source and a local Python runtime compatible with this repository for callable inspection and smoke execution.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/400_tutorial_hsc_dynamo_megakaryocytes_appearance.ipynb`

Code inspected:

- `dynamo/vectorfield/vector_calculus.py`
- `dynamo/plot/topography.py`
- `dynamo/tools/graph_operators.py`
- `dynamo/sample_data.py`

## Inspected Interfaces

### `dyn.sample_data.hematopoiesis`

Observed signature:

```python
(url='https://figshare.com/ndownloader/files/47439635', filename='hematopoiesis.h5ad')
```

Observed runtime note:

- the loader uses a processed worked example and may download `hematopoiesis.h5ad` into `./data/`

### `dyn.vf.topography`

Observed signature:

```python
(adata, basis='umap', layer=None, X=None, dims=None, n=25, VecFld=None, show_progress=False, **kwargs)
```

Observed behavior:

- updates `adata.uns['VecFld_' + basis]`
- is the remapping step when fixed points need recalculation

### `dyn.pl.topography`

Observed signature includes:

```python
(..., basis='umap', fps_basis='umap', terms=['streamline', 'fixed_points'], quiver_source='raw', fate='both', n=25, ...)
```

Observed branch details from current source:

- `quiver_source`: `raw`, `reconstructed`
- `fate`: `history`, `future`, `both`
- `terms` is normalized into a list and `trajectory` is appended automatically when `init_cells` or `init_states` is provided
- integration direction maps to `backward`, `forward`, or `both` depending on `fate`

Observed runtime trap:

- `quiver_source='reconstructed'` currently hits `ImportError: cannot import name 'vector_field_function' from 'dynamo.tools.utils'`

Observed source trap:

- the docstring advertises `separatrix`, but the implementation checks for `separatrices`

### `build_graph`, `div`, `potential`

Observed signatures:

```python
build_graph(adj_mat)
div(g)
potential(g, div_neg=None)
```

Observed source details:

- `build_graph(...)` uses `Graph.Weighted_Adjacency(adj_mat)`
- `potential(...)` defaults to `-div(g)` when `div_neg` is omitted
- the notebook intentionally passes different signs for the cosine and Fokker-Planck branches

### `dyn.vf.jacobian`

Observed signature includes:

```python
(..., sampling=None, sample_ncells=1000, basis='pca', method='analytical', store_in_adata=True, **kwargs)
```

Observed branch details:

- `sampling`: `None`, `random`, `velocity`, `trn`
- `method`: `analytical`, `numerical`

Observed source details:

- if `basis='umap'`, current code forces `cell_idx = np.arange(adata.n_obs)`
- when regulators or effectors are strings matching a boolean `adata.var` column, they expand to the flagged genes
- for `basis='pca'`, the selected Jacobian is inverse-transformed through `adata.uns['PCs']`

Observed output structure:

- `adata.uns['jacobian_pca']` contains `jacobian`, `cell_idx`, `regulators`, `effectors`, and `jacobian_gene`

### `dyn.pl.jacobian`

Observed signature includes:

```python
(..., basis='umap', jkey='jacobian', j_basis='pca', layer='M_s', cmap='bwr', sort='abs', ...)
```

Observed runtime detail:

- `save_show_or_return='return'` returned a `GridSpec` on the worked example

### `dyn.vf.speed`

Observed signature:

```python
(adata, basis='umap', vector_field_class=None, method='analytical')
```

Observed branch detail:

- `method='analytical'` evaluates the vector field directly
- non-analytical mode falls back to stored velocity vectors such as `adata.obsm['velocity_' + basis]`

### `dyn.vf.divergence`

Observed signature includes:

```python
(..., sampling=None, sample_ncells=1000, basis='pca', method='analytical', store_in_adata=True, **kwargs)
```

Observed branch details:

- `sampling`: `None`, `random`, `velocity`, `trn`
- `method`: `analytical`, `numeric`

Observed source detail:

- if `jacobian_pca` is already present, current code reuses those entries before computing any missing divergence values

### `dyn.vf.acceleration`

Observed signature:

```python
(adata, basis='umap', vector_field_class=None, Qkey='PCs', method='analytical', **kwargs)
```

Observed branch details:

- `method`: `analytical`, `numerical`
- for `basis='pca'`, the function creates both `obs['acceleration_pca']`, `obsm['acceleration_pca']`, and a high-dimensional `layers['acceleration']`

### `dyn.vf.curvature`

Observed signature:

```python
(adata, basis='pca', vector_field_class=None, formula=2, Qkey='PCs', method='analytical', **kwargs)
```

Observed branch details:

- `formula`: `1`, `2`
- `method`: `analytical`, `numerical`

Observed runtime trap:

- `formula=1` failed in reviewer execution because `curv_mat` was `None` while the function still wrote it into `adata.obsm[curvature_key]`

## Empirical Checks Run

### Worked-Example Data Contract

Observed on `dyn.sample_data.hematopoiesis()`:

- shape `1947 x 1956`
- `obsp` includes `cosine_transition_matrix` and `fp_transition_rate`
- `uns` includes `VecFld_pca`, `VecFld_umap`, and `PCs`
- `obsm` includes `X_umap`, `velocity_umap`, `acceleration_pca`, and `curvature_pca`

### Core Analysis Smoke

A bounded local smoke used the shipped hematopoiesis example and successfully:

1. built cosine and Fokker-Planck graphs
2. computed `cosine_potential`, `potential_fp`, and `pseudotime_fp`
3. ran `dyn.vf.jacobian(...)` on `FLI1` and `KLF1`
4. plotted `dyn.pl.jacobian(..., save_show_or_return='return')`
5. ran `speed`, `divergence`, `acceleration`, and `curvature(formula=2)` in `basis='pca'`
6. plotted `dyn.pl.topography(..., save_show_or_return='return')`

Observed outputs:

- `jacobian_plot_return` was `GridSpec`
- `topography_return` was `Axes`
- `speed_pca`, `divergence_pca`, `acceleration_pca`, and `curvature_pca` existed
- `VecFld_umap` remained present

### Branch Diagnostics

Additional local diagnostics also checked two branch-heavy runtime paths:

- `dyn.vf.curvature(..., formula=1)` failed in the current runtime
- `dyn.pl.topography(..., quiver_source='reconstructed')` failed in the current runtime

Those branches are documented as compatibility risks instead of recommended defaults.
