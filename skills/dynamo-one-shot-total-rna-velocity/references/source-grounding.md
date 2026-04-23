# Source Grounding

This skill was generated from live code inspection and empirical execution on the hematopoiesis worked example, not from notebook prose alone.

## Inspection Notes

Source inspection and empirical checks were performed in a local Python runtime compatible with this repository's dependencies.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/301_tutorial_hsc_velocity.ipynb`

Code inspected:

- `dynamo.preprocessing.Preprocessor`
- `dynamo.tools.moments`
- `dynamo.tools.dynamics`
- `dynamo.tools.cell_velocities`
- `dynamo.plot.scatters`

## Inspected Interfaces

### `Preprocessor.config_monocle_recipe`

Observed signature:

```python
(self, adata, n_top_genes=2000) -> None
```

### `Preprocessor.preprocess_adata_monocle`

Observed signature:

```python
(self, adata, tkey=None, experiment_type=None)
```

### `Preprocessor.preprocess_adata`

Observed signature:

```python
(self, adata, recipe='monocle', tkey=None, experiment_type=None)
```

Observed `recipe` branches in current source:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

### `dyn.tl.moments`

Observed signature:

```python
(adata, X_data=None, genes=None, group=None, conn=None, use_gaussian_kernel=False, normalize=True, use_mnn=False, layers='all', n_pca_components=30, n_neighbors=30, copy=False)
```

Important source detail:

- the function writes `adata.obsp["moments_con"]`
- for this labeling workflow it also materializes `M_n`, `M_t`, `M_s`, and related second-moment layers

### `dyn.tl.dynamics`

Observed signature includes:

```python
(adata, ..., model='auto', est_method='auto', ..., group=None, ..., one_shot_method='combined', ...)
```

Observed `one_shot_method` branches:

- `combined`
- `sci-fate`
- `sci_fate`

Observed source-level `model` branches include:

- `deterministic`
- `stochastic`
- `mixture`
- `model_selection`

Important source details:

- the docstring describes `sci-fate` as the direct first-order decay fit and `combined` as the steady-state-plus-absolute-gamma route
- the one-shot branch choice is not stored back into `adata.uns["dynamics"]` on the checked runtime

### `calculate_velocity_alpha_minus_gamma_s`

Observed signature:

```python
(adata, gene_subset_key='use_for_pca', velocity_layer_name='velocity_alpha_minus_gamma_s')
```

Important source details:

- requires `gene_subset_key` to exist in `adata.var`
- hard-codes `adata.obs.time.astype(float)`
- reads `adata[:, pca_genes].layers["M_n"]`
- reads `adata.layers["M_s"][:, pca_genes_array]`
- looks up time-specific gamma parameters and writes `adata.layers[velocity_layer_name]`

### `dyn.tl.cell_velocities`

Observed signature includes:

```python
(adata, ..., X=None, V=None, basis='umap', adj_key='distances', ..., method='pearson', ...)
```

Observed `method` branches:

- `kmc`
- `fp`
- `cosine`
- `pearson`
- `transform`

Important source details:

- `method in ['pearson', 'cosine']` stores `adata.obsp["<method>_transition_matrix"]`
- `method='transform'` depends on `adata.uns["umap_fit"]`

### `dyn.pl.streamline_plot`

Observed signature includes:

```python
(..., basis='umap', method='gaussian', ...)
```

Observed `method` branches:

- `gaussian`
- `SparseVFC`

### `dyn.pl.phase_portraits`

Observed signature includes:

```python
(adata, genes, ..., vkey=None, ekey=None, basis='umap', ..., use_smoothed=True, ...)
```

## Empirical Checks Run

### Sample Data Load

Observed on current `dyn.sample_data.hematopoiesis_raw()`:

- shape is `1947 x 26193`
- `.obs` includes `batch`, `cell_type`, `time`
- `.layers` includes `new`, `spliced`, `total`, `unspliced`
- `.obsm` includes `X_umap`
- `.uns` includes `genes_to_use`
- `genes_to_use` length is `2552`
- unique `time` values are `3` and `5`

Worked-example note:

- these metadata names are useful for reproducing the tutorial
- they are not universal assumptions for all one-shot labeling inputs

### Notebook-Like Core Path

Checked on a `224`-cell subset across `HSC`, `MEP-like`, and `Mon` at time `3` and `5`:

1. preserve `raw.uns["genes_to_use"]` before concatenation
2. `Preprocessor(force_gene_list=selected)`
3. `config_monocle_recipe(..., n_top_genes=len(selected))`
4. `preprocess_adata_monocle(..., tkey='time', experiment_type='one-shot')`
5. `reduceDimension(...)`
6. `moments(..., group='time')`
7. set `adata.uns["pp"]["has_splicing"] = False`
8. `dynamics(..., group='time', one_shot_method='sci_fate', model='deterministic')`
9. `calculate_velocity_alpha_minus_gamma_s(...)`
10. `cell_velocities(..., X=M_t, V=velocity_alpha_minus_gamma_s, method='cosine')`

Observed outputs:

- concatenation dropped `uns["genes_to_use"]` until it was restored manually
- preprocessing started from the `2552`-gene curated list, reported `1925` genes in use, and ended with `1292` `use_for_pca` genes
- `reduceDimension(...)` reused the existing UMAP basis and created `obsp["connectivities"]` and `obsp["distances"]`
- `moments(...)` created `M_n`, `M_t`, `M_s`, related second moments, and `obsp["moments_con"]`
- `dynamics(...)` created `velocity_N`, `velocity_T`, `use_for_dynamics`, and time-specific parameter-name keys
- `dynamics(...)` did not create `velocity_S`
- `calculate_velocity_alpha_minus_gamma_s(...)` created `adata.layers["velocity_alpha_minus_gamma_s"]` and processed both time points
- `cell_velocities(..., method='cosine')` created `adata.obsm["velocity_umap"]` and `adata.obsp["cosine_transition_matrix"]`

Observed runtime warning with this path:

- `641` genes were removed because of NaN velocity values before the cosine projection

### Alternate `combined` One-Shot Branch

Checked on an `80`-cell subset across `HSC` and `MEP-like` at time `3` and `5`:

1. same preprocessing and grouped-moments path
2. set `adata.uns["pp"]["has_splicing"] = False`
3. `dynamics(..., group='time', one_shot_method='combined', model='deterministic')`

Observed outputs:

- `velocity_T` existed
- `adata.uns["dynamics"]["model"] == "deterministic"`
- `adata.uns["dynamics"]["est_method"] == "ols"`

### Alternate `pearson` Projection Branch

Checked on an `80`-cell subset across `HSC` and `MEP-like` at time `3` and `5`:

1. same `sci_fate` deterministic total-RNA path through `calculate_velocity_alpha_minus_gamma_s(...)`
2. `cell_velocities(..., X=M_t, V=velocity_alpha_minus_gamma_s, method='pearson')`

Observed outputs:

- `adata.obsm["velocity_umap"]`
- `adata.obsp["pearson_transition_matrix"]`

### Small-Group Warning

Observed on the checked `80`-cell branch:

- `dynamics(...)` warned that groups with fewer than `50` cells may produce all-NaN velocities for some cells and downstream issues

## Human Review Notes

- The stable reusable job is one-shot total RNA velocity, not "run the HSC notebook."
- The highest-signal compatibility traps are preserving `genes_to_use`, forcing `has_splicing=False` for the Model-2 branch, and keeping the time column under the literal key `time`.
- The notebook's real endpoint is the custom total-RNA projection branch, not the raw `dynamics(...)` output alone.
