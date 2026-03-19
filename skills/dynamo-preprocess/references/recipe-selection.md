# Recipe Selection

This reference summarizes the `recipe` branches used by `dynamo.preprocessing.Preprocessor.preprocess_adata(...)` for `docs/tutorials/notebooks/100_tutorial_preprocess.ipynb`.

## Source-Grounded Branch List

The live source for `Preprocessor.preprocess_adata(...)` dispatches `recipe` across five values:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

The notebook explicitly demonstrates the first four. The fifth branch, `monocle_pearson_residuals`, is in current source and should not be omitted when documenting the workflow.

## Default Choice

Use `monocle` unless the user has a concrete reason to choose a different branch.

It is the safest general-purpose preprocessing route when the downstream goal includes velocity, vector field analysis, or layer-preserving normalization.

## Branch Summary

### `monocle`

Use when:

- the user wants the standard dynamo preprocessing path
- downstream work includes velocity or vector field analysis
- layer-preserving normalization matters

Core source-level sequence:

1. `standardize_adata`
2. filter cells
3. filter genes
4. size-factor calculation
5. normalize by cells
6. select genes
7. append / exclude / force gene lists
8. `log1p`
9. optional regress-out
10. PCA
11. new-to-total ratio
12. cell cycle scoring

### `seurat`

Use when:

- the user wants Seurat-style HVG selection
- the user still wants a mostly monocle-shaped pipeline

Source-grounded differences from `monocle`:

- `config_seurat_recipe(...)` calls `config_monocle_recipe(...)` first
- then swaps `select_genes` to `select_genes_by_seurat_recipe`
- uses `algorithm="seurat_dispersion"`
- tightens `filter_genes_by_outliers_kwargs` to `{"shared_count": 20}`

### `sctransform`

Use when:

- the user explicitly requests sctransform
- the runtime has `KDEpy`

Important behavior:

- subsets genes for efficiency before running sctransform
- writes PCA with `n_pca_components=50`
- is not the safest default for velocity-preserving workflows

Important warning from source behavior:

- the recipe warns that if you want to avoid the built-in preliminary subsetting, you should run sctransform without this recipe wrapper

### `pearson_residuals`

Use when:

- the goal is Pearson-residual-driven HVG selection and PCA on `adata.X`
- the main objective is dimensionality reduction or exploratory preprocessing

Important behavior:

- disables standard cell and gene outlier filtering functions
- uses `select_genes_by_pearson_residuals`
- normalizes selected genes via `normalize_layers_pearson_residuals`
- filters cells by highly variable genes
- runs PCA with `n_pca_components=50`

### `monocle_pearson_residuals`

Use when:

- the user wants Pearson-residual feature selection and PCA
- the user still needs monocle-style normalized layers for downstream velocity and vector field workflows

Why it matters:

- source docstring explicitly says pure Pearson residual output can contain negative values that are undesirable for later RNA velocity analysis
- this hybrid branch preserves monocle normalization while still using Pearson residuals for feature selection and PCA

## Stepwise Customization

Use direct stepwise calls only when:

- the user wants to tune thresholds manually
- the user wants to replace part of the recipe
- the user wants debugging visibility at each stage

Typical manual sequence from the notebook:

1. `pp.standardize_adata(...)`
2. `pp.filter_cells_by_outliers(...)`
3. `pp.filter_genes_by_outliers(...)`
4. `pp.normalize_by_cells(...)`
5. `pp.select_genes(...)`
6. `pp.norm_method(...)`
7. `pp.pca(...)`

For monocle-style behavior, do not forget that the wrapped recipe also performs size-factor calculation before normalization.

## Common Validation Targets

Check these after execution:

- `adata.var["use_for_pca"]`
- `adata.obsm["X_pca"]`
- expected normalized layers for velocity-safe workflows
- successful `dyn.tl.reduceDimension(adata, basis="pca")`
- successful UMAP plotting with the chosen color key

## Decision Rule

If the user does not mention a branch:

- default to `monocle`

If the user asks for Pearson residuals but also cares about velocity layers:

- prefer `monocle_pearson_residuals`

If the user asks for notebook parity cell-by-cell:

- match the exact branch shown in the relevant notebook section
