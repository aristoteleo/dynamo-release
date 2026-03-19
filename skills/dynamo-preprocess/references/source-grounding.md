# Source Grounding

This skill was not written from notebook prose alone. The concrete interface claims below were checked against live source and runtime inspection.

## Inspection Notes

Interface inspection and smoke checks were performed in a local Python runtime compatible with this repository's dependencies.

## Primary Source Files

Code inspected in:

- `dynamo/preprocessing/Preprocessor.py`

Notebook inspected in:

- `docs/tutorials/notebooks/100_tutorial_preprocess.ipynb`

## Inspected Signatures

### `Preprocessor.__init__`

Observed signature summary:

- exposes overridable callables and kwargs dictionaries for filtering, normalization, feature selection, PCA, gene list forcing, sctransform, regress-out, and cell-cycle scoring
- default constructor behavior is monocle-oriented

Why this matters:

- the notebook only mutates a few kwargs dictionaries
- the full constructor is much broader, so the skill intentionally documents only the stable notebook-relevant subset

### `Preprocessor.preprocess_adata`

Inspected signature:

```python
(self, adata, recipe='monocle', tkey=None, experiment_type=None) -> None
```

Inspected annotation details:

- `recipe` is a `Literal` over:
  `monocle`, `seurat`, `sctransform`, `pearson_residuals`, `monocle_pearson_residuals`
- default is `monocle`

Source-level branch scan:

- `if recipe == "monocle"`
- `elif recipe == "seurat"`
- `elif recipe == "sctransform"`
- `elif recipe == "pearson_residuals"`
- `elif recipe == "monocle_pearson_residuals"`

Interpretation:

- `recipe` is the main capability selector for this workflow even though the notebook only demonstrates four branches explicitly

### Recipe configuration entrypoints

Inspected signatures:

```python
config_monocle_recipe(self, adata, n_top_genes=2000) -> None
config_seurat_recipe(self, adata) -> None
config_sctransform_recipe(self, adata) -> None
config_pearson_residuals_recipe(self, adata) -> None
config_monocle_pearson_residuals_recipe(self, adata) -> None
```

First-line docstrings observed:

- monocle: automatically configure for monocle recipe
- seurat: automatically configure for seurat style recipe
- sctransform: automatically configure for sctransform style recipe
- pearson residuals: automatically configure for Pearson residuals style recipe
- monocle + Pearson residuals: automatically configure for the hybrid recipe

## Source-Derived Behavior Notes

### `monocle`

Observed source-specific details:

- computes size factors before normalization
- runs `log1p`
- computes PCA
- also computes new-to-total ratio and cell-cycle score

### `seurat`

Observed source-specific details:

- starts from monocle config
- swaps gene selection to `select_genes_by_seurat_recipe`
- uses `algorithm='seurat_dispersion'`

### `sctransform`

Observed source-specific details:

- subsets genes first for efficiency
- warns that this recipe wrapper imposes that preliminary step
- expects `KDEpy` at runtime

### `pearson_residuals`

Observed source-specific details:

- disables standard outlier filtering functions
- uses `normalize_layers_pearson_residuals`
- treats `adata.X` as the normalization target

### `monocle_pearson_residuals`

Observed source-specific details:

- exists in live source even though the notebook does not make it a main section
- combines Pearson-residual feature selection and PCA with monocle-style normalization for downstream velocity-safe layers

## Human Review Notes

- The helper script originally emphasized `method` / `backend` / `mode` branch names. For this workflow, `recipe` is the critical branch parameter and was reviewed explicitly from source.
- The generated skill therefore documents `recipe` as a first-class branch selector instead of inheriting only the notebook's demonstrated path.
