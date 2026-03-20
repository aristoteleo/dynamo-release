---
name: dynamo-preprocess
description: Run or adapt dynamo preprocessing with `dynamo.preprocessing.Preprocessor`, including the `recipe` branches `monocle`, `seurat`, `sctransform`, `pearson_residuals`, and `monocle_pearson_residuals`. Use when converting or reproducing `docs/tutorials/notebooks/100_tutorial_preprocess.ipynb`, preprocessing an `AnnData` object for downstream dynamo analysis, customizing preprocessing kwargs, or translating notebook-level preprocessing into a reusable agent workflow.
---

# Dynamo Preprocess

## Goal

Preprocess an `AnnData` object with the current `dynamo.preprocessing.Preprocessor` API, choose the correct `recipe` branch for the downstream task, customize kwargs before execution when needed, and validate that the expected keys and embeddings exist for later dynamo analysis.

## Quick Workflow

1. Inspect the user's data, environment, and downstream goal.
2. Choose the `recipe` branch that matches the goal instead of defaulting blindly to the notebook's first example.
3. Use `Preprocessor.preprocess_adata(...)` for the common case.
4. If the user needs customization, call `config_*_recipe(...)`, mutate kwargs, then run the matching recipe-specific method.
5. If the user needs debugging or a custom pipeline, run the preprocessing steps individually in notebook order.
6. Validate `obs`, `var`, `layers`, and `obsm` keys before treating preprocessing as complete.

## Interface Summary

- `Preprocessor.preprocess_adata(adata, recipe='monocle', tkey=None, experiment_type=None)` is the main wrapper.
- The live source dispatches `recipe` across five branches:
  `monocle`, `seurat`, `sctransform`, `pearson_residuals`, `monocle_pearson_residuals`.
- `config_monocle_recipe(adata, n_top_genes=2000)` is the only recipe config in this notebook family that exposes an extra tuning argument directly in the signature.
- The constructor exposes many overridable callables and kwargs dictionaries, but the notebook mainly mutates:
  `filter_cells_by_outliers_kwargs`, `filter_genes_by_outliers_kwargs`, and `select_genes_kwargs`.

Read `references/source-grounding.md` before documenting parameters in more detail or when you need the exact inspected signatures.

## Recipe Selection

- Use `monocle` as the default when the goal is standard dynamo preprocessing for velocity or vector-field analysis.
- Use `seurat` when the user specifically wants Seurat-style highly variable gene selection inside the current `Preprocessor` wrapper.
- Use `sctransform` only when the user explicitly wants that transformation and the environment has `KDEpy`.
- Use `pearson_residuals` when the goal is HVG selection and PCA on `adata.X`, not layer-preserving velocity normalization.
- Use `monocle_pearson_residuals` when the user wants Pearson-residual-based feature selection and PCA but still needs monocle-style normalized layers for downstream velocity analysis.

Read `references/recipe-selection.md` before choosing a non-default `recipe`.

## Minimal Execution Patterns

For the common case, use the wrapper:

```python
import dynamo as dyn
from dynamo.preprocessing import Preprocessor

adata = dyn.sample_data.zebrafish()
preprocessor = Preprocessor()
preprocessor.preprocess_adata(adata, recipe="monocle")
dyn.tl.reduceDimension(adata, basis="pca")
```

For recipe customization, configure first, then mutate kwargs, then run the matching method:

```python
import numpy as np
from dynamo.preprocessing import Preprocessor

preprocessor = Preprocessor()
preprocessor.config_monocle_recipe(adata)

preprocessor.filter_cells_by_outliers_kwargs = {
    "filter_bool": None,
    "layer": "all",
    "min_expr_genes_s": 300,
    "min_expr_genes_u": 100,
    "min_expr_genes_p": 50,
    "max_expr_genes_s": np.inf,
    "max_expr_genes_u": np.inf,
    "max_expr_genes_p": np.inf,
    "shared_count": None,
}

preprocessor.select_genes_kwargs = {
    "n_top_genes": 2500,
    "sort_by": "cv_dispersion",
    "keep_filtered": True,
    "SVRs_kwargs": {
        "relative_expr": True,
        "total_szfactor": "total_Size_Factor",
        "min_expr_cells": 0,
        "min_expr_avg": 0,
        "max_expr_avg": np.inf,
        "winsorize": False,
        "winsor_perc": (1, 99.5),
        "sort_inverse": False,
        "svr_gamma": None,
    },
}

preprocessor.preprocess_adata_monocle(adata)
```

For direct stepwise execution, follow the notebook spine:

1. `pp.standardize_adata(adata, tkey, experiment_type)`
2. `pp.filter_cells_by_outliers(...)`
3. `pp.filter_genes_by_outliers(...)`
4. `pp.normalize_by_cells(...)`
5. `pp.select_genes(...)`
6. `pp.norm_method(...)`
7. `pp.pca(...)`
8. `dyn.tl.reduceDimension(adata, basis="pca")`

Use the stepwise path for debugging or nonstandard workflows, not as the default wrapper replacement.

## Validation

After preprocessing, check these items:

- `adata.obs` may contain QC keys such as `pass_basic_filter`, `Size_Factor`, `nGenes`, `nCounts`, and `pMito`.
- `adata.var` may contain `pass_basic_filter`, `use_for_pca`, and recipe-specific variability metrics.
- `adata.obsm["X_pca"]` should exist after PCA-based recipes.
- Layer-preserving workflows may create normalized layers such as `X_spliced` and `X_unspliced`.
- `dyn.tl.reduceDimension(adata, basis="pca")` should run without shape or key errors.

If notebook-style visualization parity matters, the standard post-check is:

```python
dyn.tl.reduceDimension(adata, basis="pca")
dyn.pl.umap(adata, color=celltype_key)
```

## Constraints

- Do not assume the notebook covers every `recipe` branch; the live source adds `monocle_pearson_residuals` as a distinct hybrid path.
- Pearson residuals and sctransform are intended for `adata.X`; using them on velocity-relevant layers can create downstream problems.
- `sctransform` requires `KDEpy`.
- `dyn.sample_data.zebrafish()` may require data download if the dataset is not cached.
- Prefer the current `dynamo.preprocessing.Preprocessor` API over older recipe helpers unless the user explicitly asks for legacy behavior.

## Resource Map

- Read `references/recipe-selection.md` for branch-by-branch guidance.
- Read `references/source-notebook-map.md` to trace each part of the notebook to the generated skill.
- Read `references/source-grounding.md` for inspected signatures, doc-derived notes, and source-level `recipe` branch evidence.
- Read `references/compatibility.md` when notebook prose and current source behavior may have drifted.
