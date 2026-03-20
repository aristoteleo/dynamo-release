# Source Notebook Map

Source notebook:
`docs/tutorials/notebooks/100_tutorial_preprocess.ipynb`

Use this file to trace which notebook sections were preserved, condensed, or intentionally dropped.

## Notebook Sections To Skill Responsibilities

### 1. Tutorial overview and glossary

Notebook purpose:

- introduce the `Preprocessor` workflow
- explain preprocessing-generated keys
- position the current class API relative to older recipe helpers

Preserved in the skill:

- `SKILL.md` Goal
- `SKILL.md` Validation
- `references/compatibility.md`

Important notebook keys carried forward:

- `obs["pass_basic_filter"]`
- `var["pass_basic_filter"]`
- `var["use_for_pca"]`
- `obsm["X_pca"]`
- recipe-dependent normalized layers such as `X_spliced` and `X_unspliced`

### 2. Wrapper-based recipe usage

Notebook cells show:

- loading `dyn.sample_data.zebrafish()`
- importing `Preprocessor`
- calling `preprocess_adata(..., recipe=...)`
- running PCA-based UMAP afterward

Preserved in the skill:

- `SKILL.md` Quick Workflow
- `SKILL.md` Minimal Execution Patterns
- `references/recipe-selection.md`

### 3. Recipe comparisons

Notebook explicitly demonstrates:

- `monocle`
- `pearson_residuals`
- `sctransform`
- `seurat`

Additional source-grounded branch added beyond the notebook:

- `monocle_pearson_residuals`

Reason:

- current source dispatches this branch in `Preprocessor.preprocess_adata(...)`
- omitting it would make the generated skill incomplete

### 4. Customizing recipe kwargs

Notebook demonstrates mutating:

- `filter_cells_by_outliers_kwargs`
- `filter_genes_by_outliers_kwargs`
- `select_genes_kwargs`

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns

Preserved rule:

- configure a recipe first
- mutate kwargs second
- run the matching recipe-specific method third

### 5. Running each step directly

Notebook demonstrates:

- `standardize_adata`
- `filter_cells_by_outliers`
- `filter_genes_by_outliers`
- `normalize_by_cells`
- `select_genes`
- `norm_method`
- `pca`
- `dyn.tl.reduceDimension(...)`

Preserved in the skill:

- `SKILL.md` Minimal Execution Patterns
- `references/recipe-selection.md`

Preserved rule:

- use the stepwise path for custom pipelines and debugging
- use the wrapper for the normal case

## Source-Grounded Additions Beyond The Notebook

These details were added from live source and interface inspection, not just from notebook cells:

- the exact `recipe` signature and default on `Preprocessor.preprocess_adata(...)`
- the extra source-level `recipe` branch `monocle_pearson_residuals`
- the constructor signature showing extensive callable and kwargs injection points
- `config_*_recipe(...)` signatures and their role as configuration entrypoints

## What Was Intentionally Not Carried Over

- plotting style setup
- long pedagogical prose
- repeated UMAP cells whose only purpose is visualization parity
- full output logs and rendered figures
- every constructor kwarg from `Preprocessor.__init__`, because most are not needed for the notebook-derived stable job

## When To Reopen The Notebook

Reopen the notebook only when:

- the user wants figure parity or cell-by-cell reproduction
- the user asks for one of the exact kwargs blocks not preserved in the skill
- the source code and notebook appear to disagree
- a future notebook revision adds or removes preprocessing branches
