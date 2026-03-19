# Preprocess Handoff

This notebook is mainly about identifier conversion, but the last section hands the result into preprocessing. Use this reference so the generated skill does not freeze that handoff to one notebook path.

## Rule

Finish gene-ID conversion before preprocessing unless the user explicitly asks for historical notebook order.

Why:

- `standardize_adata(...)` inside preprocessing calls the configured gene-name conversion function
- changing `var_names` after preprocessing can invalidate assumptions in downstream cached results
- the notebook itself warns that conversion is best done before the main preprocessing pipeline

## Downstream `recipe` Branches

Current source for `Preprocessor.preprocess_adata(...)` dispatches these values:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

## Default Choice

Use `recipe='monocle'` unless the user asks for a different preprocessing objective.

## Branch Summary

### `monocle`

Use when:

- the user wants standard dynamo preprocessing after conversion
- downstream work includes velocity or vector-field analysis

### `seurat`

Use when:

- the user explicitly wants Seurat-style HVG selection inside the current `Preprocessor`

### `sctransform`

Use when:

- the user explicitly requests sctransform
- the runtime can support the extra dependencies and preprocessing cost

### `pearson_residuals`

Use when:

- the user mainly wants Pearson-residual feature selection and PCA

### `monocle_pearson_residuals`

Use when:

- the user wants Pearson-residual feature selection and PCA
- the user still needs monocle-style normalized layers for downstream velocity-safe workflows

## Minimal Handoff Pattern

```python
from dynamo.preprocessing import Preprocessor

preprocessor = Preprocessor()
preprocessor.preprocess_adata(adata, recipe="monocle", tkey="time", experiment_type="one-shot")
```

## Validation After Handoff

At minimum, check:

- `adata.var_names` already reflects the intended symbol space
- the chosen `recipe` was explicit
- preprocessing no longer needs to rename genes implicitly
