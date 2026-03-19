# Compatibility

Use this reference when notebook wording and current source behavior diverge.

## Current Conversion Backend

Prefer the current source truth:

- `dynamo.preprocessing.convert2gene_symbol(...)` uses vendored `pyensembl`

Do not present the skill as if it depends on a generic MyGene batch-query service just because the notebook explains the idea that way.

## `ensembl_release` Drift

Observed mismatch:

- docstring says `ensembl_release=None` defaults to `109`
- current code assigns `77`

Conservative rule:

- if annotation-build reproducibility matters, always pass `ensembl_release` explicitly
- do not rely on the docstring default

## Species / Scope Drift

Observed mismatch between the two helpers:

- `convert2gene_symbol(...)` can infer zebrafish from `ENSDARG...`
- `convert2symbol(adata)` does not auto-select zebrafish scope and raises unless `scopes` is provided

Conservative rule:

- for zebrafish, use `convert2gene_symbol(...)` or call `convert2symbol(adata, scopes='ensembl.gene')`

## Cold-Start Runtime Costs

Observed during local validation:

- the first conversion may install `polars`
- first use may also download or index the required Ensembl data

Conservative rule:

- warn the user that the first successful run can be slower than later cached runs
- treat a cold-start delay as normal unless the command actually fails

## Preprocessing Integration

Prefer:

- convert IDs first
- preprocess second

Do not rename `adata.var_names` after preprocessing unless the user explicitly wants to repair an already-processed object and accepts the downstream risk.
