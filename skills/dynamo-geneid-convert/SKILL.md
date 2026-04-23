---
name: dynamo-geneid-convert
description: Convert Ensembl-style gene IDs to gene symbols in `dynamo` with `dynamo.preprocessing.convert2gene_symbol` or `dynamo.preprocessing.convert2symbol`, including human and zebrafish IDs, version-suffix stripping, `AnnData.var_names` updates, and optional preprocessing handoff. Use when adapting `docs/tutorials/notebooks/110_geneid_convert_tutorial.ipynb`, standardizing `adata.var_names`, mapping Ensembl IDs to symbols, or doing identifier cleanup before a `Preprocessor` recipe.
---

# Dynamo Gene ID Convert

## Goal

Convert Ensembl-style identifiers to gene symbols in a way that another agent can actually rerun: choose the correct conversion path, preserve traceability columns, handle species and version-suffix edge cases, and only hand off to `Preprocessor` after IDs are standardized.

## Quick Workflow

1. Inspect `adata.var_names` or the raw ID list and determine whether the user needs a batch table or an in-place `AnnData` update.
2. Strip version suffixes such as `.1` into a `query` column before you treat mapping quality as final.
3. Use `convert2gene_symbol(...)` when you need an explicit mapping table, reproducible merge logic, or non-human species control.
4. Use `convert2symbol(adata, ...)` only when in-place `AnnData` mutation is the right abstraction and the prefix / `scopes` rules are satisfied.
5. Keep the original identifier in `adata.var`, set `adata.var_names` from `symbol` only after validating duplicates and missing mappings, and run preprocessing afterward, not before.

## Interface Summary

- `convert2gene_symbol(input_names, scopes='ensembl.gene', ensembl_release=None, species=None, force_rebuild=False)` returns a DataFrame indexed by `query` with at least `symbol` and `_score`.
- The live source uses vendored `pyensembl` data access, not a remote MyGene batch API.
- `convert2symbol(adata, scopes=None, subset=True)` updates `adata.var`, adds `query` plus `symbol`, and can rewrite `adata.var_names` in place.
- `Preprocessor.preprocess_adata(adata, recipe='monocle', tkey=None, experiment_type=None)` is only the downstream handoff. Current source exposes five `recipe` branches:
  `monocle`, `seurat`, `sctransform`, `pearson_residuals`, `monocle_pearson_residuals`.

Read `references/source-grounding.md` before documenting parameters more narrowly than the notebook does.

## Conversion Path Selection

- Use `convert2gene_symbol(...)` as the default for notebook conversion work.
  It is the safest path when you need explicit `species`, `ensembl_release`, manual merge logic, or per-ID validation.
- Use `convert2symbol(adata)` for human or mouse Ensembl-style `adata.var_names` when direct in-place conversion is acceptable.
- Use `convert2symbol(adata, scopes='ensembl.gene')` for zebrafish or other non-human prefixes if you still want the in-place helper.
- Do not trust the notebook's generic `scopes` explanation alone.
  In current source, `convert2gene_symbol` keeps `scopes` only for compatibility, while `convert2symbol` still branches on `scopes` and prefix heuristics.

## Minimal Execution Patterns

For an explicit mapping table and controlled merge:

```python
import dynamo as dyn

adata = dyn.sample_data.hematopoiesis_raw()
adata.var["ensembl_id"] = adata.var_names
adata.var["query"] = adata.var_names.str.split(".").str[0]

mapping = dyn.preprocessing.convert2gene_symbol(
    adata.var["query"].tolist(),
    species="human",
)

adata.var = (
    adata.var
    .merge(mapping, left_on="query", right_index=True, how="left")
    .set_index(adata.var.index)
)

mapped = adata.var["symbol"].notna()
adata = adata[:, mapped].copy()
adata.var_names = adata.var["symbol"].astype(str)
```

For in-place conversion on supported prefixes:

```python
import dynamo as dyn

adata = dyn.sample_data.hematopoiesis_raw()
adata.var["ensembl_id"] = adata.var_names
dyn.preprocessing.convert2symbol(adata, subset=True)
```

For zebrafish IDs with version suffixes:

```python
import dynamo as dyn

result = dyn.preprocessing.convert2gene_symbol(
    ["ENSDARG00000035558.1"],
    ensembl_release=77,
)

# Or, if the user wants in-place AnnData mutation:
dyn.preprocessing.convert2symbol(adata, scopes="ensembl.gene")
```

## Optional Preprocess Integration

- Perform gene-ID conversion before preprocessing unless the user explicitly wants to preserve notebook timing.
- If the user proceeds into preprocessing, default to `recipe='monocle'` unless they ask for a different branch.
- If the user requests Pearson residuals but still cares about downstream velocity-safe layers, prefer `monocle_pearson_residuals`.

Read `references/preprocess-handoff.md` before choosing a non-default `recipe`.

## Validation

After conversion, check these items:

- `adata.var["query"]` stores the version-stripped identifier actually used for lookup.
- `adata.var["symbol"]` exists and the mapping rate is acceptable for the dataset.
- The original ID remains preserved, for example in `adata.var["ensembl_id"]`.
- `adata.var_names` contains symbols only after duplicate and null handling is explicit.
- Representative conversions still match live source behavior:
  `ENSG00000141510 -> TP53` and `ENSDARG00000035558(.1) -> gps2` with `ensembl_release=77`.
- If preprocessing follows, `Preprocessor.preprocess_adata(..., recipe=...)` should run only after symbol assignment is settled.

## Constraints

- Do not describe conversion as MyGene-backed just because older prose or notebook wording suggests that pattern; current source uses vendored `pyensembl`.
- Do not assume `convert2symbol(adata)` auto-detects every species. In current source it auto-classifies some human / mouse gene or transcript prefixes, but zebrafish without explicit `scopes` raises.
- Do not assume the docstring's `ensembl_release` default is authoritative. Current code assigns `77` when the argument is omitted.
- The first conversion run may install `polars` or download / index Ensembl data, so cold-start execution can be slower than the notebook suggests.

## Resource Map

- Read `references/source-grounding.md` for inspected signatures, live-source behavior, and branch coverage.
- Read `references/conversion-paths.md` for human vs zebrafish decision rules and `scopes` handling.
- Read `references/preprocess-handoff.md` for the downstream `recipe` branches.
- Read `references/source-notebook-map.md` to see which notebook sections were preserved or intentionally dropped.
- Read `references/compatibility.md` when notebook wording and current source behavior appear to disagree.
