# Conversion Paths

This reference captures the branch-heavy parts of the notebook that are easy to miss if you only read the demonstrated cells.

## Default Choice

Use `convert2gene_symbol(...)` by default when converting notebook logic into a reusable workflow.

Why:

- it exposes `species`, `ensembl_release`, and `force_rebuild`
- it supports explicit merge and duplicate handling
- it works for zebrafish auto-detection directly
- it keeps the lookup table visible instead of mutating `AnnData` immediately

## Path A: Manual Table + Merge

Use when:

- the user wants auditability
- the dataset may contain duplicate mapped symbols
- you need a `query` column, mapping-rate check, or null filtering
- you want explicit `species` or `ensembl_release`

Recommended sequence:

1. Copy original IDs into `adata.var["ensembl_id"]`
2. Create `adata.var["query"] = adata.var_names.str.split(".").str[0]`
3. Call `convert2gene_symbol(adata.var["query"].tolist(), ...)`
4. Merge on `query`
5. Decide how to handle missing or duplicated `symbol`
6. Set `adata.var_names` from `symbol`

## Path B: `convert2symbol(adata, ...)`

Use when:

- the user already has an `AnnData`
- in-place mutation is acceptable
- the prefix / `scopes` behavior is understood

### Human / Mouse Gene IDs

Current source auto-detects some cases:

- `ENSG...` -> gene scope
- `ENSMUSG...` -> gene scope

For these, `convert2symbol(adata)` is usually enough.

### Human / Mouse Transcript IDs

Current source also recognizes:

- `ENST...`
- `ENSMUST...`

For these, `convert2symbol(adata)` auto-selects transcript scope.

### Zebrafish And Other Non-Human Prefixes

Do not rely on `convert2symbol(adata)` without `scopes`.

Observed runtime behavior:

- zebrafish `ENSDARG...` enters conversion mode because IDs start with `ENS`
- auto-scope detection then fails
- the helper raises and tells you to pass the correct scope

Recommended fix:

```python
dyn.preprocessing.convert2symbol(adata, scopes="ensembl.gene")
```

If the user also cares about annotation-build reproducibility, prefer the manual `convert2gene_symbol(...)` path with explicit `ensembl_release`.

## Version Suffix Handling

The notebook mentions suffix stripping. Current source really does strip version suffixes before lookup:

- `ENSG00000141510.5` is queried as `ENSG00000141510`
- `ENSDARG00000035558.1` is queried as `ENSDARG00000035558`

Still keep the original versioned ID in `adata.var["ensembl_id"]` when traceability matters.

## Supported Species In Current Source

`convert2gene_symbol(...)` explicitly handles these prefixes:

- `human`
- `mouse`
- `rat`
- `zebrafish`
- `fly`
- `chicken`
- `dog`
- `pig`
- `cow`
- `macaque`

If the prefix is unrecognized, the code warns and defaults to `human`.

## Decision Rule

If the user says only "convert gene IDs to symbols in this AnnData":

- use `convert2symbol(adata)` for human / mouse prefixes
- use `convert2symbol(adata, scopes="ensembl.gene")` for zebrafish if they want the helper
- otherwise prefer `convert2gene_symbol(...)` plus explicit merge

If the user says "make it reproducible across annotation builds":

- use `convert2gene_symbol(...)` with explicit `species` and `ensembl_release`
