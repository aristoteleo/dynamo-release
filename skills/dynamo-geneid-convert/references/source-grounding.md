# Source Grounding

This skill was grounded against live source and runtime inspection instead of notebook prose alone.

## Inspection Notes

Source inspection and empirical checks were performed in a local Python runtime compatible with this repository's dependencies.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/110_geneid_convert_tutorial.ipynb`

Code inspected:

- `dynamo/preprocessing/utils.py`
- `dynamo/preprocessing/Preprocessor.py`

## Inspected Signatures

### `dynamo.preprocessing.convert2gene_symbol`

Observed signature:

```python
(input_names: List[str], scopes: Optional[List[str]] = 'ensembl.gene',
 ensembl_release: Optional[int] = None, species: Optional[str] = None,
 force_rebuild: bool = False) -> pandas.core.frame.DataFrame
```

Observed behavior from source and runtime:

- returns a DataFrame indexed by `query`
- writes at least `symbol` and `_score`
- strips version suffixes before lookup
- infers `species` from ID prefix when `species` is omitted
- rebuilds / downloads the Ensembl database when needed

Source-derived species prefixes:

- `ENSG` -> `human`
- `ENSMUSG` -> `mouse`
- `ENSRNOG` -> `rat`
- `ENSDARG` -> `zebrafish`
- `FBGN` -> `fly`
- `ENSGALG` -> `chicken`
- `ENSCAFG` -> `dog`
- `ENSSSCG` -> `pig`
- `ENSBTAG` -> `cow`
- `ENSMMUG` -> `macaque`

Important compatibility finding:

- the docstring says `ensembl_release=None` defaults to `109`
- the current code assigns `77`

### `dynamo.preprocessing.convert2symbol`

Observed signature:

```python
(adata: AnnData, scopes: Union[str, Iterable, None] = None, subset=True) -> AnnData
```

Observed behavior from source and runtime:

- enters conversion mode when all `adata.var_names` start with `ENS` or when `scopes` is provided
- auto-selects `scopes='ensembl.gene'` for some human / mouse gene prefixes
- auto-selects `scopes='ensembl.transcript'` for some human / mouse transcript prefixes
- raises for other prefixes when `scopes` is omitted
- stores `query` in `adata.var`
- merges the result of `convert2gene_symbol(...)`
- rewrites `adata.var.index` to `symbol` when `subset=True`

Notebook-important runtime finding:

- `convert2symbol(adata)` fails on zebrafish `ENSDARG...` IDs without explicit `scopes`
- `convert2gene_symbol(...)` succeeds on the same IDs because species inference happens there

### `dynamo.preprocessing.Preprocessor.preprocess_adata`

Observed signature:

```python
(self, adata, recipe='monocle', tkey=None, experiment_type=None) -> None
```

Source-level `recipe` branches:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

Why the branch list matters here:

- the notebook only uses preprocessing as an optional follow-on
- the generated skill still documents the current `recipe` branch set so downstream handoff is not notebook-limited

## Empirical Checks Run

Representative empirical checks:

- `convert2gene_symbol(['ENSG00000141510'])` -> `TP53`
- `convert2gene_symbol(['ENSDARG00000035558.1'], ensembl_release=77)` -> `gps2`
- `convert2symbol(...)` on human IDs rewrote `adata.var_names` to `['TP53', 'CD3D']`
- `convert2symbol(...)` on zebrafish without `scopes` raised an exception
- `convert2symbol(..., scopes='ensembl.gene')` on zebrafish rewrote `adata.var_names` to `['gps2']`

## Human Review Notes

- The notebook frames conversion in MyGene-style service terms, but current executable behavior comes from vendored `pyensembl`.
- The most important hidden branch in this notebook family is not `method` but the species / prefix path for conversion plus the downstream `recipe` path for preprocessing.
