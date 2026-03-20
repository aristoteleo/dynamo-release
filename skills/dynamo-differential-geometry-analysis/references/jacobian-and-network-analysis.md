# Jacobian And Network Analysis

Use this reference when the user wants regulatory interpretation instead of only scalar geometry.

## Preconditions

Before running any Jacobian-derived analysis, confirm:

- `adata.uns['VecFld_pca']` exists
- `adata.uns['PCs']` or `adata.varm['PCs']` exists
- the gene set is narrowed to something tractable

The notebook does not make the scaling problem explicit enough. Whole-transcriptome Jacobians are expensive in both memory and interpretation.

## Shrink The Gene Universe First

Current helper:

```python
dyn.pp.top_pca_genes(
    adata,
    pc_key="PCs",
    n_top_genes=100,
    pc_components=None,
    adata_store_key="top_pca_genes",
)
```

Storage rule:

- `top_pca_genes(...)` writes a boolean mask to `adata.var['top_pca_genes']`

Recommended pattern:

```python
dyn.pp.top_pca_genes(adata, n_top_genes=100)
genes = adata.var_names[adata.var["top_pca_genes"]][:50].tolist()
```

## Jacobian Core

Current callable:

```python
dyn.vf.jacobian(
    adata,
    regulators=None,
    effectors=None,
    cell_idx=None,
    sampling=None,
    sample_ncells=1000,
    basis="pca",
    Qkey="PCs",
    vector_field_class=None,
    method="analytical",
    store_in_adata=True,
)
```

Branch notes:

- `sampling=None` uses all available cells
- `sampling='random'` samples uniformly
- `sampling='velocity'` samples by velocity magnitude
- `sampling='trn'` uses the transition-gene based branch
- `method='analytical'` is the default reusable path
- `method='numerical'` exists for branch comparison or debugging

Storage rule:

- `jacobian(..., basis='pca')` writes `adata.uns['jacobian_pca']`
- determinant summaries are written to `adata.obs['jacobian_det_pca']`

## Ranking Modes

Current `rank_jacobian_genes(...)` modes:

- `full reg` or `full_reg`
- `full eff` or `full_eff`
- `reg`
- `eff`
- `int`
- `switch`

Interpretation:

- `full_reg`: ranked regulators for every effector in each group
- `full_eff`: ranked effectors for every regulator in each group
- `reg`: grouped summary focused on regulators
- `eff`: grouped summary focused on effectors
- `int`: pairwise interaction ranking
- `switch`: mutual inhibition candidates

Recommended usage:

```python
full_reg = dyn.vf.rank_jacobian_genes(
    adata,
    groups="Cell_type",
    mode="full_reg",
    abs=True,
    return_df=True,
)

reg_rank = dyn.vf.rank_jacobian_genes(
    adata,
    groups="Cell_type",
    mode="reg",
    abs=True,
    output_values=True,
    return_df=True,
)
```

## Divergence-Like Ranking

Current helper:

```python
dyn.vf.rank_divergence_genes(
    adata,
    jkey="jacobian_pca",
    genes=None,
    prefix_store="rank_div_gene",
)
```

Important source detail:

- this helper ranks diagonal Jacobian elements
- it does not compute geometric divergence as the full Jacobian trace in the usual vector-calculus sense
- it requires `regulators == effectors` in the stored Jacobian

## Build Cluster-Specific Networks

Current helper:

```python
dyn.vf.build_network_per_cluster(
    adata,
    cluster,
    cluster_names=None,
    full_reg_rank=None,
    full_eff_rank=None,
    genes=None,
    n_top_genes=100,
    abs=False,
)
```

Observed behavior:

- if `full_reg_rank` or `full_eff_rank` is missing, the helper recomputes grouped ranking internally
- it returns a dict keyed by cluster name
- each value is a `DataFrame` with `regulator`, `target`, and `weight`

Recommended pattern:

```python
edges = dyn.vf.build_network_per_cluster(
    adata,
    cluster="Cell_type",
    cluster_names=["Unknown"],
    full_reg_rank=full_reg,
    genes=genes[:20],
    n_top_genes=10,
    abs=True,
)
```

## Plotting Handoff

Only move into `dyn.pl.jacobian(...)`, `dyn.pl.arcPlot(...)`, or `dyn.pl.circosPlot(...)` after the underlying Jacobian or edge list has been validated. The notebook interleaves plotting with interpretation, but the reusable skill should treat those as downstream consumers.
