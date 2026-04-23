# Pseudotime And State Graph

Use this reference when the user wants vector-field pseudotime, heatmaps, or group-level transition graphs after geometry and Jacobian work.

## ddhodge Core

Current callable:

```python
dyn.ext.ddhodge(
    adata,
    X_data=None,
    layer=None,
    basis="pca",
    n=30,
    VecFld=None,
    adjmethod="graphize_vecfld",
    distance_free=False,
    n_downsamples=5000,
    up_sampling=True,
    sampling_method="velocity",
    seed=19491001,
    enforce=False,
    cores=1,
)
```

Observed branch-heavy parameters:

- `adjmethod`: `graphize_vecfld`, `naive`
- `sampling_method`: `random`, `velocity`, `trn`

Storage rules:

- `adata.obsp['pca_ddhodge']`
- `adata.obs['pca_ddhodge_div']`
- `adata.obs['pca_ddhodge_potential']`

Recommendation:

- use `basis='pca'`
- keep `adjmethod='graphize_vecfld'`
- keep `sampling_method='velocity'`

When to switch:

- use `adjmethod='naive'` only when a suitable adjacency or transition matrix already exists and the user explicitly wants that branch
- use `sampling_method='trn'` only when the user wants the transition-gene-driven cell subsampling branch

## kinetic_heatmap

Current callable shape:

```python
dyn.pl.kinetic_heatmap(
    adata,
    genes,
    mode="vector_field",
    basis=None,
    layer="X",
    project_back_to_high_dim=True,
    tkey="potential",
    gene_order_method="maximum",
    save_show_or_return="show",
)
```

Observed branch-heavy parameters:

- `mode`: `vector_field`, `lap`, `pseudotime`
- `gene_order_method`: `maximum`, `half_max_ordering`, `raw`

Important source details:

- current source stores results under `adata.uns['kinetics_heatmap']`
- when `mode='pseudotime'` and the requested `tkey` is missing, current source can auto-run `ddhodge(adata)` on defaults

Recommendation:

- use `mode='pseudotime'`
- use `basis='pca'`
- use `tkey='pca_ddhodge_potential'`
- use `gene_order_method='maximum'`
- use `save_show_or_return='return'` in validation or headless environments

## state_graph

Current callable:

```python
dyn.pd.state_graph(
    adata,
    group,
    method="vf",
    transition_mat_key="pearson_transition_matrix",
    approx=False,
    eignum=5,
    basis="umap",
    layer=None,
    arc_sample=False,
    sample_num=100,
    prune_graph=False,
)
```

Observed branch-heavy parameters:

- `method`: `vf`, `markov`, `naive`

Storage rule:

- `state_graph(..., group='Cell_type')` writes `adata.uns['Cell_type_graph']`
- that payload includes `group_graph`, `group_avg_time`, and `group_names`

Recommendation:

- use `group='<group>'`
- use `basis='pca'` when the vector field was fit in PCA space
- use `method='vf'`
- keep `sample_num` moderate on subsets

When to switch:

- use `method='markov'` or `method='naive'` only when the user explicitly wants a transition-matrix interpretation instead of vector-field integration
- use `prune_graph=True` only when the user wants a pruned state graph and accepts the extra similarity-graph dependency

## Practical Sequence

Recommended order:

1. `ddhodge(...)`
2. `kinetic_heatmap(...)`
3. `state_graph(...)`

Reason:

- `kinetic_heatmap(..., mode='pseudotime')` and notebook interpretation both depend on vector-field pseudotime
- `state_graph(..., method='vf')` works best after the PCA vector field is already validated
