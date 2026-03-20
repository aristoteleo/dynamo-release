# Compatibility Notes

## `pertubation_method` Parameter Typo

The parameter is spelled `pertubation_method` (one `r`) in the source code at `dynamo/prediction/perturbation.py`. This typo has persisted across versions. Using the correctly-spelled `perturbation_method` will be silently ignored as an unexpected keyword argument with `**kwargs`, producing no error but also having no effect — the default `j_delta_x` will be used instead.

Always use:
```python
dyn.pd.perturbation(adata, gene, expression, pertubation_method="f_x_prime_minus_f_x_0")
#                                              ^^ one 'r'
```

## Jacobian Prerequisite

The default `pertubation_method='j_delta_x'` requires `adata.uns['jacobian_pca']` with keys `jacobian`, `regulators`, `effectors`, and `cell_idx`. This is pre-populated in `dyn.sample_data.hematopoiesis()` but absent on a freshly fitted dataset unless you explicitly run:

```python
dyn.vf.jacobian(adata, regulators=gene_list, effectors=gene_list, basis='pca')
```

If a gene is not in `regulators` or `effectors`, its row/column in the Jacobian is zero and the perturbation has no effect. Always verify the gene is covered before interpreting null results.

## `f_x_prime` and `f_x_prime_minus_f_x_0` Prerequisites

These methods bypass the Jacobian and call the vector field function `f` directly. They require `adata.uns['VecFld_pca']` to be present (which it is in the hematopoiesis sample). If the vector field was only fitted on UMAP basis, use `basis='umap'` instead of `basis='pca'`.

## Overwriting Behavior

Each call to `dyn.pd.perturbation` overwrites `adata.obsm['X_{emb_basis}_perturbation']`, `adata.obsm['velocity_{emb_basis}_perturbation']`, and `adata.obsp['perturbation_transition_matrix']`. Previous perturbation results are lost unless:

- Copied manually before the next call
- Written to custom keys via `add_embedding_key`, `add_velocity_key`, `add_transition_key`

## `expression` Scalar vs. List

When `genes` is a string (single gene), `expression` can be a scalar (`-100`) or a one-element list (`[-100]`). When `genes` is a list, `expression` must be a list of matching length. Mismatched lengths will raise an error inside the function.

## `cell_velocities` Projection Dependency

Internally, `perturbation` calls `dyn.tl.cell_velocities` to project the PCA perturbation to the embedding. This requires `adata.obsp['cosine_transition_matrix']` (or the relevant transition matrix for the chosen `projection_method`). This is present in the hematopoiesis sample but may be absent on custom datasets that skipped the velocity step. Run `dyn.tl.cell_velocities(adata)` first if you see a missing key error during perturbation.

## `streamline_plot` Basis Key Construction

`dyn.pl.streamline_plot(adata, basis='umap_perturbation')` looks for:

- `adata.obsm['X_umap_perturbation']` — coordinates
- `adata.obsm['velocity_umap_perturbation']` — velocity field

Both keys must exist (written by `perturbation(..., emb_basis='umap')`). If a different `emb_basis` was used, adjust the `basis` parameter accordingly:

```python
dyn.pd.perturbation(adata, gene, expr, emb_basis="tsne")
dyn.pl.streamline_plot(adata, basis="tsne_perturbation")
```
