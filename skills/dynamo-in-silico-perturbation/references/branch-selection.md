# Branch Selection

## `pertubation_method` Options

Note: the parameter name is `pertubation_method` (one `r`) in the source — a persistent typo. Always spell it this way when calling the function.

| value | when to use |
|---|---|
| `j_delta_x` | **Default and recommended.** Linearizes the perturbation through the Jacobian: `delta_Y = J · (X_perturb - X_wild) · delta_t`. Best for small-to-moderate expression changes where the Jacobian approximation is valid. |
| `j_x_prime` | Use when you want to evaluate the Jacobian at the absolute perturbed state rather than at the delta. Less common. |
| `j_jv` | Second-order: `delta_Y = J · (J · delta_X · J_jv_delta_t) · delta_t`. Use when you want to account for the velocity change induced by the perturbation itself. Control time scales with `J_jv_delta_t` and `delta_t`. |
| `f_x_prime` | Directly evaluates the learned vector field at the perturbed expression coordinates (no Jacobian approximation). Use when the perturbation is large enough that linearization may break down. Requires `VecFld_pca` in `adata.uns`. |
| `f_x_prime_minus_f_x_0` | Computes the difference in vector field between perturbed and wild-type state: `f(X_perturb) - f(X_wt)`. Preferred over `f_x_prime` when you want the net change in trajectory, not the absolute trajectory. |

## `perturb_mode` Options

| value | when to use |
|---|---|
| `raw` | **Default.** Treats `expression` as the absolute gene expression level to set. A value of `100` means "set this gene's expression to 100 in all cells". A value of `-100` is effectively downregulation (note: expression cannot be negative physically; large negative values push cells toward zero). |
| `z_score` | Treats `expression` as a z-score relative to the population distribution. Internally applies `z_score_inv` to map back to raw units. Use when you want a normalized perturbation that is comparable across genes with different expression scales. |

## Expression Value Conventions

- **Suppression**: use large negative values (e.g., `-100`) to represent strong knockdown. The tutorial uses `-100` as a conventional "full suppression" signal.
- **Activation**: use large positive values (e.g., `100`) to represent strong overexpression.
- **Partial perturbation**: use moderate values (e.g., `-15`, `50`) to model weaker effects.
- **Mixed**: combine in the `expression` list, e.g., `expression=[-100, -15]` for two genes at different suppression levels.

## `emb_basis` Options

Any embedding basis key present in `adata.obsm` can be used. After calling `perturbation(adata, ..., emb_basis='umap')`, the result is stored as `X_umap_perturbation` and `velocity_umap_perturbation`. Visualize with `basis='umap_perturbation'` in plotting functions.

Common options:

- `'umap'` — standard UMAP (default)
- `'pca'` — PCA space (2D projection)
- `'tsne'` — t-SNE embedding

## `cells` Subset Option

To restrict the perturbation to a specific cell subset, pass `cells=<index_array>`:

```python
hsc_idx = dyn.tl.select_cell(adata, "cell_type", "HSC")
dyn.pd.perturbation(adata, "GATA1", [-100], emb_basis="umap", cells=hsc_idx)
```

This only perturbs the specified cells; remaining cells retain their unperturbed velocity.

## `zero_perturb_genes_vel` Flag

When `True`, the velocity components of the perturbed genes themselves are zeroed out in the output. Use this when you want to see the downstream (indirect) fate changes without the direct velocity contribution of the perturbed gene.

## `delta_Y` Pre-computed Override

If you have a custom perturbation effect matrix (PCA-space), pass it directly as `delta_Y`. This bypasses the internal Jacobian computation entirely and lets you inject any perturbation signal.

## `streamline_plot` `method` Options

| value | when to use |
|---|---|
| `gaussian` | **Default.** Fast Gaussian kernel grid estimation. Sufficient for most perturbation visualizations. |
| `SparseVFC` | Fits a sparse vector field to the grid. More accurate for complex flow patterns; slower. |

## Saving Perturbation Results Between Scenarios

`dyn.pd.perturbation` always writes to `X_{emb_basis}_perturbation` and `velocity_{emb_basis}_perturbation`. Each subsequent call overwrites the previous result.

To compare multiple scenarios side-by-side, either:

1. Copy results before the next call:
   ```python
   import numpy as np
   gata1_emb = adata.obsm["X_umap_perturbation"].copy()
   gata1_vel = adata.obsm["velocity_umap_perturbation"].copy()
   ```

2. Use `add_embedding_key` and `add_velocity_key` to write to custom keys:
   ```python
   dyn.pd.perturbation(
       adata, "GATA1", [-100], emb_basis="umap",
       add_embedding_key="X_umap_GATA1_down",
       add_velocity_key="velocity_umap_GATA1_down",
   )
   ```
