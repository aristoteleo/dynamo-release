# Source Grounding

This skill was generated from live code inspection and reviewer-run execution against the dynamo source, not from notebook narration alone.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/502_perturbation_tutorial.ipynb`

Code inspected:

- `dynamo/prediction/perturbation.py`
- `dynamo/plot/scVectorField.py`
- `dynamo/prediction/__init__.py`

## Inspected Interfaces

### `dyn.pd.perturbation`

Source: `dynamo/prediction/perturbation.py`

Observed signature:

```python
def perturbation(
    adata: anndata.AnnData,
    genes: Union[str, list],
    expression: Union[float, list] = 10,
    perturb_mode: str = "raw",
    cells: Optional[Union[list, np.ndarray]] = None,
    zero_perturb_genes_vel: bool = False,
    pca_key: Optional[Union[str, np.ndarray]] = None,
    PCs_key: Optional[Union[str, np.ndarray]] = None,
    pca_mean_key: Optional[Union[str, np.ndarray]] = None,
    basis: str = "pca",
    emb_basis: str = "umap",
    jac_key: str = "jacobian_pca",
    X_pca: Optional[np.ndarray] = None,
    delta_Y: Optional[np.ndarray] = None,
    projection_method: str = "fp",
    pertubation_method: str = "j_delta_x",
    J_jv_delta_t: float = 1,
    delta_t: float = 1,
    add_delta_Y_key: Optional[str] = None,
    add_transition_key: Optional[str] = None,
    add_velocity_key: Optional[str] = None,
    add_embedding_key: Optional[str] = None,
)
```

**Important**: The parameter is spelled `pertubation_method` (one `r`, not two) — this is a typo in the source that has persisted. Using `perturbation_method` (correct spelling) will be silently ignored as an unexpected keyword.

Observed `pertubation_method` branches (from source `if/elif` chain):

| value | computation |
|---|---|
| `j_delta_x` | `delta_Y = J · (X_perturb_pca - X_pca) · delta_t` — default; linearized change |
| `j_x_prime` | `delta_Y = J · X_perturb_pca · delta_t` — absolute perturbed state |
| `j_jv` | `delta_Y = J · (J · delta_X · J_jv_delta_t) · delta_t` — second-order |
| `f_x_prime` | `delta_Y = f(X_perturb_pca)` — direct vector-field evaluation |
| `f_x_prime_minus_f_x_0` | `delta_Y = f(X_perturb_pca) - f(X_pca)` — vector-field difference |

Observed `perturb_mode` branches:

| value | behavior |
|---|---|
| `raw` | treat `expression` as absolute gene expression value |
| `z_score` | treat `expression` as z-score; internally applies `z_score_inv` to convert back |

Observed storage keys written by default call (`emb_basis='umap'`, `pertubation_method='j_delta_x'`):

- `adata.obsm['X_umap_perturbation']` — perturbed UMAP coordinates
- `adata.obsm['velocity_umap_perturbation']` — perturbation velocity in UMAP space
- `adata.obsm['j_delta_x_perturbation']` — PCA-space perturbation effect matrix
- `adata.layers['j_delta_x_perturbation']` — gene-space perturbation effects (sparse)
- `adata.obsp['perturbation_transition_matrix']` — cell transition probability matrix

Custom key overrides via `add_*` parameters:

- `add_delta_Y_key` → overrides the `{pertubation_method}_perturbation` obsm key name
- `add_transition_key` → overrides `perturbation_transition_matrix`
- `add_velocity_key` → overrides `velocity_{emb_basis}_perturbation`
- `add_embedding_key` → overrides `X_{emb_basis}_perturbation`

Internal mechanism (simplified):

```python
# 1. Retrieve Jacobian
Js = adata.uns[jac_key]["jacobian"]  # shape: pcs × pcs × cells

# 2. Project perturbed expression to PCA
X_perturb_pca = expr_to_pca(X_perturb, PCs, means)

# 3. Compute delta_Y (default j_delta_x)
delta_X = X_perturb_pca - X_pca
delta_Y[i] = Js[:, :, i] @ delta_X[i] * delta_t

# 4. Project to embedding space via cell_velocities()
# writes X_{emb_basis}_perturbation and velocity_{emb_basis}_perturbation
```

### `dyn.pl.streamline_plot`

Source: `dynamo/plot/scVectorField.py`

Observed signature (selected parameters):

```python
def streamline_plot(
    adata: AnnData,
    basis: str = "umap",
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    color: Union[str, List[str]] = "ntr",
    method: Literal["gaussian", "SparseVFC"] = "gaussian",
    xy_grid_nums: Tuple[int, int] = (50, 50),
    cut_off_velocity: bool = True,
    density: float = 1,
    linewidth: float = 1,
    streamline_alpha: float = 1,
    vector: str = "velocity",
    inverse: bool = False,
    save_show_or_return: Literal["save", "show", "return"] = "show",
    **streamline_kwargs,
) -> List[Axes]
```

For perturbation visualization, pass `basis="umap_perturbation"` (or `"{emb_basis}_perturbation"`). The function automatically constructs keys:

- embedding: `adata.obsm["X_" + basis]` → `"X_umap_perturbation"`
- velocity: `adata.obsm["{vector}_" + basis]` → `"velocity_umap_perturbation"`

Observed `method` branches:

- `gaussian` (default) — Gaussian kernel grid velocity estimation
- `SparseVFC` — sparse vector field reconstruction for grid

Observed `sort` branches: `"raw"` (default), `"abs"`, `"neg"`

### `dyn.pl.cell_wise_vectors`

Source: `dynamo/plot/scVectorField.py`

Observed signature (selected parameters):

```python
def cell_wise_vectors(
    adata: AnnData,
    basis: str = "umap",
    vector: str = "velocity",
    projection: Literal["2d", "3d"] = "2d",
    quiver_size: Optional[float] = 1,
    quiver_length: Optional[float] = None,
    inverse: bool = False,
    cell_inds: str = "all",
    save_show_or_return: Literal["save", "show", "return"] = "show",
) -> Optional[List[Axes]]
```

Use `basis="umap_perturbation"` to visualize perturbation vectors as quiver arrows per cell.

### `dyn.pl.cell_wise_vectors_3d`

Source: `dynamo/plot/scVectorField.py`

Observed `plot_method` branches:

- `"pv"` — PyVista backend (requires `pyvista` installed)
- `"matplotlib"` — standard matplotlib 3D axes
