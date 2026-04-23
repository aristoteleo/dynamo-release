# Visualization And Animation

Use this reference only when the user explicitly wants notebook-style plots, topology visuals, or animations.

## Streamline And Cell-Wise Velocity Plots

Primary entrypoints:

- `dyn.pl.streamline_plot(...)`
- `dyn.pl.cell_wise_vectors(...)`

Recommended prerequisites:

1. `dyn.tl.pseudotime_velocity(...)`
2. a matching low-dimensional embedding such as `X_umap`

Notebook-like calls:

```python
dyn.pl.streamline_plot(
    adata,
    color=["clusters"],
    basis="umap",
    show_legend="on data",
)

dyn.pl.cell_wise_vectors(
    adata,
    color=["clusters"],
    basis="umap",
    show_legend="on data",
)
```

## Vector Field Topology

Primary entrypoint:

- `dyn.pl.topography(...)`

Recommended prerequisites:

1. `dyn.tl.pseudotime_velocity(...)`
2. `dyn.vf.VectorField(..., basis='umap', pot_curl_div=True)`

Notebook-like call:

```python
dyn.pl.topography(
    adata,
    basis="umap",
    color=["ntr", "clusters"],
    streamline_color="black",
    show_legend="on data",
    frontier=True,
)
```

Use `dyn.pl.umap(adata, color='umap_ddhodge_potential')` only after validating that `adata.obs['umap_ddhodge_potential']` exists.

## Fate Animation

Primary entrypoints:

- `dyn.mv.StreamFuncAnim(...)`
- `dyn.mv.animate_fates(...)`

Required prerequisite:

- `dyn.pd.fate(...)` must already have populated `adata.uns['fate_<basis>']`

Notebook-like path:

```python
fig, ax = plt.subplots(figsize=(4, 4))
ax = dyn.pl.topography(adata, color="clusters", ax=ax, save_show_or_return="return")

instance = dyn.mv.StreamFuncAnim(
    adata=adata,
    color="clusters",
    ax=ax,
)
```

Important branch note:

- `StreamFuncAnim(..., logspace=False)` samples evenly by default
- `dyn.mv.animate_fates(..., logspace=True)` is an explicit alternate branch for log-spaced time sampling

## Streamplot GIF Path

Notebook-specific optional path:

1. `X_grid, V_grid = dyn.pl.compute_velocity_on_grid(adata.obsm['X_umap'], adata.obsm['velocity_umap'])`
2. `dyn.pl.animate_streamplot(X_grid, V_grid, adata, ...)`

This is presentation-heavy and not required for the analytical workflow.

## Constraints

- Do not let animation setup block the main analysis.
- Keep notebook magics, `font_path='Arial'`, and background-color toggles out of the default workflow unless the user explicitly wants presentation parity.
- GIF export may require external animation tooling such as `imagemagick`.
