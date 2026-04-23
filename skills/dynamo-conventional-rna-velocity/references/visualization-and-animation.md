# Visualization And Animation

Use this reference only when the user explicitly wants notebook-style plots, topology visuals, or animations.

## Topography

Primary entrypoint:

- `dyn.pl.topography(...)`

Recommended prerequisites:

1. `dyn.tl.reduceDimension(...)`
2. `dyn.tl.cell_velocities(..., basis='umap')`
3. `dyn.vf.VectorField(..., basis='umap', pot_curl_div=True)`

Why:

- topography depends on the learned vector field and is most natural on a 2D basis such as UMAP
- `pot_curl_div=True` precomputes potential, curl, and divergence keys used throughout the notebook interpretation

Notebook-like call:

```python
dyn.pl.topography(
    adata,
    basis="umap",
    color=["ntr", "Cell_type"],
    streamline_color="black",
    show_legend="on data",
    frontier=True,
)
```

## Potential Trends

After UMAP ddHodge potential is available:

```python
dyn.pl.umap(adata, color="umap_ddhodge_potential")
```

Use this only after verifying that `adata.obs["umap_ddhodge_potential"]` exists.

## Stream Function Animation

Primary entrypoints:

- `dyn.mv.StreamFuncAnim(...)`
- `dyn.mv.animate_fates(...)`

Required prerequisite:

- `dyn.pd.fate(...)` must already have populated `adata.uns["fate_<basis>"]`

Notebook-like path:

```python
fig, ax = plt.subplots(figsize=(4, 4))
ax = dyn.pl.topography(adata, color="Cell_type", ax=ax, save_show_or_return="return")

instance = dyn.mv.StreamFuncAnim(
    adata=adata,
    color="Cell_type",
    ax=ax,
)
```

Or save directly:

```python
dyn.mv.animate_fates(
    adata,
    color="Cell_type",
    basis="umap",
    n_steps=100,
    fig=fig,
    ax=ax,
    save_show_or_return="save",
    logspace=True,
    max_time=None,
)
```

## Streamline GIF Path

Notebook-specific optional path:

1. `X_grid, V_grid = dyn.pl.compute_velocity_on_grid(...)`
2. `dyn.pl.animate_streamplot(X_grid, V_grid, adata, ...)`

This is presentation-heavy and not required for the analytical workflow.

## Constraints

- Do not let animation setup block the main analysis.
- Keep `font_path='Arial'`, notebook magics, and background-color changes out of the default workflow unless the user explicitly wants presentation parity.
- `animate_fates(..., save_show_or_return='save')` may require external animation tooling such as `imagemagick`.
