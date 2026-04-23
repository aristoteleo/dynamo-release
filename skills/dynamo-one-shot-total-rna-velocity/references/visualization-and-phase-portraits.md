# Visualization And Phase Portraits

Use this reference only when the user explicitly wants notebook-style figures after the analytical workflow has already populated `velocity_umap` or the total-RNA velocity layer.

## Streamline Plot

Primary entrypoint:

- `dyn.pl.streamline_plot(...)`

Recommended prerequisites:

1. `dyn.tl.cell_velocities(..., X=adata.layers["M_t"], V=adata.layers["velocity_alpha_minus_gamma_s"])`
2. a matching low-dimensional embedding such as `adata.obsm["X_umap"]`

Notebook-like call:

```python
dyn.pl.streamline_plot(
    adata,
    color=["batch", "cell_type", "PF4"],
    ncols=2,
    basis="umap",
    s_kwargs_dict={"adjust_legend": True, "dpi": 80},
    figsize=(4, 4),
)
```

Observed plotting `method` branches in current source:

- `gaussian`
- `SparseVFC`

Recommendation:

- keep `method='gaussian'` unless the user explicitly wants a vector-field-backed streamline rendering

## Total RNA Phase Portraits

Primary entrypoint:

- `dyn.pl.phase_portraits(...)`

The notebook ends with phase-diagram prose but does not include the executed code cell in the extracted source. The adaptation below is an inference from the current `phase_portraits` signature plus the notebook's total-RNA execution spine.

Inference-backed total-RNA adaptation:

```python
dyn.pl.phase_portraits(
    adata,
    genes=["PF4"],
    ekey="M_t",
    vkey="velocity_alpha_minus_gamma_s",
    color="cell_type",
    basis="umap",
    use_smoothed=True,
)
```

Why this inference is reasonable:

- the notebook's analytical path produces total-RNA expression in `M_t`
- the custom velocity layer is `velocity_alpha_minus_gamma_s`
- relying on splicing defaults would not match the notebook's total-RNA branch

Conservative rule:

- label this as an adaptation unless the user provides a newer notebook revision with the exact phase-portrait cell

## Constraints

- Do not let plotting block the main analytical workflow.
- Do not assume `streamline_plot(...)` itself computes the total-RNA velocity layer; it only visualizes the outputs already present on `adata`.
- Do not reuse notebook font, style, or background tweaks unless the user explicitly wants presentation parity.
