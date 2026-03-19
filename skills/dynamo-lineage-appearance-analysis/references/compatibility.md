# Compatibility

Use this note when notebook behavior and current runtime behavior diverge.

## Fixed-Point Indices Are Not Stable Defaults

The notebook's manual fixed-point selection:

```python
good_fixed_points = [2, 8, 1, 195, 4, 5]
```

is only meaningful after remapping the same dataset with the same `basis`, `n`, and vector-field state.

Do not copy those indices into a different dataset or a differently remapped vector field.

## `quiver_source='reconstructed'` Is Not Safe In The Current Runtime

Reviewer execution on the current source hit:

```text
ImportError: cannot import name 'vector_field_function' from 'dynamo.tools.utils'
```

when `dyn.pl.topography(..., quiver_source='reconstructed')` was used.

Practical rule:

- use the default `quiver_source='raw'`
- if you need reconstructed quivers, patch or verify the source first

## `curvature(formula=1)` Is Not Safe In The Current Runtime

Reviewer execution on the current source hit:

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

because `formula=1` produced no curvature matrix while the function still attempted to write it into `adata.obsm`.

Practical rule:

- prefer `formula=2`
- only use `formula=1` after confirming the runtime has been fixed

## `terms` Naming Drift In `dyn.pl.topography`

The docstring mentions `separatrix`, but the implementation checks for `separatrices`.

Practical rule:

- rely on the default `terms=['streamline', 'fixed_points']`
- verify any non-default `terms` branch directly against current source

## Notebook Cosmetics Are Not Part Of The Skill Contract

The notebook includes presentation-only setup such as:

- `dyn.configuration.set_figure_params(...)`
- `dyn.pl.style(font_path='Arial')`
- bespoke seaborn and matplotlib layout code

These should stay out of the core reusable workflow unless the user explicitly asks for figure styling parity.

## Upstream Dependency Boundary

This skill assumes the data already has vector-field outputs and transition graphs.

If a user only has raw data, they still need upstream preprocessing, dynamics, velocity, and vector-field reconstruction before this notebook-derived analysis makes sense.
