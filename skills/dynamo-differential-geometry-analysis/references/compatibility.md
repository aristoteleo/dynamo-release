# Compatibility

Use this reference when notebook wording and current runtime behavior are not identical.

## Current API Preference

Prefer the current callable paths:

- `dyn.pp.Preprocessor`
- `dyn.tl.dynamics`
- `dyn.tl.cell_velocities`
- `dyn.vf.VectorField`
- `dyn.vf.acceleration`
- `dyn.vf.curvature`
- `dyn.vf.jacobian`
- `dyn.ext.ddhodge`
- `dyn.pd.state_graph`

Do not present the notebook as if it were the API contract.

## Source Vs Notebook

- The notebook presentation reads like the zebrafish example is already downstream-ready. Reviewer execution showed `dyn.sample_data.zebrafish()` is raw in the current runtime and needs preprocessing, dynamics, embedding, and velocity fitting.
- Reviewer execution found non-unique observation names on the zebrafish worked example, so `obs_names_make_unique()` is part of the practical bootstrap path.
- The notebook shows the simple `cell_velocities(..., basis='pca')` call. Reviewer execution on smaller subsets needed `transition_genes=adata.var.use_for_pca.values` for that branch to succeed reliably.
- `kinetic_heatmap(...)` stores under `adata.uns['kinetics_heatmap']`, and `cleanup(...)` removes that key. The notebook does not call out that interaction.
- `rank_divergence_genes(...)` ranks diagonal Jacobian elements, not the usual geometric divergence trace.
- `state_graph(..., method='vf')` can emit warnings on sparse or lightly sampled group transitions while still producing a usable `'<group>_graph'` payload.

## Runtime Constraints

- Importing `dynamo` for reviewer execution was done in a repository-compatible Python environment with the repository root on `PYTHONPATH`.
- `VectorField(..., method='dynode')` is a real branch but depends on an additional backend that was not required for the default path.
- `kinetic_heatmap(..., save_show_or_return='show')` is not ideal for headless validation. Use `'return'` in smoke tests and automated review.

## Conservative Rule

If notebook text and live source disagree:

1. trust the current source for executable behavior
2. mention the divergence explicitly
3. keep the notebook only as a worked example or parity reference
