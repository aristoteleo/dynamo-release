# Compatibility

Use this reference when notebook wording and current source behavior differ.

Scope rule:

- zebrafish-specific details here are compatibility notes for the worked example dataset
- they are not universal assumptions for every conventional spliced/unspliced input

## Sample Data Shape

Observed on current `dyn.sample_data.zebrafish()` worked example:

- raw dataset includes `spliced` and `unspliced` layers
- raw dataset includes `Cell_type` in `.obs`
- raw dataset does not include `obsm["X_umap"]`
- observation names are not unique

Conservative rule:

- run `adata.obs_names_make_unique()` immediately after loading
- run `dyn.tl.reduceDimension(...)` before expecting `X_umap`
- do not generalize zebrafish sample metadata layout to unrelated datasets

## Transition Matrix Storage

Notebook prose treats the transition matrix generically.

Observed current behavior for `method='pearson'`:

- velocity projection key: `adata.obsm["velocity_umap"]`
- transition matrix key: `adata.obsp["pearson_transition_matrix"]`

Conservative rule:

- do not hard-code `transition_matrix` unless the code path actually created that key

## `VectorField(..., M=1000)`

The notebook passes `M=1000`.

Observed current source behavior:

- `VectorField` accepts this via `**kwargs`
- SparseVFC defaults are set in `_get_svc_default_arguments(...)`
- `M` defaults to `None` there

Conservative rule:

- document `M` as a pass-through SparseVFC tuning knob, not a top-level public parameter

## Potential / Topology Path

Observed current behavior:

- `pot_curl_div=True` on a 2D basis computes ddHodge potential, curl, and divergence automatically

Conservative rule:

- prefer this compact path over manually assuming that plotting functions themselves compute all prerequisites

## Animation Tooling

Observed current behavior:

- `animate_fates(..., save_show_or_return='save')` may require `imagemagick`

Conservative rule:

- keep animation optional and do not make it a validation prerequisite for the main analytical workflow
