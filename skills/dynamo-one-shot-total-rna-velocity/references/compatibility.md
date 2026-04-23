# Compatibility

Use this reference when notebook wording and current source behavior differ.

## Sample Data Layout

Observed on current `dyn.sample_data.hematopoiesis_raw()`:

- `.obs` includes `batch`, `cell_type`, `time`
- `.layers` includes `new`, `spliced`, `total`, `unspliced`
- `.obsm` already includes `X_umap`
- `.uns` includes `genes_to_use`

Conservative rule:

- treat these as worked-example conveniences, not guarantees for unrelated one-shot labeling data

## Curated Gene List Persistence

Observed current behavior:

- `anndata.concat(...)` dropped `uns["genes_to_use"]` during empirical subset reconstruction

Conservative rule:

- preserve the gene list explicitly before concatenating or rebuilding subsets

## Model-2 Forcing

Notebook prose:

- the tutorial wants the no-splicing Model-2 branch even though the raw object includes splicing layers

Observed current behavior:

- preprocessing auto-detected both labeling and splicing on the hematopoiesis raw object
- empirical notebook parity required `adata.uns["pp"]["has_splicing"] = False` before `dynamics(...)`

Conservative rule:

- do not skip the explicit `has_splicing=False` override when reproducing the tutorial's total-RNA path

## Output-Key Mismatch

Notebook implication:

- `dynamics(...)` is only the setup for total-RNA velocity, not the final projected result

Observed current behavior:

- empirically checked `dynamics(...)` wrote `velocity_N` and `velocity_T`
- it did not write `velocity_S`
- the notebook-like visualization path relied on `velocity_alpha_minus_gamma_s` plus `velocity_umap`

Conservative rule:

- do not stop at `velocity_T` if the user asked for the tutorial's low-dimensional total-RNA velocity output

## Time-Key Requirement

Observed current behavior:

- `calculate_velocity_alpha_minus_gamma_s(...)` reads `adata.obs["time"]` directly

Conservative rule:

- rename or copy any alternate time key to `time` before calling the function

## Projection Storage

Observed current behavior:

- `method='cosine'` stores `adata.obsp["cosine_transition_matrix"]`
- `method='pearson'` stores `adata.obsp["pearson_transition_matrix"]`

Conservative rule:

- do not look for a generic transition-matrix key after projection

## Small Time Groups

Observed current behavior:

- reviewer runs with fewer than `50` cells per time group emitted NaN warnings and filtered genes before projection

Conservative rule:

- prefer coarser groups or more cells per time point before trusting downstream velocity visualization
