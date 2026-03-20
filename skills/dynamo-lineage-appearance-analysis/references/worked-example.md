# Worked Example

This notebook family uses the processed hematopoiesis example bundled in `dyn.sample_data.hematopoiesis()`.

## Observed Worked-Example Inputs

Observed on the current sample:

- shape: `1947 x 1956`
- `obs['cell_type']` includes labels such as `HSC`, `MEP-like`, `Meg`, `Ery`, and `Bas`
- `obsm` already includes `X_pca`, `X_umap`, `velocity_pca`, `velocity_umap`, `acceleration_pca`, and `curvature_pca`
- `obsp` already includes `cosine_transition_matrix` and `fp_transition_rate`
- `uns` already includes `VecFld_pca`, `VecFld_umap`, `PCs`, and `dynamics`

## Notebook-Specific Defaults

The notebook focuses on:

- lineage labels: `HSC`, `MEP-like`, `Meg`, `Ery`, `Bas`
- regulator pair: `FLI1`, `KLF1`
- fixed-point curation after `dyn.vf.topography(..., n=750)`
- notebook-style plots on `basis='umap'`
- Jacobian and vector-calculus quantities in `basis='pca'`

## Reuse Rule

Treat these as defaults only when the user is explicitly adapting the megakaryocyte-appearance notebook or analyzing a closely related hematopoiesis dataset.

For a generalized lineage-appearance task:

- replace `cell_type` with the user's grouping column
- replace `FLI1` and `KLF1` with the regulator and effector genes of interest
- keep the graph-operator and vector-calculus workflow intact

## Worked-Example Cell Filter

The notebook narrows some plots to:

- `HSC`
- `MEP-like`
- `Meg`
- `Ery`
- `Bas`

That filtering is useful for the narrative, but it is not part of the generic skill contract.
