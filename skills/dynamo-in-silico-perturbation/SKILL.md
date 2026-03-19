---
name: dynamo-in-silico-perturbation
description: Perform in silico gene perturbation on a dynamo vector-field AnnData to predict cell fate diversion after single or multi-gene activation or suppression, then visualize the results with streamline or quiver plots. Use when running dyn.pd.perturbation, predicting transcription factor perturbation effects, simulating gene knockdown or overexpression in scRNA-seq data, reproducing 502_perturbation_tutorial.ipynb, or choosing among pertubation_method, perturb_mode, and emb_basis branches.
---

# Dynamo In Silico Gene Perturbation

## Goal

Given a `dynamo` AnnData with a fitted PCA-space vector field and Jacobian, apply expression-level changes to one or more genes and project the resulting velocity shift into a low-dimensional embedding to predict cell fate diversion. Visualize the perturbed trajectories with streamline or cell-wise vector plots.

## Quick Workflow

1. Confirm the input AnnData has `VecFld_pca`, `jacobian_pca`, `X_pca`, and the target embedding (e.g., `X_umap`) precomputed.
2. Call `dyn.pd.perturbation(adata, genes, expression, emb_basis=...)` to compute the perturbation effect and project it to the embedding space.
3. Visualize with `dyn.pl.streamline_plot(adata, basis="{emb_basis}_perturbation", ...)`.
4. Optionally use `dyn.pl.cell_wise_vectors` for quiver-style single-cell vector plots.
5. Repeat steps 2–4 for each perturbation scenario (single gene, multi-gene, mixed activation/suppression).

## Interface Summary

- `dyn.pd.perturbation(adata, genes, expression=10, perturb_mode='raw', cells=None, zero_perturb_genes_vel=False, pca_key=None, PCs_key=None, pca_mean_key=None, basis='pca', emb_basis='umap', jac_key='jacobian_pca', X_pca=None, delta_Y=None, projection_method='fp', pertubation_method='j_delta_x', J_jv_delta_t=1, delta_t=1, add_delta_Y_key=None, add_transition_key=None, add_velocity_key=None, add_embedding_key=None)` — core perturbation call; writes perturbed embedding and velocity into `adata.obsm`.

  **Note on typo**: the parameter is spelled `pertubation_method` (one `r`) in the source — use that exact spelling.

- `dyn.pl.streamline_plot(adata, basis='umap', ..., method='gaussian', vector='velocity', save_show_or_return='show', **streamline_kwargs)` — after perturbation, call with `basis='{emb_basis}_perturbation'` to visualize diverted cell trajectories.

- `dyn.pl.cell_wise_vectors(adata, basis='umap', ..., vector='velocity', projection='2d', quiver_size=1, quiver_length=None, save_show_or_return='show')` — single-cell quiver vectors; works with perturbation basis.

Read `references/source-grounding.md` for full inspected signatures and storage key evidence.

## Stage Selection

- **Single-gene suppression**: pass one gene string and a single negative value, e.g. `genes='GATA1', expression=[-100]`.
- **Single-gene activation**: pass one gene string and a single positive value, e.g. `genes='KLF1', expression=[100]`.
- **Multi-gene simultaneous perturbation**: pass a list of gene names and matching list of expression values.
- **Mixed activation + suppression**: combine positive and negative values in the expression list.
- Use `perturb_mode='raw'` (default) to shift expression by the given absolute value. Use `perturb_mode='z_score'` when expression units are z-scores.
- Use `pertubation_method='j_delta_x'` (default) for the standard linearized Jacobian approach. Use `'f_x_prime'` or `'f_x_prime_minus_f_x_0'` when you want direct vector-field evaluation instead of Jacobian approximation.
- Restrict perturbation to a cell subset by passing `cells=<index_array>`.

Read `references/branch-selection.md` for details on all `pertubation_method` branches and when to use them.

## Input Contract

- `AnnData` with `VecFld_pca` in `.uns` (PCA-space vector field must already be fitted).
- `adata.uns['jacobian_pca']` must exist with keys `jacobian`, `regulators`, `effectors`, and `cell_idx`. Run `dyn.vf.jacobian(adata, basis='pca')` first if absent.
- `adata.uns['PCs']` and `adata.uns['pca_mean']` (or `adata.obsm['X_pca']`) must exist for PCA projection.
- `adata.obsm['X_{emb_basis}']` must exist for the target embedding projection (default `X_umap`).
- The hematopoiesis sample dataset (`dyn.sample_data.hematopoiesis()`) already satisfies all these requirements.
- Gene names in `genes` must appear in both `adata.var_names` and the Jacobian `regulators`/`effectors` arrays.

## Minimal Execution Patterns

### Single-gene suppression

```python
import dynamo as dyn

adata = dyn.sample_data.hematopoiesis()

gene = "GATA1"
dyn.pd.perturbation(adata, gene, [-100], emb_basis="umap")
dyn.pl.streamline_plot(adata, color=["cell_type", gene], basis="umap_perturbation")
```

### Single-gene activation

```python
gene = "KLF1"
dyn.pd.perturbation(adata, gene, [100], emb_basis="umap")
dyn.pl.streamline_plot(adata, color=["cell_type", gene], basis="umap_perturbation")
```

### Multi-gene mixed perturbation

```python
selected_genes = ["SPI1", "GATA1"]
expr_vals = [-100, -15]
dyn.pd.perturbation(adata, selected_genes, expr_vals, emb_basis="umap")
dyn.pl.streamline_plot(adata, color=["cell_type"] + selected_genes, basis="umap_perturbation")
```

### Restrict to a cell subset

```python
hsc_idx = dyn.tl.select_cell(adata, "cell_type", "HSC")
dyn.pd.perturbation(adata, "GATA1", [-100], emb_basis="umap", cells=hsc_idx)
```

### Custom perturbation method

```python
# Use direct vector-field evaluation instead of Jacobian linearization
dyn.pd.perturbation(
    adata, "GATA1", [-100],
    emb_basis="umap",
    pertubation_method="f_x_prime_minus_f_x_0",  # note: one 'r' in 'pertubation'
)
```

### Cell-wise quiver visualization

```python
dyn.pl.cell_wise_vectors(
    adata,
    basis="umap_perturbation",
    color="cell_type",
    quiver_size=1,
    save_show_or_return="show",
)
```

## Validation

After `dyn.pd.perturbation(adata, genes, expression, emb_basis='umap')`, confirm:

- `adata.obsm['X_umap_perturbation']` exists and has shape `(n_obs, 2)`
- `adata.obsm['velocity_umap_perturbation']` exists and has shape `(n_obs, 2)`
- `adata.obsm['j_delta_x_perturbation']` exists (or the key matching your `pertubation_method`)
- `adata.obsp['perturbation_transition_matrix']` exists (sparse matrix)
- Terminal log includes: `you can use dyn.pl.streamline_plot(adata, basis='umap_perturbation') to visualize the perturbation vector`

After `dyn.pl.streamline_plot(adata, basis='umap_perturbation')`:

- A matplotlib figure appears with streamlines overlaid on the UMAP perturbation embedding
- Streamlines show directional bias consistent with the expected cell fate diversion

## Constraints

- Jacobian must cover the perturbed genes: if a gene is not in `adata.uns['jacobian_pca']['regulators']`, the Jacobian row for that gene will be zero and the perturbation will have no effect. Run `dyn.vf.jacobian(adata, regulators=gene_list, effectors=gene_list, basis='pca')` with an explicit gene list if needed.
- `expression` can be a scalar or a list; if a list, it must be the same length as `genes`.
- Large absolute expression values (e.g., `±100`) saturate the perturbation in PCA space; they are conventional in the tutorial but not physically bounded.
- Each `dyn.pd.perturbation` call overwrites `adata.obsm['X_{emb_basis}_perturbation']` and `adata.obsm['velocity_{emb_basis}_perturbation']` for that basis. Save or rename intermediate results between scenarios if needed.
- The hematopoiesis dataset uses human gene names (all-caps, e.g., `GATA1`). Confirm gene naming convention for non-hematopoiesis datasets.

## Resource Map

- Read `references/source-grounding.md` for full inspected signatures and storage key details.
- Read `references/branch-selection.md` for `pertubation_method` and `perturb_mode` branch options.
- Read `references/source-notebook-map.md` to trace `502_perturbation_tutorial.ipynb` sections to skill resources.
- Read `references/compatibility.md` for known API traps, including the `pertubation_method` typo and Jacobian prerequisite.
- Use `assets/acceptance.json` for the bounded smoke path used by local acceptance.
