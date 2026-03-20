---
name: dynamo-lineage-appearance-analysis
description: Compare lineage appearance timing and its regulators on a precomputed `dynamo` vector-field `AnnData` using topography, graph potentials, Jacobian, and vector-calculus outputs. Use when checking whether one lineage appears earlier than its peers, curating fixed points, analyzing regulator pairs on a downstream-ready vector field, or adapting `400_tutorial_hsc_dynamo_megakaryocytes_appearance.ipynb`.
---

# Dynamo Lineage Appearance Analysis

## Goal

Analyze whether one lineage appears earlier than related lineages in a precomputed `dynamo` dataset by combining fixed-point curation, transition-graph potentials, regulator-pair Jacobian analysis, and vector-calculus quantities such as speed, divergence, acceleration, and curvature.

## Quick Workflow

1. Inspect whether the `AnnData` already contains the vector-field and transition-graph outputs this workflow depends on.
2. Choose the stage or stages you actually need: topography curation, appearance-order comparison, regulator-pair Jacobian, or vector-calculus follow-up.
3. Use `build_graph(...)`, `div(...)`, and `potential(...)` on the relevant transition matrix instead of inferring lineage order from plots alone.
4. Run `dyn.vf.jacobian(...)` on the regulator and effector genes of interest before interpreting lineage-specific interactions.
5. Compute `speed`, `divergence`, `acceleration`, and `curvature` in the same basis you want to compare across cells.
6. Validate output keys and return types before treating the analysis as complete.

## Interface Summary

- `dyn.sample_data.hematopoiesis(url='https://figshare.com/ndownloader/files/47439635', filename='hematopoiesis.h5ad')` loads the worked example used in this notebook family.
- `dyn.vf.topography(adata, basis='umap', layer=None, X=None, dims=None, n=25, VecFld=None, show_progress=False, **kwargs)` maps fixed points onto the current vector field when they are missing or need remapping.
- `dyn.pl.topography(..., basis='umap', fps_basis='umap', terms=['streamline', 'fixed_points'], quiver_source='raw', fate='both', n=25, ...)` is the main visualization entrypoint.
- `build_graph(adj_mat)`, `div(g)`, and `potential(g, div_neg=None)` are the graph-operator primitives behind the notebook's appearance-order comparison.
- `dyn.vf.jacobian(..., sampling=None, sample_ncells=1000, basis='pca', method='analytical')` computes cell-wise Jacobians from the reconstructed vector field.
- `dyn.pl.jacobian(..., basis='umap', jkey='jacobian', j_basis='pca', layer='M_s', cmap='bwr')` visualizes selected Jacobian entries on an embedding.
- `dyn.vf.speed(..., method='analytical')`, `dyn.vf.divergence(..., sampling=None, basis='pca', method='analytical')`, `dyn.vf.acceleration(..., basis='umap', method='analytical')`, and `dyn.vf.curvature(..., basis='pca', formula=2, method='analytical')` provide the main dynamical quantities used in the notebook.

Read `references/source-grounding.md` before documenting narrower parameter behavior than the current source supports.

## Stage Selection

- Use the fixed-point stage when the task is to inspect or curate candidate attractors and saddles before downstream interpretation.
- Use the graph-potential stage when the task is to compare appearance timing across related lineages with `cosine_transition_matrix` or `fp_transition_rate`.
- Use the Jacobian stage when the task is to test a specific regulator pair, not when the user only wants a lineage-order figure.
- Use the vector-calculus stage when the task is to compare `speed_pca`, `divergence_pca`, `acceleration_pca`, or `curvature_pca` across cell states.
- Keep `basis='umap'` for notebook-style plotting and `basis='pca'` for Jacobian, divergence, and other quantities that rely on the PCA vector field.

Read `references/stage-selection.md` before choosing non-default `method`, `formula`, `fate`, `terms`, or worked-example gene and lineage labels.

## Input Contract

- Expect a processed `AnnData` that already contains a reconstructed vector field, not raw counts alone.
- Expect `adata.uns['VecFld_umap']` and usually `adata.uns['VecFld_pca']`.
- Expect `adata.obsm['X_umap']` for plotting and `adata.uns['PCs']` for PCA-space back-projection.
- Expect `adata.obsp['cosine_transition_matrix']` for the notebook's first potential comparison.
- Expect `adata.obsp['fp_transition_rate']` if you want the Fokker-Planck-style potential branch.
- Expect `adata.var.use_for_dynamics` and `adata.var.use_for_pca` when using Jacobian or PCA-basis vector calculus.
- Treat notebook-specific `obs['cell_type']` labels such as `HSC`, `MEP-like`, `Meg`, `Ery`, and `Bas`, and genes such as `FLI1` and `KLF1`, as worked-example defaults rather than hard requirements.

If the user only has raw counts or only wants upstream preprocessing or velocity fitting, route to a more appropriate skill instead of forcing this downstream analysis workflow.

## Minimal Execution Patterns

For the worked example load and appearance-order comparison:

```python
import dynamo as dyn
from dynamo.tools.graph_operators import build_graph, div, potential

adata = dyn.sample_data.hematopoiesis()

g = build_graph(adata.obsp["cosine_transition_matrix"])
cosine_div = div(g)
adata.obs["cosine_potential"] = potential(g, -cosine_div)

g_fp = build_graph(adata.obsp["fp_transition_rate"])
fp_div = div(g_fp)
adata.obs["potential_fp"] = potential(g_fp, fp_div)
adata.obs["pseudotime_fp"] = -adata.obs["potential_fp"]
```

For fixed-point remapping or manual curation:

```python
dyn.vf.topography(adata, n=750, basis="umap")

Xss = adata.uns["VecFld_umap"]["Xss"]
ftype = adata.uns["VecFld_umap"]["ftype"]
# choose the subset after inspecting the remapped fixed points for this dataset
keep_idx = [...]

adata.uns["VecFld_umap"]["Xss"] = Xss[keep_idx]
adata.uns["VecFld_umap"]["ftype"] = ftype[keep_idx]

dyn.pl.topography(
    adata,
    basis="umap",
    fps_basis="umap",
    color=["cell_type"],
    streamline_alpha=0.9,
    save_show_or_return="return",
)
```

For regulator-pair Jacobian analysis and notebook-style plotting:

```python
genes = ["FLI1", "KLF1"]
dyn.vf.jacobian(adata, regulators=genes, effectors=genes, basis="pca")

dyn.pl.jacobian(
    adata,
    regulators=genes,
    effectors=["FLI1"],
    basis="umap",
    save_show_or_return="return",
)
```

For the vector-calculus follow-up:

```python
basis = "pca"
dyn.vf.speed(adata, basis=basis)
dyn.vf.divergence(adata, basis=basis)
dyn.vf.acceleration(adata, basis=basis)
dyn.vf.curvature(adata, basis=basis, formula=2)
```

For a bounded worked-example smoke check, use the command recorded in `assets/acceptance.json`.

## Validation

Before analysis, check these items:

- `VecFld_umap` exists in `adata.uns`
- `X_umap` exists in `adata.obsm`
- `cosine_transition_matrix` exists in `adata.obsp`
- `fp_transition_rate` exists in `adata.obsp` if the Fokker-Planck branch is needed

After graph-potential comparison, check these items:

- `cosine_potential` or your renamed cosine-potential column exists in `adata.obs`
- `potential_fp` exists in `adata.obs` when you run the `fp_transition_rate` branch
- `pseudotime_fp` exists and is the sign-flipped version used for notebook-style time interpretation

After Jacobian analysis, check these items:

- `jacobian_pca` exists in `adata.uns`
- `adata.uns['jacobian_pca']` includes `jacobian`, `cell_idx`, `regulators`, and `effectors`
- `dyn.pl.jacobian(..., save_show_or_return='return')` returns a `GridSpec`

After vector-calculus analysis, check these items:

- `speed_pca` exists in `adata.obs`
- `divergence_pca` exists in `adata.obs`
- `acceleration_pca` exists in `adata.obs`
- `curvature_pca` exists in `adata.obs`
- `acceleration` and `curvature` layers are created for the `pca` branch

After topography plotting, check these items:

- `adata.uns['VecFld_umap']['Xss']` and `adata.uns['VecFld_umap']['ftype']` still align after any manual curation
- `dyn.pl.topography(..., save_show_or_return='return')` returns an `Axes`

## Constraints

- Do not treat the notebook's fixed-point indices as reusable defaults. They depend on the exact remapped topography, `n`, and current vector-field fit.
- Do not use this skill as a raw-data preprocessing recipe. It assumes a downstream-ready `dynamo` object with vector-field outputs already present.
- Do not recommend `curvature(..., formula=1)` as a normal branch in the current runtime. Source-grounded reviewer execution hit a runtime failure when that branch tried to write a missing matrix into `obsm`.
- Do not recommend `dyn.pl.topography(..., quiver_source='reconstructed')` as a normal branch in the current runtime. Reviewer execution hit an `ImportError` in that branch.
- Prefer the default topography `terms=['streamline', 'fixed_points']` unless the user explicitly needs additional overlays and you have checked the current source behavior.
- Keep notebook cosmetics such as `dyn.pl.style(font_path='Arial')` out of the reusable workflow.

## Resource Map

- Read `references/stage-selection.md` when choosing between topography, appearance-order, Jacobian, and vector-calculus stages.
- Read `references/worked-example.md` when the user specifically wants the HSC / MEP-like / Meg / Ery / Bas and FLI1 / KLF1 setup from the notebook.
- Read `references/source-notebook-map.md` to trace notebook sections into this reusable skill.
- Read `references/source-grounding.md` for inspected signatures, source-level branch behavior, empirical checks, and known runtime traps.
- Read `references/compatibility.md` when notebook prose, parameter names, or current runtime behavior diverge.
- Use `assets/acceptance.json` for the bounded worked-example smoke path used by local validation.
