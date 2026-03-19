---
name: dynamo-differential-geometry-analysis
description: Run downstream differential-geometry analysis on a `dynamo` vector-field `AnnData`, including velocity, acceleration, curvature, Jacobian, regulatory-network, ddhodge pseudotime, and state-graph branches. Use when adapting the `403_Differential_geometry.ipynb` tutorial, extending a conventional spliced/unspliced RNA velocity workflow into vector calculus, or choosing among `method`, `mode`, `sampling`, `formula`, `adjmethod`, or `gene_order_method` branches.
---

# Dynamo Differential Geometry Analysis

## Goal

Turn a conventional `dynamo` RNA-velocity dataset into reusable downstream differential-geometry outputs in PCA space, then optionally continue into Jacobian-based regulatory-network analysis, ddhodge pseudotime, kinetic heatmaps, and state graphs without treating the zebrafish notebook as the skill identity.

## Quick Workflow

1. Inspect whether the user already has a downstream-ready `AnnData` with `velocity_pca` and `VecFld_pca`, or whether raw conventional preprocessing and velocity fitting still need to happen.
2. If starting from raw spliced / unspliced counts, preprocess with `recipe='monocle'`, fit dynamics, build the embedding, and compute both low-dimensional and PCA-space velocities.
3. Fit `dyn.vf.VectorField(..., basis='pca')` before calling acceleration, curvature, Jacobian, or ddhodge on the PCA basis.
4. Run the stage the user actually needs: scalar/vector geometry, Jacobian and ranking, regulatory-network extraction, or ddhodge pseudotime and state-graph analysis.
5. Validate the concrete storage keys produced by each stage before interpreting plots or rankings.
6. Export ranking tables or clean the object only after all downstream calculations that depend on `kinetics_heatmap`, fate, or vector-field internals are finished.

## Interface Summary

- `Preprocessor.preprocess_adata(adata, recipe='monocle', tkey=None, experiment_type=None)` is the conventional preprocessing wrapper.
- `dyn.tl.reduceDimension(..., reduction_method='umap')` is the default embedding path; current source also exposes `trimap`, `diffusion_map`, `tsne`, `psl`, and `sude`.
- `dyn.tl.cell_velocities(..., basis=None)` is the low-dimensional and PCA-space velocity entrypoint. In the current runtime, the `basis='pca'` branch may need an explicit `transition_genes` argument on smaller subsets.
- `dyn.vf.VectorField(..., basis='pca', method='SparseVFC', pot_curl_div=False, **kwargs)` reconstructs the PCA-space vector field used by downstream differential geometry.
- `dyn.vf.acceleration(..., basis='pca', method='analytical')` and `dyn.vf.curvature(..., basis='pca', formula=2, method='analytical')` generate per-cell and gene-space geometry outputs.
- `dyn.vf.jacobian(..., sampling=None, sample_ncells=1000, basis='pca', method='analytical')` computes cell-wise Jacobians; `dyn.vf.rank_jacobian_genes(..., mode=...)` and `dyn.vf.build_network_per_cluster(...)` turn them into grouped rankings and edge lists.
- `dyn.ext.ddhodge(..., basis='pca', adjmethod='graphize_vecfld', sampling_method='velocity')` creates vector-field pseudotime and potential, while `dyn.pl.kinetic_heatmap(..., mode='vector_field' | 'lap' | 'pseudotime')` and `dyn.pd.state_graph(..., method='vf' | 'markov' | 'naive')` consume those outputs.
- `dyn.export_rank_xlsx(..., rank_prefix='rank')` exports ranking tables from `.uns`, and `dyn.cleanup(..., del_prediction=False, del_2nd_moments=False)` strips heavyweight internals before save.

Read `references/source-grounding.md` before documenting narrower branch behavior than the current source supports.

## Stage Selection

- Use the bootstrap stage when the user starts from raw conventional spliced / unspliced data instead of an already fitted vector field.
- Use the geometry-ranking stage when the user wants `velocity_S`, `acceleration`, or `curvature` rankings by cell group.
- Use the Jacobian-network stage when the user wants regulator and effector ranking, interaction ranking, or a cluster-specific edge list.
- Use the ddhodge-state stage when the user wants vector-field pseudotime, kinetic heatmaps, or cell-state transition graphs.
- Keep `basis='pca'` for vector-field reconstruction, Jacobian, divergence-like ranking, ddhodge, and notebook-style state graphs. Use `basis='umap'` mainly for display or upstream velocity plots.
- Keep `method='SparseVFC'`, `curvature(..., formula=2)`, `jacobian(..., method='analytical')`, `ddhodge(..., adjmethod='graphize_vecfld')`, and `state_graph(..., method='vf')` as the default reusable path unless the user explicitly asks for a different branch.

Read `references/stage-selection.md` before choosing non-default `recipe`, `method`, `formula`, `sampling`, `mode`, `adjmethod`, or `gene_order_method` branches.

## Input Contract

- Expect an `AnnData` with conventional spliced / unspliced layers if the user wants the full bootstrap path.
- Expect `adata.obsm['X_pca']`, `adata.var['use_for_pca']`, and `adata.layers['velocity_S']` before PCA-space vector-field analysis.
- Expect `adata.obsm['velocity_pca']` and `adata.uns['VecFld_pca']` before calling acceleration, curvature, Jacobian, ddhodge, or `state_graph(..., method='vf', basis='pca')`.
- Expect a meaningful grouping column such as `adata.obs['Cell_type']` before using ranking helpers or `build_network_per_cluster(...)`.
- Expect `adata.uns['PCs']` or `adata.varm['PCs']` before `top_pca_genes(...)` or any PCA-basis inverse transform.
- Treat notebook-specific zebrafish cell-type labels and genes such as `tfec` and `pnp4a` as worked-example defaults, not hard requirements.

If the user only wants upstream preprocessing or only wants pseudotime-derived velocity without conventional kinetics, route to a more appropriate skill instead of forcing this downstream analysis path.

## Minimal Execution Patterns

For the default bootstrap from raw conventional zebrafish-style data:

```python
import dynamo as dyn

adata = dyn.sample_data.zebrafish()
adata.obs_names_make_unique()

pre = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
pre.preprocess_adata(adata, recipe="monocle")

dyn.tl.dynamics(adata, cores=1)
dyn.tl.reduceDimension(adata)
dyn.tl.cell_velocities(adata)
dyn.tl.cell_velocities(
    adata,
    basis="pca",
    transition_genes=adata.var.use_for_pca.values,
)

dyn.vf.VectorField(adata, basis="pca", M=50, cores=1)
```

For geometry and grouped ranking:

```python
dyn.vf.rank_velocity_genes(adata, groups="Cell_type", vkey="velocity_S")

dyn.vf.acceleration(adata, basis="pca")
dyn.vf.rank_acceleration_genes(adata, groups="Cell_type", akey="acceleration")

dyn.vf.curvature(adata, basis="pca", formula=2)
dyn.vf.rank_curvature_genes(adata, groups="Cell_type", ckey="curvature")
```

For Jacobian ranking and cluster-specific network extraction:

```python
dyn.pp.top_pca_genes(adata, n_top_genes=100)
genes = adata.var_names[adata.var["top_pca_genes"]][:50].tolist()

dyn.vf.jacobian(
    adata,
    regulators=genes,
    effectors=genes,
    sampling="trn",
    sample_ncells=200,
    basis="pca",
)

full_reg = dyn.vf.rank_jacobian_genes(
    adata,
    groups="Cell_type",
    mode="full_reg",
    abs=True,
    return_df=True,
)

reg_rank = dyn.vf.rank_jacobian_genes(
    adata,
    groups="Cell_type",
    mode="reg",
    abs=True,
    output_values=True,
    return_df=True,
)

edges = dyn.vf.build_network_per_cluster(
    adata,
    cluster="Cell_type",
    cluster_names=["Unknown"],
    full_reg_rank=full_reg,
    genes=genes[:20],
    n_top_genes=10,
    abs=True,
)
```

For ddhodge pseudotime, kinetic heatmaps, and state graphs:

```python
dyn.ext.ddhodge(adata, basis="pca", sampling_method="velocity")

transition_genes = adata.var_names[adata.var["top_pca_genes"]][:20].tolist()

heat = dyn.pl.kinetic_heatmap(
    adata,
    genes=transition_genes,
    basis="pca",
    mode="pseudotime",
    tkey="pca_ddhodge_potential",
    gene_order_method="maximum",
    save_show_or_return="return",
)

dyn.pd.state_graph(
    adata,
    group="Cell_type",
    basis="pca",
    method="vf",
    sample_num=30,
)
```

For export and cleanup after ranking:

```python
dyn.export_rank_xlsx(adata, path="result/rank_info.xlsx", rank_prefix="rank")
dyn.cleanup(adata, del_prediction=False, del_2nd_moments=False)
```

## Validation

After bootstrap preprocessing, check these items:

- `adata.obsm["X_pca"]` exists
- `adata.obsm["X_umap"]` exists if you ran the default embedding path
- `adata.var["use_for_pca"]` and `adata.var["use_for_dynamics"]` exist
- `adata.layers["velocity_S"]` exists after `cell_velocities(...)`

After PCA velocity and vector-field fitting, check these items:

- `adata.obsm["velocity_pca"]` exists
- `adata.uns["VecFld_pca"]` exists
- `adata.uns["VecFld_pca"]` includes `X`, `Y`, and vector-field parameters

After geometry ranking, check these items:

- `adata.layers["acceleration"]` exists
- `adata.layers["curvature"]` exists
- `adata.obs["acceleration_pca"]` and `adata.obs["curvature_pca"]` exist
- ranking tables were written into `.uns` with `rank` or `rank_abs` prefixes

After Jacobian and network analysis, check these items:

- `adata.uns["jacobian_pca"]` exists
- `adata.uns["jacobian_pca"]` includes `jacobian_gene`, `cell_idx`, `regulators`, and `effectors`
- `full_reg` contains one table per requested group
- `edges[group_name]` is a `DataFrame` with `regulator`, `target`, and `weight`

After ddhodge, heatmap, and state graph, check these items:

- `adata.obsp["pca_ddhodge"]` exists
- `adata.obs["pca_ddhodge_potential"]` exists
- `adata.uns["kinetics_heatmap"]` exists after `kinetic_heatmap(...)`
- `adata.uns["Cell_type_graph"]` exists after `state_graph(...)`

After cleanup and export, check these items:

- the Excel file exists at the requested path
- `cleanup(...)` removed `kinetics_heatmap` if it existed
- `cleanup(...)` did not remove outputs you still need for downstream interpretation

## Constraints

- Do not assume `dyn.sample_data.zebrafish()` is already processed in the current runtime. Reviewer execution saw raw counts with `spliced` and `unspliced` layers but no downstream embedding or vector-field outputs.
- Call `adata.obs_names_make_unique()` on the zebrafish worked example before concatenation or heavy preprocessing. Reviewer execution saw non-unique observation names.
- Do not assume `dyn.tl.cell_velocities(adata, basis='pca')` is stable on small subsets without extra guidance. Reviewer execution needed `transition_genes=adata.var.use_for_pca.values` to avoid a PCA-projection failure.
- Do not treat `jacobian(...)` as a whole-transcriptome default on large datasets. Narrow with `top_pca_genes(...)`, explicit regulators and effectors, or cell sampling first.
- Do not imply that `rank_divergence_genes(...)` computes geometric divergence in the usual trace sense. Current source ranks diagonal Jacobian elements after `jacobian(...)`.
- `ddhodge(..., adjmethod='naive')` depends on a preexisting transition matrix. Keep `adjmethod='graphize_vecfld'` unless the user explicitly wants the alternate branch and already has the needed adjacency.
- `kinetic_heatmap(...)` stores results under `adata.uns["kinetics_heatmap"]`, and `cleanup(...)` removes that key.
- `VectorField(..., method='dynode')` is a real branch but requires an additional backend. Do not recommend it by default.

## Resource Map

- Read `references/stage-selection.md` when choosing among bootstrap, geometry ranking, Jacobian/network, and ddhodge/state-graph stages.
- Read `references/jacobian-and-network-analysis.md` when the user wants Jacobian ranking modes, divergence-like ranking, or cluster-specific regulatory edges.
- Read `references/pseudotime-and-state-graph.md` when the user wants `ddhodge`, `kinetic_heatmap`, or `state_graph`.
- Read `references/source-grounding.md` for inspected signatures, source-level branch evidence, and reviewer-run empirical execution notes.
- Read `references/source-notebook-map.md` to trace `403_Differential_geometry.ipynb` into this reusable skill layout.
- Read `references/compatibility.md` when notebook prose and current runtime behavior diverge.
- Use `assets/acceptance.json` for the bounded smoke path used by local acceptance.
