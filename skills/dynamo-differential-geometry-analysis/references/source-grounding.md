# Source Grounding

This skill was generated from live code inspection and reviewer-run execution in the current repository, not from notebook narration alone.

## Primary Source Files

Notebook inspected:

- `docs/tutorials/notebooks/403_Differential_geometry.ipynb`

Code inspected:

- `dynamo/preprocessing/Preprocessor.py`
- `dynamo/preprocessing/pca.py`
- `dynamo/vectorfield/VectorField.py`
- `dynamo/vectorfield/vector_calculus.py`
- `dynamo/vectorfield/rank_vf.py`
- `dynamo/vectorfield/networks.py`
- `dynamo/external/hodge.py`
- `dynamo/plot/time_series.py`
- `dynamo/prediction/state_graph.py`
- `dynamo/data_io.py`

## Inspected Interfaces

### `Preprocessor.preprocess_adata`

Observed signature:

```python
(self, adata, recipe='monocle', tkey=None, experiment_type=None) -> None
```

Observed `recipe` branches:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

### `dyn.vf.VectorField`

Observed signature:

```python
(adata, basis=None, layer=None, dims=None, genes=None, normalize=False, grid_velocity=False, grid_num=50, velocity_key='velocity_S', method='SparseVFC', min_vel_corr=0.6, restart_num=5, restart_seed=[0, 100, 200, 300, 400], model_buffer_path=None, return_vf_object=False, map_topography=False, pot_curl_div=False, cores=1, result_key=None, copy=False, n=25, **kwargs)
```

Observed `method` branches:

- `SparseVFC`
- `dynode`

Important source detail:

- notebook argument `M=50` or `M=1000` is forwarded through `**kwargs`, not exposed in the public signature

### `dyn.vf.acceleration`

Observed signature:

```python
(adata, basis='umap', vector_field_class=None, Qkey='PCs', method='analytical', **kwargs)
```

Observed `method` branches:

- `analytical`
- `numerical`

Observed PCA-basis storage:

- `adata.obs['acceleration_pca']`
- `adata.obsm['acceleration_pca']`
- `adata.layers['acceleration']`

### `dyn.vf.curvature`

Observed signature:

```python
(adata, basis='pca', vector_field_class=None, formula=2, Qkey='PCs', method='analytical', **kwargs)
```

Observed branch-heavy parameters:

- `formula`: `1`, `2`
- `method`: `analytical`, `numerical`

Observed PCA-basis storage:

- `adata.obs['curvature_pca']`
- `adata.obsm['curvature_pca']`
- `adata.layers['curvature']`

### `dyn.vf.jacobian`

Observed signature:

```python
(adata, regulators=None, effectors=None, cell_idx=None, sampling=None, sample_ncells=1000, basis='pca', Qkey='PCs', vector_field_class=None, method='analytical', store_in_adata=True, **kwargs)
```

Observed `sampling` branches:

- `None`
- `random`
- `velocity`
- `trn`

Observed `method` branches:

- `analytical`
- `numerical`

Observed storage:

- `adata.uns['jacobian_pca']`
- `adata.obs['jacobian_det_pca']`

### `dyn.vf.rank_jacobian_genes`

Observed signature:

```python
(adata, groups=None, jkey='jacobian_pca', abs=False, mode='full reg', exclude_diagonal=False, normalize=False, return_df=False, **kwargs)
```

Observed `mode` branches:

- `full reg` or `full_reg`
- `full eff` or `full_eff`
- `reg`
- `eff`
- `int`
- `switch`

### `dyn.vf.rank_divergence_genes`

Observed signature:

```python
(adata, jkey='jacobian_pca', genes=None, prefix_store='rank_div_gene', **kwargs)
```

Important source detail:

- this helper ranks diagonal Jacobian elements after `jacobian(...)`
- it requires matching regulators and effectors in the stored Jacobian

### `dyn.pp.top_pca_genes`

Observed signature:

```python
(adata, pc_key='PCs', n_top_genes=100, pc_components=None, adata_store_key='top_pca_genes')
```

Observed storage:

- writes a boolean mask into `adata.var['top_pca_genes']`

### `dyn.ext.ddhodge`

Observed signature:

```python
(adata, X_data=None, layer=None, basis='pca', n=30, VecFld=None, adjmethod='graphize_vecfld', distance_free=False, n_downsamples=5000, up_sampling=True, sampling_method='velocity', seed=19491001, enforce=False, cores=1, **kwargs)
```

Observed branch-heavy parameters:

- `adjmethod`: `graphize_vecfld`, `naive`
- `sampling_method`: `random`, `velocity`, `trn`

Observed storage:

- `adata.obsp['pca_ddhodge']`
- `adata.obs['pca_ddhodge_div']`
- `adata.obs['pca_ddhodge_potential']`

### `dyn.pl.kinetic_heatmap`

Observed signature includes:

```python
(adata, genes, mode='vector_field', basis=None, layer='X', project_back_to_high_dim=True, tkey='potential', dist_threshold=1e-10, color_map='BrBG', gene_order_method='maximum', ..., save_show_or_return='show', ...)
```

Observed branch-heavy parameters:

- `mode`: `vector_field`, `lap`, `pseudotime`
- `gene_order_method`: `maximum`, `half_max_ordering`, `raw`

Important source detail:

- current source writes to `adata.uns['kinetics_heatmap']`

### `dyn.pd.state_graph`

Observed signature:

```python
(adata, group, method='vf', transition_mat_key='pearson_transition_matrix', approx=False, eignum=5, basis='umap', layer=None, arc_sample=False, sample_num=100, prune_graph=False, **kwargs)
```

Observed `method` branches:

- `vf`
- `markov`
- `naive`

Observed storage:

- `adata.uns['<group>_graph']`

### Export Helpers

Observed signatures:

```python
export_rank_xlsx(adata, path='rank_info.xlsx', ext='excel', rank_prefix='rank') -> None
cleanup(adata, del_prediction=False, del_2nd_moments=False) -> AnnData
```

Important source detail:

- `cleanup(...)` removes `adata.uns['kinetics_heatmap']` if present

## Reviewer-Run Empirical Checks

### Worked Example Data State

Observed on `dyn.sample_data.zebrafish()` in the current runtime:

- shape `4181 x 16940`
- `.layers` includes `spliced`, `unspliced`
- `.obsm` is empty before preprocessing
- `.obsp` is empty before preprocessing
- `.uns` is empty before preprocessing
- observation names are not unique

This matters because the notebook wording reads like a processed example, but the current runtime provides raw conventional data.

### Stable Bootstrap And Geometry Path

Reviewer execution succeeded on a `118`-cell subset spanning:

- `Proliferating Progenitor`
- `Pigment Progenitor`
- `Melanophore`
- `Unknown`

Executed sequence:

1. `obs_names_make_unique()`
2. `Preprocessor(...).preprocess_adata(..., recipe='monocle')`
3. `dynamics(..., cores=1)`
4. `reduceDimension(...)`
5. `cell_velocities(adata)` for the embedding
6. `cell_velocities(adata, basis='pca', transition_genes=adata.var.use_for_pca.values)`
7. `VectorField(adata, basis='pca', M=50)`
8. `acceleration(adata, basis='pca')`
9. `curvature(adata, basis='pca')`
10. `top_pca_genes(adata, n_top_genes=20)`
11. `jacobian(..., sampling='trn', sample_ncells=60, basis='pca')`
12. `rank_jacobian_genes(..., mode='full_reg')`
13. `build_network_per_cluster(...)`
14. `ddhodge(adata, basis='pca')`
15. `state_graph(adata, group='Cell_type', basis='pca', method='vf', sample_num=15)`
16. `kinetic_heatmap(..., mode='pseudotime', tkey='pca_ddhodge_potential', save_show_or_return='return')`

Observed outputs:

- `velocity_umap`
- `velocity_pca`
- `pearson_transition_matrix`
- `VecFld_pca`
- `acceleration`
- `curvature`
- `jacobian_pca`
- `pca_ddhodge`
- `pca_ddhodge_potential`
- `Cell_type_graph`
- `kinetics_heatmap`

### Important Runtime Trap

Reviewer execution found a stable PCA projection fix:

- `cell_velocities(adata, basis='pca')` can fail on smaller subsets
- `cell_velocities(adata, basis='pca', transition_genes=adata.var.use_for_pca.values)` succeeded reliably on the checked subset

That fix is encoded in the skill because it came from live execution, not notebook prose.
