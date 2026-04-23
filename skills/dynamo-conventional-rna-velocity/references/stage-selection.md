# Stage Selection

This notebook mixes preprocessing, kinetics, projection, vector-field analysis, topology, differential geometry, and animation. Use this reference to keep those stages separate instead of blindly replaying all cells or overfitting the skill to the zebrafish sample.

## Default Job

If the user asks to run a conventional spliced/unspliced `dynamo` velocity workflow:

1. preprocess with `recipe='monocle'`
2. run `dyn.tl.dynamics(..., model='stochastic')`
3. run `dyn.tl.reduceDimension(...)`
4. run `dyn.tl.cell_velocities(..., basis='umap', method='pearson')`
5. run `dyn.vf.VectorField(..., basis='umap')`

Only add later stages if the user’s goal needs them.

## Stage 1: Preprocessing

Current source branches for `Preprocessor.preprocess_adata(...)`:

- `monocle`
- `seurat`
- `sctransform`
- `pearson_residuals`
- `monocle_pearson_residuals`

Recommendation for this workflow family:

- use `monocle` by default

Reason:

- the tutorial is conventional spliced / unspliced scRNA-seq
- downstream velocity and vector-field analysis rely on the monocle-style normalized layers and PCA path

## Stage 2: Kinetics With `dyn.tl.dynamics`

Relevant branch-heavy parameters:

- `model`: `auto`, `deterministic`, `stochastic`
- `est_method`: `ols`, `rlm`, `ransac`, `gmm`, `negbin`, `auto`, `twostep`, `direct`
- `one_shot_method`: `combined`, `sci-fate`, `sci_fate`

Recommendation for this workflow family:

- use `model='stochastic'`
- let `est_method='auto'` resolve unless the user explicitly asks otherwise

Observed runtime result on the zebrafish worked example:

- `experiment_type='conventional'`
- `model='stochastic'`
- `est_method='gmm'`

When to switch:

- use `model='deterministic'` for faster legacy-style RNA velocity parity
- use `est_method='negbin'` only if the user explicitly wants that steady-state branch

## Stage 3: Velocity Projection

Relevant branch-heavy parameters on `dyn.tl.cell_velocities(...)`:

- `basis`
- `method`: `kmc`, `fp`, `cosine`, `pearson`, `transform`

Recommendation for this workflow family:

- use `basis='umap'`, `method='pearson'` for notebook parity

When to switch:

- use `method='kmc'` if the user wants the newer Itô-kernel style projection
- use `method='cosine'` for closer legacy velocyto-style behavior
- avoid `method='transform'` unless the user explicitly wants UMAP-transform projection and accepts the warning that it is not recommended
- use `basis='pca'` for the later vector-calculus stage

Important storage rule:

- the transition matrix key is kernel-specific, for example `pearson_transition_matrix`

## Stage 4: Vector Field

Relevant branch-heavy parameters on `dyn.vf.VectorField(...)`:

- `basis`
- `method`: `SparseVFC`, `dynode`
- `pot_curl_div`: `False` / `True`

Recommendation for this workflow family:

- use `basis='umap'`, `method='SparseVFC'`

When to switch:

- use `basis='pca'` for speed / divergence / acceleration / curvature
- use `pot_curl_div=True` when the user wants potential, curl, or divergence keys added automatically
- use `method='dynode'` only if the environment has `dynode` and the user explicitly wants that branch

About `M`:

- the notebook uses `M=1000`
- current source passes `M` through `**kwargs` into SparseVFC defaults where its default is `None`

## Stage 5: Confidence And Correction

Relevant branch-heavy parameter:

- `dyn.tl.cell_wise_confidence(..., method=...)`

Current signature values:

- `cosine`
- `consensus`
- `correlation`
- `jaccard`
- `hybrid`
- `divergence`

Recommendation for this workflow family:

- start with `jaccard`
- use `gene_wise_confidence(...)` and `confident_cell_velocities(...)` only when debugging suspicious flow directions or applying lineage priors

## Stage 6: Fate Prediction

Relevant branch-heavy parameters on `dyn.pd.fate(...)`:

- `direction`: `forward`, `backward`, `both`
- `average`: `False`, `True`, `origin`, `trajectory`
- `sampling`: `arc_length`, `logspace`, `uniform_indices`
- `inverse_transform`: `False` / `True`

Recommendation for this workflow family:

- use `basis='umap'`
- use `direction='forward'`
- use `average=False`
- use `inverse_transform=False`
- keep `sampling='arc_length'` unless the user explicitly wants another path

## Decision Rule

If the user asks for:

- "the notebook result" -> run the default job through UMAP vector field, then apply the notebook's dataset-specific labels as a worked example
- "potential" or "fixed points" -> add `pot_curl_div=True` and `topography`
- "speed / acceleration / curvature" -> switch to the PCA vector-calculus stage
- "animation" -> run `fate` first, then animation entrypoints
