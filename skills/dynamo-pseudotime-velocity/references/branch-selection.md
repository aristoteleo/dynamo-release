# Branch Selection

This notebook mixes preprocessing, embedding reuse, pseudotime-to-velocity conversion, vector-field analysis, topology, fate, and animation. Use this reference to keep those stages separate instead of replaying every cell literally.

## Default Job

If the user asks to run a pseudotime-derived `dynamo` velocity workflow without measured spliced / unspliced kinetics:

1. preprocess with `recipe='monocle'`
2. compute or validate a 2D basis such as UMAP
3. explicitly materialize `obsp['distances']` if it is missing
4. create `M_s`
5. run `dyn.tl.pseudotime_velocity(..., method='hodge', dynamics_info=True)`
6. add `VectorField` and `fate` only if the goal needs them

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

- the notebook reuses the preprocessing family from the earlier dynamo tutorial
- downstream embedding, neighbor-graph, and velocity handoff work cleanly on the monocle path

## Stage 2: Embedding And Neighbor Graph

Relevant branch-heavy parameter on `dyn.tl.reduceDimension(...)`:

- `reduction_method`: `trimap`, `diffusion_map`, `tsne`, `umap`, `psl`, `sude`

Recommendation for this workflow family:

- use `reduction_method='umap'` for the default analytical path

When to switch:

- use an existing 2D embedding only when the user explicitly wants notebook-style parity with precomputed coordinates
- use `tsne` or another branch only when the user specifically asks for that embedding and understands downstream plots will key off the matching `basis`

Important runtime rule:

- after `reduceDimension(...)`, check `adata.obsp`
- if `distances` is missing, run `dyn.tl.neighbors(adata, basis='pca')` before `pseudotime_velocity(...)`

## Stage 3: Pseudotime To Velocity

Relevant branch-heavy parameters on `dyn.tl.pseudotime_velocity(...)`:

- `method`: `hodge`, `gradient`, `naive`
- `dynamics_info`: `False` / `True`
- `unspliced_RNA`: `False` / `True`

Recommendation for this workflow family:

- use `method='hodge'`
- use `dynamics_info=True` for the reusable skill path
- keep `unspliced_RNA=False` unless the user explicitly wants pseudo-unspliced outputs

When to switch:

- use `method='gradient'` when the user explicitly wants the alternate graph-gradient transition construction
- use `method='naive'` only for explicit branch comparison or legacy parity debugging
- use `dynamics_info=False` only when notebook parity matters more than downstream compatibility metadata

Important storage rules:

- `ekey` defaults to `M_s`
- `vkey` defaults to `velocity_S`
- transition output is stored under `adata.obsp['pseudotime_transition_matrix']`
- `unspliced_RNA=True` stores the pseudo-unspliced result under `add_ukey`, default `M_u_pseudo`

## Stage 4: Vector Field

Relevant branch-heavy parameters on `dyn.vf.VectorField(...)`:

- `method`: `SparseVFC`, `dynode`
- `pot_curl_div`: `False` / `True`

Recommendation for this workflow family:

- use `basis='umap'`, `method='SparseVFC'`

When to switch:

- use `pot_curl_div=True` when the user wants potential, curl, divergence, or `topography`
- use `method='dynode'` only when `dynode` is installed and explicitly requested

About `M`:

- the notebook uses `M=1000`
- current source forwards `M` through `**kwargs`, not the public signature

## Stage 5: Fate Prediction

Relevant branch-heavy parameters on `dyn.pd.fate(...)`:

- `direction`: `forward`, `backward`, `both`
- `average`: `False`, `True`, `origin`, `trajectory`
- `sampling`: `arc_length`, `logspace`, `uniform_indices`
- `inverse_transform`: `False` / `True`

Recommendation for this workflow family:

- use `basis='umap'`
- use `direction='forward'`
- use `average=False`
- use `sampling='arc_length'`
- use `inverse_transform=False`

When to switch:

- use `direction='both'` only when the user wants both ancestry and descendants from the same start states
- use `average='trajectory'` only when the user explicitly wants averaged trajectories over multiple initial cells
- use `sampling='uniform_indices'` only when the user accepts that multiprocessing is disabled

## Decision Rule

If the user asks for:

- "the notebook result" -> run the default job, but treat bone marrow labels and the `X_tsne` reuse cell as worked-example branches
- "just pseudotime-derived velocity" -> stop after `pseudotime_velocity(...)`
- "potential" or "fixed points" -> add `VectorField(..., pot_curl_div=True)` and `topography`
- "fate" -> run `VectorField` first, then `pd.fate(...)`
- "animation" -> run `fate` first, then the visualization / animation entrypoints
