# Compatibility

Use this reference when notebook wording and current source behavior may not be identical.

## Current API Preference

Prefer:

- `dynamo.preprocessing.Preprocessor`
- `dyn.tl.pseudotime_velocity`
- `dyn.vf.VectorField`
- `dyn.pd.fate`

Do not default to older helper paths unless the user explicitly asks for legacy behavior.

## Source Vs Notebook

- The notebook copies `X_tsne` into `X_umap` for demonstration. The generated skill keeps that as an optional parity branch, not the default workflow.
- Current `pseudotime_velocity(...)` source exposes three real `method` branches: `hodge`, `gradient`, `naive`. The notebook shows only the default path.
- In this runtime, `reduceDimension(...)` can leave `uns['neighbors']` populated while `obsp['distances']` is still missing. The generated skill adds an explicit `neighbors(...)` verification step.
- Current `pseudotime_velocity(...)` reads `adata.layers[ekey].toarray()`. If you alias from dense `adata.X`, convert it to `csr_matrix` first.
- `unspliced_RNA=True` stores output under `add_ukey`, default `M_u_pseudo`; it does not create a canonical `unspliced` layer automatically.

## Runtime Constraints

- Importing `dynamo` in this workspace is most stable in a repository-compatible Python environment with writable matplotlib and numba cache directories.
- `dyn.sample_data.bone_marrow()` may require a network download if the dataset is not cached.
- GIF export and some animation save paths may require external tooling such as `imagemagick`.

## Conservative Rule

If notebook text and current source disagree:

1. trust the current source for executable behavior
2. mention the drift explicitly
3. reopen the notebook only if figure parity or historical reproduction is the real goal
