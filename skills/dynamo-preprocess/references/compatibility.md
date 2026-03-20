# Compatibility

Use this reference when notebook wording and current source behavior may not be identical.

## Current API Preference

Prefer:

- `dynamo.preprocessing.Preprocessor`

Do not default to older helpers such as:

- `dynamo.pp.recipe_monocle`

unless the user explicitly asks for legacy behavior.

## Source Vs Notebook

- The notebook is centered on current `Preprocessor` usage, but live source now clearly exposes the extra `recipe` branch `monocle_pearson_residuals`.
- The generated skill therefore includes that branch even though it is not a main notebook section.

## Runtime Constraints

- `sctransform` requires `KDEpy`.
- `dyn.sample_data.zebrafish()` may need a network or cached dataset in the runtime environment.
- importing dynamo in this workspace is most stable in a repository-compatible Python environment with writable cache directories for matplotlib and numba.

## Conservative Rule

If notebook text and current source disagree:

1. trust the current source for executable behavior
2. mention the drift explicitly
3. reopen the notebook only if figure parity or historical reproduction is the real goal
