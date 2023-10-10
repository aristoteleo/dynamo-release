import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from sklearn.utils.sparsefuncs import mean_variance_axis

from ..configuration import DKM
from ..dynamo_logger import main_info_insert_adata_obsm, main_info_insert_adata_var


def _truncatedSVD_with_center(
    X: Union[csc_matrix, csr_matrix],
    n_components: int = 30,
    random_state: int = 0,
) -> Dict:
    """Center a sparse matrix and perform truncated SVD on it.

    It uses `scipy.sparse.linalg.LinearOperator` to express the centered sparse
    input by given matrix-vector and matrix-matrix products. Then truncated
    singular value decomposition (SVD) can be solved without calculating the
    individual entries of the centered matrix. The right singular vectors after
    decomposition represent the principal components. This function is inspired
    by the implementation of scanpy (https://github.com/scverse/scanpy).

    Args:
        X: The input sparse matrix to perform truncated SVD on.
        n_components: The number of components to keep. Default is 30.
        random_state: Seed for the random number generator. Default is 0.

    Returns:
        The transformed input matrix and a sklearn PCA object containing the
        right singular vectors and amount of variance explained by each
        principal component.
    """
    random_state = check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    v0 = random_state.uniform(-1, 1, np.min(X.shape))
    n_components = min(n_components, X.shape[1] - 1)

    mean = X.mean(0)
    X_H = X.T.conj()
    mean_H = mean.T.conj()
    ones = np.ones(X.shape[0])[None, :].dot

    # Following callables implements different type of matrix calculation.
    def matvec(x):
        """Matrix-vector multiplication. Performs the operation X_centered*x
        where x is a column vector or an 1-D array."""
        return X.dot(x) - mean.dot(x)

    def matmat(x):
        """Matrix-matrix multiplication. Performs the operation X_centered*x
        where x is a matrix or ndarray."""
        return X.dot(x) - mean.dot(x)

    def rmatvec(x):
        """Adjoint matrix-vector multiplication. Performs the operation
        X_centered^H * x where x is a column vector or an 1-d array."""
        return X_H.dot(x) - mean_H.dot(ones(x))

    def rmatmat(x):
        """Adjoint matrix-matrix multiplication. Performs the operation
        X_centered^H * x where x is a matrix or ndarray."""
        return X_H.dot(x) - mean_H.dot(ones(x))

    # Construct the LinearOperator with callables above.
    X_centered = LinearOperator(
        shape=X.shape,
        matvec=matvec,
        matmat=matmat,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
        dtype=X.dtype,
    )

    # Solve SVD without calculating individuals entries in LinearOperator.
    U, Sigma, VT = svds(X_centered, solver="arpack", k=n_components, v0=v0)
    Sigma = Sigma[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])
    X_transformed = U * Sigma
    components_ = VT
    exp_var = np.var(X_transformed, axis=0)
    _, full_var = mean_variance_axis(X, axis=0)
    full_var = full_var.sum()

    result_dict = {
        "X_pca": X_transformed,
        "components_": components_,
        "explained_variance_ratio_": exp_var / full_var,
    }

    fit = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    X_pca = result_dict["X_pca"]
    fit.mean_ = mean.A1.flatten()
    fit.components_ = result_dict["components_"]
    fit.explained_variance_ratio_ = result_dict["explained_variance_ratio_"]

    return fit, X_pca


def _pca_fit(
    X: np.ndarray,
    pca_func: Callable,
    n_components: int = 30,
    **kwargs,
) -> Tuple:
    """Apply PCA to the input data array X using the specified PCA function.

    Args:
        X: the input data array of shape (n_samples, n_features).
        pca_func: the PCA function to use, which should have a 'fit' and
            'transform' method, such as the PCA class or the IncrementalPCA
            class from sklearn.decomposition.
        n_components: the number of principal components to compute. If
            n_components is greater than or equal to the number of features in
            X, it will be set to n_features - 1 to avoid overfitting.
        **kwargs: any additional keyword arguments that will be passed to the
            PCA function.

    Returns:
        A tuple containing two elements:
            - The fitted PCA object, which has a 'fit' and 'transform' method.
            - The transformed array X_pca of shape (n_samples, n_components).
    """
    fit = pca_func(
        n_components=min(n_components, X.shape[1] - 1),
        **kwargs,
    ).fit(X)
    X_pca = fit.transform(X)
    return fit, X_pca


def pca(
    adata: AnnData,
    X_data: np.ndarray = None,
    n_pca_components: int = 30,
    pca_key: str = "X_pca",
    pcs_key: str = "PCs",
    genes_to_append: Union[List[str], None] = None,
    layer: Union[List[str], str, None] = None,
    svd_solver: Literal["randomized", "arpack"] = "randomized",
    random_state: int = 0,
    use_truncated_SVD_threshold: int = 500000,
    use_incremental_PCA: bool = False,
    incremental_batch_size: Optional[int] = None,
    return_all: bool = False,
) -> Union[AnnData, Tuple[AnnData, Union[PCA, TruncatedSVD], np.ndarray]]:
    """Perform PCA reduction for monocle recipe.

    When large dataset is used (e.g. 1 million cells are used), Incremental PCA
    is recommended to avoid the memory issue. When cell number is less than half
    a million, by default PCA or _truncatedSVD_with_center (use sparse matrix
    that doesn't explicitly perform centering) will be used. TruncatedSVD is the
    fastest method. Unlike other methods which will center the data first,  it
    performs SVD decomposition on raw input. Only use this when dataset is too
    large for other methods.

    Args:
        adata: an AnnData object.
        X_data: the data to perform dimension reduction on. Defaults to None.
        n_pca_components: number of PCA components reduced to. Defaults to 30.
        pca_key: the key to store the reduced data. Defaults to "X".
        pcs_key: the key to store the principle axes in feature space. Defaults
            to "PCs".
        genes_to_append: a list of genes should be inspected. Defaults to None.
        layer: the layer(s) to perform dimension reduction on. Would be
            overrided by X_data. Defaults to None.
        svd_solver: the svd_solver to solve svd decomposition in PCA.
        random_state: the seed used to initialize the random state for PCA.
        use_truncated_SVD_threshold: the threshold of observations to use
            truncated SVD instead of standard PCA for efficiency.
        use_incremental_PCA: whether to use Incremental PCA. Recommend enabling
            incremental PCA when dataset is too large to fit in memory.
        incremental_batch_size: The number of samples to use for each batch when
            performing incremental PCA. If batch_size is None, then batch_size
            is inferred from the data and set to 5 * n_features.
        return_all: whether to return the PCA fit model and the reduced array
            together with the updated AnnData object. Defaults to False.

    Raises:
        ValueError: layer provided is not invalid.
        ValueError: list of genes to append is invalid.

    Returns:
        The updated AnnData object with reduced data if `return_all` is False.
        Otherwise, a tuple (adata, fit, X_pca), where adata is the updated
        AnnData object, fit is the fit model for dimension reduction, and X_pca
        is the reduced array, will be returned.
    """

    # only use genes pass filter (based on use_for_pca) to perform dimension reduction.
    if X_data is None:
        if "use_for_pca" not in adata.var.keys():
            adata.var["use_for_pca"] = True

        if layer is None:
            X_data = adata.X[:, adata.var.use_for_pca.values]
        else:
            if "X" in layer:
                X_data = adata.X[:, adata.var.use_for_pca.values]
            elif "total" in layer:
                X_data = adata.layers["X_total"][:, adata.var.use_for_pca.values]
            elif "spliced" in layer:
                X_data = adata.layers["X_spliced"][:, adata.var.use_for_pca.values]
            elif "protein" in layer:
                X_data = adata.obsm["X_protein"]
            elif type(layer) is str:
                X_data = adata.layers["X_" + layer][:, adata.var.use_for_pca.values]
            else:
                raise ValueError(
                    f"your input layer argument should be either a `str` or a list that includes one of `X`, "
                    f"`total`, `protein` element. `Layer` currently is {layer}."
                )

        cm_genesums = X_data.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        bad_genes = np.where(adata.var.use_for_pca)[0][~valid_ind]
        if genes_to_append is not None and len(adata.var.index[bad_genes].intersection(genes_to_append)) > 0:
            raise ValueError(
                f"The gene list passed to argument genes_to_append contains genes with no expression "
                f"across cells or non finite values. Please check those genes:"
                f"{set(bad_genes).intersection(genes_to_append)}!"
            )

        adata.var.iloc[bad_genes, adata.var.columns.tolist().index("use_for_pca")] = False
        X_data = X_data[:, valid_ind]

    if use_incremental_PCA:
        from sklearn.decomposition import IncrementalPCA

        fit, X_pca = _pca_fit(
            X_data,
            pca_func=IncrementalPCA,
            n_components=n_pca_components,
            batch_size=incremental_batch_size,
        )
    else:
        if adata.n_obs < use_truncated_SVD_threshold:
            if not issparse(X_data):
                fit, X_pca = _pca_fit(
                    X_data,
                    pca_func=PCA,
                    n_components=n_pca_components,
                    svd_solver=svd_solver,
                    random_state=random_state,
                )
            else:
                fit, X_pca = _truncatedSVD_with_center(
                    X_data,
                    n_components=n_pca_components,
                    random_state=random_state,
                )
        else:
            # TruncatedSVD is the fastest method we have. It doesn't center the
            # data. It only performs SVD decomposition, which is the second part
            # in our _truncatedSVD_with_center function.
            fit, X_pca = _pca_fit(
                X_data, pca_func=TruncatedSVD, n_components=n_pca_components + 1, random_state=random_state
            )
            # first columns is related to the total UMI (or library size)
            X_pca = X_pca[:, 1:]

    main_info_insert_adata_obsm(pca_key, log_level=20)
    adata.obsm[pca_key] = X_pca

    if use_incremental_PCA or adata.n_obs < use_truncated_SVD_threshold:
        adata.uns[pcs_key] = fit.components_.T
        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_
    else:
        # first columns is related to the total UMI (or library size)
        adata.uns[pcs_key] = fit.components_.T[:, 1:]
        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

    adata.uns["pca_mean"] = fit.mean_ if hasattr(fit, "mean_") else np.zeros(X_data.shape[1])

    if return_all:
        return adata, fit, X_pca
    else:
        return adata


def pca_genes(PCs: list, n_top_genes: int = 100) -> np.ndarray:
    """For each gene, if the gene is n_top in some principle component then it is valid. Return all such valid genes.

    Args:
        PCs: principle components(PC) of PCA
        n_top_genes: number of gene candidates in EACH PC. Defaults to 100.

    Returns:
        A bool array indicating whether the gene is valid.
    """

    valid_genes = np.zeros(PCs.shape[0], dtype=bool)
    for q in PCs.T:
        sorted_q = np.sort(np.abs(q))[::-1]
        is_pc_top_n = np.abs(q) > sorted_q[n_top_genes]
        valid_genes = np.logical_or(is_pc_top_n, valid_genes)
    return valid_genes


def top_pca_genes(
    adata: AnnData,
    pc_key: str = "PCs",
    n_top_genes: int = 100,
    pc_components: Union[int, None] = None,
    adata_store_key: str = "top_pca_genes",
) -> AnnData:
    """Define top genes as any gene that is ``n_top_genes`` in some principle component.

    Args:
        adata: an AnnData object.
        pc_key: component key stored in adata.uns. Defaults to "PCs".
        n_top_genes: number of top genes as valid top genes in each component. Defaults to 100.
        pc_components: number of top principle components to use. Defaults to None.
        adata_store_key: the key for storing pca genes. Defaults to "top_pca_genes".

    Raises:
        Exception: invalid pc_key.

    Returns:
        The AnnData object with top genes stored as values of adata.var[adata_store_key].
    """

    if pc_key in adata.uns.keys():
        Q = adata.uns[pc_key]
    elif pc_key in adata.varm.keys():
        Q = adata.varm[pc_key]
    else:
        raise Exception(f"No PC matrix {pc_key} found in neither .uns nor .varm.")
    if pc_components is not None:
        if type(pc_components) == int:
            Q = Q[:, :pc_components]
        elif type(pc_components) == list:
            Q = Q[:, pc_components]

    pcg = pca_genes(Q, n_top_genes=n_top_genes)
    genes = np.zeros(adata.n_vars, dtype=bool)
    if DKM.VAR_USE_FOR_PCA in adata.var.keys():
        genes[adata.var[DKM.VAR_USE_FOR_PCA]] = pcg
    else:
        genes = pcg
    main_info_insert_adata_var(adata_store_key, indent_level=2)
    adata.var[adata_store_key] = genes
    return adata
