from tqdm import tqdm
import numpy as np
import numdifftools as nd
from .utils import timeit, get_pd_row_column_idx
from .sampling import sample_by_velocity, trn

def grad(f, x):
    """Gradient of scalar-valued function f evaluated at x"""
    return nd.Gradient(f)(x)


def laplacian(f, x):
    """Laplacian of scalar field f evaluated at x"""
    hes = nd.Hessdiag(f)(x)
    return sum(hes)


def get_fjac(f, input_vector_convention='row'):
    '''
        Get the numerical Jacobian of the vector field function.
        If the input_vector_convention is 'row', it means that fjac takes row vectors
        as input, otherwise the input should be an array of column vectors. Note that
        the returned Jacobian would behave exactly the same if the input is an 1d array.

        The column vector convention is slightly faster than the row vector convention.
        So the matrix of row vector convention is converted into column vector convention
        under the hood.

        No matter the input vector convention, the returned Jacobian is of the following
        format:
                df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
                df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
                df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
                ...         ...         ...         ...
    '''
    fjac = nd.Jacobian(lambda x: f(x.T).T)
    if input_vector_convention == 'row' or input_vector_convention == 0:
        def f_aux(x):
            x = x.T
            return fjac(x)

        return f_aux
    else:
        return fjac


@timeit
def elementwise_jacobian_transformation(fjac, X, qi, qj):
    """Inverse transform low dimension Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to original space.
    The formula used to inverse transform Jacobian matrix calculated from low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function only take one element from Q to form qi or qj.

    Parameters
    ----------
        fjac: `function`:
            The function for calculating numerical Jacobian matrix.
        X: `np.ndarray`:
            The samples coordinates with dimension n_obs x n_PCs, from which Jacobian will be calculated.
        Qi: `np.ndarray`:
            One sampled gene's PCs loading matrix with dimension n' x n_PCs, from which local dimension Jacobian matrix
            (k x k) will be inverse transformed back to high dimension.
        Qj: `np.ndarray`
            Another gene's (can be the same as those in Qi or different) PCs loading matrix with dimension  n' x n_PCs,
            from which local dimension Jacobian matrix (k x k) will be inverse transformed back to high dimension.

    Returns
    -------
        ret `np.ndarray`
            The calculated vector of Jacobian matrix (:math:`\partial F_i / \partial x_j`) for each cell.
    """

    Js = fjac(X)
    ret = np.zeros(len(X))
    for i in tqdm(range(len(X)), "calculating Jacobian for each cell"):
        J = Js[:, :, i]
        ret[i] = qi @ J @ qj
    return ret


@timeit
def subset_jacobian_transformation(fjac, X, Qi, Qj):
    """Inverse transform low dimension Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to original space.
    The formula used to inverse transform Jacobian matrix calculated from low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes multiple elements from Q to form Qi or Qj.

    Parameters
    ----------
        fjac: `function`:
            The function for calculating numerical Jacobian matrix.
        X: `np.ndarray`:
            The samples coordinates with dimension n_obs x n_PCs, from which Jacobian will be calculated.
        Qi: `np.ndarray`:
            Sampled genes' PCs loading matrix with dimension n' x n_PCs, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        Qj: `np.ndarray`
            Sampled genes' (sample genes can be the same as those in Qi or different) PCs loading matrix with dimension
            n' x n_PCs, from which local dimension Jacobian matrix (k x k) will be inverse transformed back to high dimension.

    Returns
    -------
        ret `np.ndarray`
            The calculated Jacobian matrix (n_gene x n_gene x n_obs) for each cell.

    """
    X = np.atleast_2d(X)
    Qi = np.atleast_2d(Qi)
    Qj = np.atleast_2d(Qj)
    d1, d2, n = Qi.shape[0], Qj.shape[0], X.shape[0]

    Js = fjac(X)
    ret = np.zeros((d1, d2, n))
    for i in tqdm(range(n), desc='Transforming subset Jacobian'):
        J = Js[:, :, i]
        ret[:, :, i] = Qi @ J @ Qj.T
    return ret


def _divergence(f, x):
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)


@timeit
def compute_divergence(f_jac, X, vectorize=True):
    """calculate divergence for many samples by taking the trace of a Jacobian matrix"""
    if vectorize:
        J = f_jac(X)
        div = np.trace(J)
    else:
        div = np.zeros(len(X))
        for i in tqdm(range(len(X)), desc="Calculating divergence"):
            J = f_jac(X[i])
            div[i] = np.trace(J)

    return div


def _curl(f, x):
    """Curl of the reconstructed vector field f evaluated at x in 3D"""
    jac = nd.Jacobian(f)(x)
    return np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


def curl2d(f, x):
    """Curl of the reconstructed vector field f evaluated at x in 2D"""
    jac = nd.Jacobian(f)(x)
    curl = jac[1, 0] - jac[0, 1]

    return curl


def curl(adata,
         basis='umap',
         vecfld_dict=None,
         ):
    """Calculate Curl for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vecfld_dict: `dict`
            The true ODE function, useful when the data is generated through simulation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `curl` key in the .obs.
    """

    if vecfld_dict is None:
        vf_key = 'VecFld' if basis is None else 'VecFld_' + basis
        if vf_key not in adata.uns.keys():
            raise ValueError(f"Your adata doesn't have the key for Vector Field with {basis} basis. "
                             f"Try firstly running dyn.tl.VectorField(adata, basis={basis}).")

        vecfld_dict = adata.uns[vf_key]

    X_data = adata.obsm["X_" + basis]

    curl = np.zeros((adata.n_obs, 1))
    func = vecfld_dict['func']

    for i, x in tqdm(enumerate(X_data), f"Calculating curl with the reconstructed vector field on the {basis} basis. "):
        curl[i] = curl2d(func, x.flatten())

    adata.obs['curl'] = curl


def divergence(adata,
               cell_idx=None,
               sampling='velocity',
               sample_ncells=1000,
               basis='pca',
               vecfld_dict=None,
               ):
    """Calculate divergence for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        vecfld_dict: `dict`
            The true ODE function, useful when the data is generated through simulation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `divergence` key in the .obs.
    """

    vf_key = 'VecFld' if basis is None else 'VecFld_' + basis

    if vecfld_dict is None:
        if vf_key not in adata.uns.keys():
            raise ValueError(f"Your adata doesn't have the key for Vector Field with {basis} basis."
                             f"Try firstly running dyn.tl.VectorField(adata, basis={basis}).")

        vecfld_dict = adata.uns[vf_key]

    X, V = adata.uns[vf_key]['VecFld']['X'], adata.uns[vf_key]['VecFld']['V']

    if basis == 'umap': cell_idx = np.arange(adata.n_obs)

    if cell_idx is None:
        if sampling == 'velocity':
            cell_idx = sample_by_velocity(V, sample_ncells)
        elif sampling == 'trn':
            cell_idx = trn(X, sample_ncells)
        else:
            raise NotImplementedError(f"the sampling method {sampling} is not implemented. Currently only support velocity "
                                      f"based (velocity) or topology representing network (trn) based sampling.")

    cell_idx = np.arange(adata.n_obs) if cell_idx is None else cell_idx

    func = vecfld_dict['func']

    div = compute_divergence(get_fjac(func), X[cell_idx], vectorize=True)

    adata.obs['divergence'] = None
    adata.obs.loc[adata.obs_names[cell_idx], 'divergence'] = div


def jacobian(adata,
             source_genes,
             target_genes,
             cell_idx=None,
             sampling='velocity',
             sample_ncells=1000,
             basis='pca',
             vecfld_dict=None,
             input_vector_convention='row',
             ):
    """Calculate Jacobian for each cell with the reconstructed vector field function.

    If the vector field was reconstructed from the reduced PCA space, the Jacobian matrix will then be inverse
    transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be
    supported shortly. Note that calculation of Jacobian matrix is computationally expensive, thus by default, only
    1000 cells sampled (using velocity magnitude based sampling) for calculation.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        source_genes: `list`
            The list of genes that will be used as regulators when calculating the cell-wise Jacobian matrix. Each of
            those genes' partial derivative will be placed in the denominator of each element of the Jacobian matrix.
            It can be used to access how much effect the increase of those genes will affect the change of the velocity
            of the target genes (see below).
        target_genes: `List` or `None` (default: `None`)
            The list of genes that will be used as targets when calculating the cell-wise Jacobian matrix. Each of
            those genes' velocities' partial derivative will be placed in the numerator of each element of the Jacobian
            matrix. It can be used to access how much effect the velocity of the target genes will receive when increasing
            the expression of the source genes (see above).
        cell_idx: `np.ndarray` or None (default: `None`)
            One-dimension numpy array or list that represents the indices of the samples that will be used for calculating
            Jacobian matrix.
        sampling: `str` or None (default: `velocity`)
           Which sampling method for cell sampling. When it is `velocity`, it will use the magnitude of velocity for
           sampling cells; when it is `trn`, the topology representing network based method will be used to sample cells
           with the low dimensional embeddings. The `velocity` based sampling ensures sampling evenly for cells with both
           low (corresponds to densely populated stable cell states) and high velocity (corresponds to less populated
           transient cell states) cells. The `trn` based sampling ensures the global topology of cell states will be well
           maintained after the sampling.
        sample_ncells: `int` (default: `100`)
            Total number of cells to be sampled.
        basis: `str` or None (default: `pca`)
            The embedding data in which the vector field was reconstructed. If `None`, use the vector field function that
            was reconstructed directly from the original unreduced gene expression space.
        vecfld_dict: `dict`
            The true ODE (ordinary differential equations) function, useful when the data is generated through simulation
            with known ODE functions.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `Jacobian` key in the .uns. This is a 3-dimensional tensor with dimensions
            n_obs x n_source_genes x n_target_genes.
    """

    vf_key = 'VecFld' if basis is None else 'VecFld_' + basis

    if vecfld_dict is None:
        if vf_key not in adata.uns.keys():
            raise ValueError(f"Your adata doesn't have the key for Vector Field with {basis} basis."
                             f"Try firstly running dyn.tl.VectorField(adata, basis={basis}).")

        vecfld_dict = adata.uns[vf_key]

    X, V = adata.uns[vf_key]['VecFld']['X'], adata.uns[vf_key]['VecFld']['V']

    if basis == 'umap': cell_idx = np.arange(adata.n_obs)

    if cell_idx is None:
        if sampling == 'velocity':
            cell_idx = sample_by_velocity(V, sample_ncells)
        elif sampling == 'trn':
            cell_idx = trn(X, sample_ncells)
        else:
            raise NotImplementedError(f"the sampling method {sampling} is not implemented. Currently only support velocity "
                                      f"based (velocity) or topology representing network (trn) based sampling.")

    cell_idx = np.arange(adata.n_obs) if cell_idx is None else cell_idx

    if type(source_genes) == str: source_genes = [source_genes]
    if type(target_genes) == str: target_genes = [target_genes]
    var_df = adata[:, adata.var.use_for_dynamo].var
    source_genes = var_df.index.intersection(source_genes)
    target_genes = var_df.index.intersection(target_genes)

    source_idx, target_idx = get_pd_row_column_idx(var_df, source_genes, "row"), \
                             get_pd_row_column_idx(var_df, target_genes, "row")
    if len(source_genes) == 0 or len(target_genes) == 0:
        raise ValueError(f"the source and target gene list you provided are not in the velocity gene list!")

    PCs_ = "PCs" if basis == 'pca' else "PCs_" + basis
    Jacobian_ = "jacobian" #if basis is None else "jacobian_" + basis

    Q, func = adata.varm[PCs_][:, :X.shape[1]], vecfld_dict['func']

    Jac_fun = get_fjac(func, input_vector_convention)

    if basis is None:
        Jacobian = Jac_fun(X)
    else:
        if len(source_genes) == 1 and len(target_genes) == 1:
            Jacobian = elementwise_jacobian_transformation(Jac_fun, X[cell_idx], Q[target_idx, :].flatten(),
                                                      Q[source_idx, :].flatten(), timeit=True)
        else:
            Jacobian = subset_jacobian_transformation(Jac_fun, X[cell_idx], Q[target_idx, :],
                                                 Q[source_idx, :], timeit=True)

    adata.uns[Jacobian_] = {"Jacobian": Jacobian,
                            "source_gene": source_genes,
                            "target_genes": target_genes,
                            "cell_idx": cell_idx,
                            "sampling": sampling}


