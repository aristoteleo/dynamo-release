from tqdm import tqdm
#from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import itertools, functools
import numpy as np
import numdifftools as nd
from .utils import (
    timeit,
    get_pd_row_column_idx,
    vector_field_function,
    _from_adata,
    con_K,
)
from .sampling import sample_by_velocity, trn


def grad(f, x):
    """Gradient of scalar-valued function f evaluated at x"""
    return nd.Gradient(f)(x)


def laplacian(f, x):
    """Laplacian of scalar field f evaluated at x"""
    hes = nd.Hessdiag(f)(x)
    return sum(hes)


def Jacobian_rkhs_gaussian(x, vf_dict, vectorize=False):
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.

        Arguments
        ---------
        x: :class:`~numpy.ndarray`
            Coordinates where the Jacobian is evaluated.
        vf_dict: dict
            A dictionary containing RKHS vector field control points, Gaussian bandwidth,
            and RKHS coefficients.
            Essential keys: 'X_ctrl', 'beta', 'C'

        Returns
        -------
        J: :class:`~numpy.ndarray`
            Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
            d is the number of dimensions and n the number of coordinates in x.
    """
    if x.ndim == 1: 
        K, D = con_K(x[None, :], vf_dict['X_ctrl'], vf_dict['beta'], return_d=True)
        J = (vf_dict['C'].T * K) @ D[0].T
    elif not vectorize:
        n, d = x.shape
        J = np.zeros((d, d, n))
        for i, xi in enumerate(x):
            K, D = con_K(xi[None, :], vf_dict['X_ctrl'], vf_dict['beta'], return_d=True)
            J[:, :, i] = (vf_dict['C'].T * K) @ D[0].T
    else:
        K, D = con_K(x, vf_dict['X_ctrl'], vf_dict['beta'], return_d=True)
        if K.ndim == 1: K = K[None, :]
        J = np.einsum('nm, mi, njm -> ijn', K, vf_dict['C'], D)

    return -2 * vf_dict['beta'] * J


def Jacobian_rkhs_gaussian_parallel(x, vf_dict, cores=None):
    n = len(x)
    if cores is None: cores = mp.cpu_count()
    n_j_per_core = int(np.ceil(n / cores))
    xx = []
    for i in range(0, n, n_j_per_core):
        xx.append(x[i:i+n_j_per_core])
    with mp.Pool(cores) as p:
        ret = p.starmap(Jacobian_rkhs_gaussian, zip(xx, itertools.repeat(vf_dict)))
    ret = [np.transpose(r, axes=(2, 0, 1)) for r in ret]
    ret = np.transpose(np.vstack(ret), axes=(1, 2, 0))
    return ret


def Jacobian_numerical(f, input_vector_convention='row'):
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
def elementwise_jacobian_transformation(fjac, X, qi, qj, return_J=False):
    """Inverse transform low dimension Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to original space.
    The formula used to inverse transform Jacobian matrix calculated from low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes only one row from Q to form qi or qj.

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
        return_J: `bool` (default: `False`)
            Whether to return the raw tensor of Jacobian matrix of each cell before transformation.

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

    if return_J:
        return ret, Js
    else:
        return ret

@timeit
def subset_jacobian_transformation(fjac, X, Qi, Qj, cores=1, return_J=False):
    """Transform Jacobian matrix (:math:`\partial F_i / \partial x_j`) from PCA space to the original space.
    The formula used for transformation:
                                            :math:`\hat{J} = Q J Q^T`,
    where `Q, J, \hat{J}` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes multiple rows from Q to form Qi or Qj.

    Parameters
    ----------
        fjac: `function`:
            The function for calculating numerical Jacobian matrix.
        X: `np.ndarray`:
            The samples coordinates with dimension n_obs x n_PCs, from which Jacobian will be calculated.
        Qi: `np.ndarray`:
            Sampled genes' PCA loading matrix with dimension n' x n_PCs, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        Qj: `np.ndarray`
            Sampled genes' (sample genes can be the same as those in Qi or different) PCs loading matrix with dimension
            n' x n_PCs, from which local dimension Jacobian matrix (k x k) will be inverse transformed back to high dimension.
        cores: `int` (default: 1):
            Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.
        return_J: `bool` (default: `False`)
            Whether to return the raw tensor of Jacobian matrix of each cell before transformation.

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

    if cores == 1:
        #for i in tqdm(range(n), desc='Transforming subset Jacobian'):
        #    J = Js[:, :, i]
        #    ret[:, :, i] = Qi @ J @ Qj.T
        ret = transform_jacobian(Js, Qi, Qj, pbar=True)
    else:
        #pool = ThreadPool(cores)
        #res = pool.starmap(pool_cal_J, zip(np.arange(n), itertools.repeat(Js), itertools.repeat(Qi),
        #                              itertools.repeat(Qj), itertools.repeat(ret)))
        #pool.close()
        #pool.join()
        #ret = functools.reduce((lambda a, b: a + b), res)
        if cores is None: cores = mp.cpu_count()
        n_j_per_core = int(np.ceil(n / cores))
        JJ = []
        for i in range(0, n, n_j_per_core):
            JJ.append(Js[:, :, i:i+n_j_per_core])
        with mp.Pool(cores) as p:
            ret = p.starmap(transform_jacobian, zip(JJ, 
                        itertools.repeat(Qi), itertools.repeat(Qj)))
        ret = [np.transpose(r, axes=(2, 0, 1)) for r in ret]
        ret = np.transpose(np.vstack(ret), axes=(1, 2, 0))
    if return_J:
        return ret, Js
    else:
        return ret


def transform_jacobian(Js, Qi, Qj, pbar=False):
    d1, d2, n = Qi.shape[0], Qj.shape[0], Js.shape[2]
    ret = np.zeros((d1, d2, n))
    if pbar:
        iterj = tqdm(range(n), desc='Transforming subset Jacobian')
    else:
        iterj = range(n)
    for i in iterj:
        J = Js[:, :, i]
        ret[:, :, i] = Qi @ J @ Qj.T
    return ret


def vector_field_function_transformation(vf_func, Q):
    """Transform vector field function from PCA space to original space.
    The formula used for transformation:
                                            :math:`\hat{f} = f Q^T`,
    where `Q, f, \hat{f}` are the PCA loading matrix, low dimensional vector field function and the
    transformed high dimensional vector field function.

    Parameters
    ----------
        vf_func: `function`:
            The vector field function.
        Q: `np.ndarray`:
            PCA loading matrix with dimension d x k, where d is the dimension of the original space,
            and k the number of leading PCs.

    Returns
    -------
        ret `np.ndarray`
            The transformed vector field function.

    """
    return lambda x: vf_func.func(x) @ Q.T


def _divergence(f, x):
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)

 
@timeit
def compute_divergence(f_jac, X, vectorize_size=1):
    """Calculate divergence for many samples by taking the trace of a Jacobian matrix.

    vectorize_size is used to control the number of samples computed in each vectorized batch.
        If vectorize_size = 1, there's no vectorization whatsoever.
        If vectorize_size = None, all samples are vectorized.
    """
    n = len(X)
    if vectorize_size is None: vectorize_size = n

    div = np.zeros(n)
    for i in tqdm(range(0, n, vectorize_size), desc="Calculating divergence"):
        J = f_jac(X[i:i+vectorize_size])
        div[i:i+vectorize_size] = np.trace(J)
    return div


def _curl(f, x, method='analytical', VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 3D"""
    if jac is None:
        if method == 'analytical' and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    return np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


def curl2d(f, x, method='analytical', VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 2D"""
    if jac is None:
        if method == 'analytical' and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    curl = jac[1, 0] - jac[0, 1]

    return curl


def curl(adata,
         basis='umap',
         VecFld=None,
         method='analytical',
         ):
    """Calculate Curl for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        VecFld: `dict`
            The true ODE function, useful when the data is generated through simulation.
        method: `str` (default: `analytical`)
            The method that will be used for calculating divergence, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `curl` key in the .obs.
    """

    if VecFld is None:
        VecFld, func = _from_adata(adata, basis)
    else:
        func = lambda x: vector_field_function(x, VecFld)

    X_data = adata.obsm["X_" + basis][:, :2]

    curl = np.zeros((adata.n_obs, 1))

    Jacobian_ = "jacobian" if basis is None else "jacobian_" + basis

    if Jacobian_ in adata.uns_keys():
        Js = adata.uns[Jacobian_]['Jacobian_raw']
        for i in tqdm(range(X_data.shape[0]), f"Calculating curl with the reconstructed vector field on the {basis} basis. "):
            curl[i] = curl2d(func, None, method=method, VecFld=None, jac=Js[:, :, i])
    else:
        for i, x in tqdm(enumerate(X_data), f"Calculating curl with the reconstructed vector field on the {basis} basis. "):
            curl[i] = curl2d(func, x.flatten(), method=method, VecFld=VecFld)

    curl_key = "curl" if basis is None else "curl_" + basis

    adata.obs[curl_key] = curl


def divergence(adata,
               cell_idx=None,
               sampling='velocity',
               sample_ncells=1000,
               basis='pca',
               VecFld=None,
               method='analytical',
               ):
    """Calculate divergence for each cell with the reconstructed vector field function.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        basis: `str` or None (default: `umap`)
            The embedding data in which the vector field was reconstructed.
        VecFld: `dict`
            The true ODE function, useful when the data is generated through simulation.
        method: `str` (default: `analytical`)
            The method that will be used for calculating divergence, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `divergence` key in the .obs.
    """

    if VecFld is None:
        VecFld, func = _from_adata(adata, basis)
    else:
        func = lambda x: vector_field_function(x, VecFld)

    X, V = VecFld['X'], VecFld['V']

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

    Jacobian_ = "jacobian" if basis is None else "jacobian_" + basis

    if Jacobian_ in adata.uns_keys():
        Js = adata.uns[Jacobian_]['Jacobian_raw']
        div = np.arange(len(cell_idx))
        for i, cur_cell_idx in tqdm(enumerate(cell_idx), desc="Calculating divergence"):
            div[i] = np.trace(Js[:, :, i]) if Js.shape[2] == len(cell_idx) else np.trace(Js[:, :, cur_cell_idx])
    else:
        if method == 'analytical':
            fjac = lambda x: Jacobian_rkhs_gaussian(x, VecFld)
        elif method == 'numeric':
            fjac = get_fjac(func)
        else:
            raise NotImplementedError(f"the divergence calculation method {method} is not implemented. Currently only "
                                      f"support `analytical` and `numeric` methods.")

        div = compute_divergence(fjac, X[cell_idx], vectorize_size=1)

    div_key = "divergence" if basis is None else "divergence_" + basis
    adata.obs[div_key] = None
    adata.obs.loc[adata.obs_names[cell_idx], div_key] = div


def jacobian(adata,
             source_genes,
             target_genes,
             cell_idx=None,
             sampling='velocity',
             sample_ncells=1000,
             basis='pca',
             VecFld=None,
             method='analytical',
             cores=1,
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
        VecFld: `dict`
            The true ODE (ordinary differential equations) function, useful when the data is generated through simulation
            with known ODE functions.
        method: `str` (default: `analytical`)
            The method that will be used for calculating Jacobian, either `analytical` or `numeric`. `analytical`
            method will use the analytical form of the reconstructed vector field for calculating Jacobian while
            `numeric` method will use numdifftools for calculation. `analytical` method is much more efficient.
        cores: `int` (default: 1):
            Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            AnnData object that is updated with the `Jacobian` key in the .uns. This is a 3-dimensional tensor with
            dimensions n_obs x n_source_genes x n_target_genes.
    """

    if VecFld is None:
        VecFld, func = _from_adata(adata, basis)
    else:
        func = lambda x: vector_field_function(x, VecFld)

    X, V = VecFld['X'], VecFld['V']

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
    var_df = adata[:, adata.var.use_for_dynamics].var
    source_genes = var_df.index.intersection(source_genes)
    target_genes = var_df.index.intersection(target_genes)

    source_idx, target_idx = get_pd_row_column_idx(var_df, source_genes, "row"), \
                             get_pd_row_column_idx(var_df, target_genes, "row")
    if len(source_genes) == 0 or len(target_genes) == 0:
        raise ValueError(f"the source and target gene list you provided are not in the velocity gene list!")

    PCs_ = "PCs" if basis == 'pca' else "PCs_" + basis
    Jacobian_ = "jacobian" if basis is None else "jacobian_" + basis

    Q = adata.uns[PCs_][:, :X.shape[1]]

    if method == 'analytical':
        Jac_fun = lambda x: Jacobian_rkhs_gaussian(x, VecFld)
    elif method == 'numeric':
        Jac_fun = Jacobian_numerical(func, input_vector_convention='row')
    else:
        raise NotImplementedError(f"the Jacobian matrix calculation method {method} is not implemented. Currently only "
                                  f"support `analytical` and `numeric` methods.")

    if basis is None:
        Jacobian = Jac_fun(X)
    else:
        if len(source_genes) == 1 and len(target_genes) == 1:
            Jacobian, Js = elementwise_jacobian_transformation(Jac_fun, X[cell_idx], Q[target_idx, :].flatten(),
                                                      Q[source_idx, :].flatten(), True, timeit=True)
        else:
            Jacobian, Js = subset_jacobian_transformation(Jac_fun, X[cell_idx], Q[target_idx, :],
                                                 Q[source_idx, :], cores=cores, return_raw_J=True, timeit=True)

    adata.uns[Jacobian_] = {"Jacobian": Jacobian,
                            "Jacobian_raw": Js,
                            "source_gene": source_genes,
                            "target_genes": target_genes,
                            "cell_idx": cell_idx,
                            "sampling": sampling}


