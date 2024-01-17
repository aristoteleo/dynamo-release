from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.matlib as matlib
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import inv


def DDRTree(
    X: np.ndarray,
    maxIter: int,
    sigma: float,
    gamma: float,
    eps: int = 0,
    dim: int = 2,
    Lambda: float = 1.0,
    ncenter: Optional[int] = None,
    keep_history: bool = False,
) -> Union[
        pd.DataFrame,
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            List[np.ndarray],
        ],
    ]:
    """Provides an implementation of the framework of reversed graph embedding (RGE).

    This function is a python version of the DDRTree algorithm originally written in R.
    (https://cran.r-project.org/web/packages/DDRTree/DDRTree.pdf)

    Args:
        X: The matrix on which DDRTree would be implemented.
        maxIter: The max number of iterations.
        sigma: The bandwidth parameter.
        gamma: Regularization parameter for k-means.
        eps: The threshold of convergency to stop the iteration. Defaults to 0.
        dim: The number of dimensions reduced to. Defaults to 2.
        Lambda: Regularization parameter for inverse praph embedding. Defaults to 1.0.
        ncenter: The number of center genes to be considered. If None, all genes would be considered. Defaults to None.
        keep_history: Whether to keep relative parameters during each iteration and return. Defaults to False.

    Returns:
        A dataframe containing `W`, `Z`, `Y`, `stree`, `R`, `objs` for each iterations if `keep_history` is True.
        Otherwise, a tuple (Z, Y, stree, R, W, Q, C, objs). The items in the tuple is from the last iteration. `Z` is
        the reduced dimension; `Y` is the latent points as the center of Z; `stree` is the smooth tree graph embedded in
        the low dimension space; `R` is used to transform the hard assignments used in K-means into soft assignments;
        `W` is the orthogonal set of d (dimensions) linear basis; `Q` is (I + lambda L)^(-1), where L = diag(B1) - B, a
        Laplacian matrix. `C` equals to XQ^(-1)X^T; `objs` is a list containing convergency conditions during the
        iterations.
    """

    X = np.array(X).T
    (D, N) = X.shape

    # initialization
    W = pca_projection(np.dot(X, X.T), dim)
    Z = np.dot(W.T, X)

    if ncenter is None:
        K = N
        Y = Z.T[0:K].T
    else:
        K = ncenter

        Y, _ = kmeans2(Z.T, K)
        Y = Y.T

    # main loop
    objs = []
    if keep_history:
        history = pd.DataFrame(
            index=[i for i in range(maxIter)],
            columns=["W", "Z", "Y", "stree", "R", "objs"],
        )
    for iter in range(maxIter):

        # Kruskal method to find optimal B
        distsqMU = csr_matrix(sqdist(Y, Y)).toarray()
        stree = minimum_spanning_tree(np.tril(distsqMU)).toarray()
        stree = stree + stree.T
        B = stree != 0
        L = np.diag(sum(B.T)) - B

        # compute R using mean-shift update rule
        distZY = sqdist(Z, Y)
        tem_min_dist = np.array(np.min(distZY, 1)).reshape(-1, 1)
        min_dist = repmat(tem_min_dist, 1, K)
        tmp_distZY = distZY - min_dist
        tmp_R = np.exp(-tmp_distZY / sigma)
        R = tmp_R / repmat(
            np.sum(tmp_R, 1).reshape(-1, 1), 1, K
        )
        Gamma = np.diag(sum(R))

        # termination condition
        obj1 = -sigma * sum(
            np.log(np.sum(np.exp(-tmp_distZY / sigma), 1)) - tem_min_dist.T[0] / sigma
        )
        xwz = np.linalg.norm(X - np.dot(W, Z), 2)
        objs.append(
            (np.dot(xwz, xwz))
            + Lambda * np.trace(np.dot(Y, np.dot(L, Y.T)))
            + gamma * obj1
        )
        print("iter = ", iter, "obj = ", objs[iter])

        if keep_history:
            history.iloc[iter]["W"] = W
            history.iloc[iter]["Z"] = Z
            history.iloc[iter]["Y"] = Y
            history.iloc[iter]["stree"] = stree
            history.iloc[iter]["R"] = R

        if iter > 0:
            if abs((objs[iter] - objs[iter - 1]) / abs(objs[iter - 1])) < eps:
                break

        # compute low dimension projection matrix
        tmp = np.dot(
            R,
            inv(
                csr_matrix(
                    ((gamma + 1) / gamma) * ((Lambda / gamma) * L + Gamma)
                    - np.dot(R.T, R)
                )
            ).toarray(),
        )
        Q = (1 / (gamma + 1)) * (eye(N, N) + np.dot(tmp, R.T))
        C = np.dot(X, Q)

        tmp1 = np.dot(C, X.T)
        W = pca_projection((tmp1 + tmp1.T) / 2, dim)
        Z = np.dot(W.T, C)
        Y = np.dot(
            np.dot(Z, R), inv(csr_matrix((Lambda / gamma) * L + Gamma)).toarray()
        )

    if keep_history:
        history.iloc[iter]["obs"] = objs

        return history
    else:
        return Z, Y, stree, R, W, Q, C, objs


def cal_ncenter(ncells: int, ncells_limit: int = 100) -> int:
    """Calculate the number of cells to be most significant in the reduced space.

    Args:
        ncells: Total number of cells.
        ncells_limit: The max number of cells to be considered. Defaults to 100.

    Returns:
        The number of cells to be most significant in the reduced space. 
    """    

    res = np.round(
        2 * ncells_limit * np.log(ncells) / (np.log(ncells) + np.log(ncells_limit))
    )

    return res


def pca_projection(C: np.ndarray, L: int) -> np.ndarray:
    """Solve the problem size(C) = NxN, size(W) = NxL. max_W trace( W' C W ) : W' W = I	

    Args:
        C: The matrix to calculate eigenvalues.
        L: The number of Eigenvalues.

    Returns:
        The L largest Eigenvalues. 
    """

    V, U = eig(C)
    eig_idx = np.argsort(V).tolist()
    eig_idx.reverse()
    W = U.T[eig_idx[0:L]].T
    return W


def sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the square distance between `a` and `b`. 

    Args:
        a: A matrix with dimension D x N
        b: A matrix with dimension D x N

    Returns:
        A numeric value for the difference between a and b. 
    """

    aa = np.sum(a ** 2, axis=0)
    bb = np.sum(b ** 2, axis=0)
    ab = a.T.dot(b)

    aa_repmat = matlib.repmat(aa[:, None], 1, b.shape[1])
    bb_repmat = matlib.repmat(bb[None, :], a.shape[1], 1)

    dist = abs(aa_repmat + bb_repmat - 2 * ab)

    return dist


def repmat(X: np.ndarray, m: int, n: int) -> np.ndarray:
    """This function returns an array containing m (n) copies of A in the row (column) dimensions.

    The size of B is size(A)*n when A is a matrix. For example, repmat(np.matrix(1:4), 2, 3) returns a 4-by-6 matrix. 

    Args:
        X: An array like matrix.
        m: Number of copies on row dimension.
        n: Number of copies on column dimension.

    Returns:
        The constructed repmat. 
    """

    xy_rep = matlib.repmat(X, m, n)

    return xy_rep


def eye(m: int, n: int) -> np.ndarray:
    """Equivalent of eye (matlab).

    Return a m x n matrix with 0th diagonal to be 1 and the rest to be 0.

    Args:
        m: Number of rows.
        n: Number of columns.

    Returns:
        The m x n eye matrix.
    """
    mat = np.eye(m, n)
    return mat
