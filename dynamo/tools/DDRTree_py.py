import numpy as np
import pandas as pd
import numpy.matlib as matlib
from scipy.linalg import eig
from scipy.cluster.vq import kmeans2
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix


def cal_ncenter(ncells, ncells_limit=100):

    res = np.round(
        2 * ncells_limit * np.log(ncells) / (np.log(ncells) + np.log(ncells_limit))
    )

    return res


def pca_projection(C, L):
    """solve the problem size(C) = NxN, size(W) = NxL. max_W trace( W' C W ) : W' W = I	
    Arguments	
    ---------	
    C: (ndarrya) The matrix of	
    L: (int) The number of Eigenvalues	
    Return	
    ------	
    W: The L largest Eigenvalues	
    """

    V, U = eig(C)
    eig_idx = np.argsort(V).tolist()
    eig_idx.reverse()
    W = U.T[eig_idx[0:L]].T
    return W


def sqdist(a, b):
    """calculate the square distance between a, b	
    Arguments	
    ---------	
        a: 'np.ndarray'	
            A matrix with :math:`D \times N` dimension	
        b: 'np.ndarray'	
            A matrix with :math:`D \times N` dimension	
    Returns	
    -------	
    dist: 'np.ndarray'	
        A numeric value for the different between a and b	
    """
    aa = np.sum(a ** 2, axis=0)
    bb = np.sum(b ** 2, axis=0)
    ab = a.T.dot(b)

    aa_repmat = matlib.repmat(aa[:, None], 1, b.shape[1])
    bb_repmat = matlib.repmat(bb[None, :], a.shape[1], 1)

    dist = abs(aa_repmat + bb_repmat - 2 * ab)

    return dist


def repmat(X, m, n):
    """This function returns an array containing m (n) copies of A in the row (column) dimensions. The size of B is	
    size(A)*n when A is a matrix.For example, repmat(np.matrix(1:4), 2, 3) returns a 4-by-6 matrix.	
    Arguments	
    ---------	
        X: 'np.ndarray'	
            An array like matrix.	
        m: 'int'	
            Number of copies on row dimension	
        n: 'int'	
            Number of copies on column dimension	
    Returns	
    -------	
    xy_rep: 'np.ndarray'	
        A matrix of repmat	
    """
    xy_rep = matlib.repmat(X, m, n)

    return xy_rep


def eye(m, n):
    """Equivalent of eye (matlab)	
    Arguments	
    ---------	
        m: 'int'	
            Number of rows	
        n: 'int'	
            Number of columns	
    Returns	
    -------	
    mat: 'np.ndarray'	
        A matrix of eye	
    """
    mat = np.eye(m, n)
    return mat


def DDRTree(
    X, maxIter, sigma, gamma, eps=0, dim=2, Lambda=1.0, ncenter=None, keep_history=False
):
    """	This function is a pure Python implementation of the DDRTree algorithm.

    Arguments	
    ---------	
        X : DxN:'np.ndarray'	
            data matrix list	
        maxIter : maximum iterations	
        eps: 'int'	
                relative objective difference	
        dim: 'int'	
                reduced dimension	
        Lambda: 'float'	
                regularization parameter for inverse graph embedding	
        sigma: 'float'	
                bandwidth parameter	
        gamma:'float'	
                regularization parameter for k-means	
        ncenter :(int)	
    Returns	
    -------	
        history: 'DataFrame'	
                the results dataframe of return	
    """
    X = np.array(X)
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
        )  ##########################3
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
