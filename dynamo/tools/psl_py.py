import numpy as np
import numpy.matlib as matlib

import scipy.spatial as ss
import scipy.sparse

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

# from scikits.sparse.cholmod import cholesky

# use for convert list of list to a list (https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists)
import functools
import operator


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


def diag_mat(values):
    """Equivalent of diag (matlab)	
    Arguments	
    ---------	
        values: 'int'	
            dim of the matrix	
    Returns	
    -------	
        mat: 'np.ndarray'	
            A diag_matrix	
    """
    # mat = np.zeros((len(values),len(values)))
    # for i in range(len(values)):
    #     mat[i][i] = values[i]
    mat = np.zeros((len(values), len(values)))
    np.fill_diagonal(mat, values)

    return mat


def psl(
    Y, sG=None, dist=None, K=10, C=1e3, param_gamma=1e-3, d=2, maxIter=10, verbose=False
):
    """This function is a pure Python implementation of the PSL algorithm.

    Reference: Li Wang and Qi Mao, Probabilistic Dimensionality Reduction via Structure Learning. T-PAMI, VOL. 41, NO. 1, JANUARY 2019	

    Arguments
    ---------	
        Y: 'numpy.ndarray'	
            data list	
        sG: 'scipy.sparse.csr_matrix'	
            a prior kNN graph passed to the algorithm	
        dist: 'numpy.ndarray'	
            a dense distance matrix between all vertices. If no distance matrix passed, we will use the kNN based algorithm,	
            otherwise we will use the original algorithm reported in the manuscript.	
        K: 'int'	
            number of nearest neighbors used to build the neighborhood graph. Large k can obtain less sparse structures.	
            Ignored if sG is used.	
        C: 'int'	
            The penalty parameter for loss term. It controls the preservation of distances. The larger it is, the distance	
            is more strictly preserve. If the structure is very clear, a larger C  is preferred.	
        param_gamma: 'int'	
            param_gamma is trying to make a matrix A nonsingular, it is like a round-off parameter. 1e-4 or 1e-5 is good.	
            It corresponds to the variance of prior embedding.	
        d: 'int'	
            embedding dimension	
        maxIter: 'int'	
            Number of maximum iterations	
        verbose: 'bool'	
            Whether to print running information	

    Returns
    -------	
        (S,Z): 'tuple'	
            a tuple of the adjacency matrix and the reduced low dimension embedding.	
    """

    if sG is None:
        if not dist:
            tree = ss.cKDTree(Y)
            dist_mat, idx_mat = tree.query(Y, k=K + 1)
            N = Y.shape[0]
            distances = dist_mat[:, 1:]
            indices = idx_mat[:, 1:]

            rows = np.zeros(N * K)
            cols = np.zeros(N * K)
            dists = np.zeros(N * K)
            location = 0

            for i in range(N):
                rows[location : location + K] = i
                cols[location : location + K] = indices[i]
                dists[location : location + K] = distances[i]
                location = location + K
            sG = csr_matrix((np.array(dists) ** 2, (rows, cols)), shape=(N, N))
            sG = scipy.sparse.csc_matrix.maximum(sG, sG.T)  # symmetrize the matrix
        else:
            N = Y.shape[0]
            sidx = np.argsort(dist)
            # flatten first rows and then cols
            i = repmat(sidx[:, 0][:, None], K, 1).flatten()  # .reshape(1, -1)[0]
            j = sidx[:, 1 : K + 1].T.flatten()  # .reshape(1, -1)[0]
            sG = csr_matrix(
                (np.repeat(1, N * K), (i, j)), shape=(N, N)
            )  # [1 for k in range(N * K)]

    if not dist:
        if list(set(sG.data)) == [1]:
            print(
                "Error: sG should not just be an adjacency graph and has to include the distance information between vertices!"
            )
            exit()
        else:
            dist = sG

    N, D = Y.shape
    G = sG

    rows, cols, s0 = scipy.sparse.find(scipy.sparse.tril(G))

    # idx_map = np.vstack((rows, cols)).T

    s = np.ones(s0.shape)
    m = len(s)
    #############################################
    objs = np.zeros(maxIter)

    for iter in range(maxIter):
        S = csr_matrix((s, (rows, cols)), shape=(N, N))  # .toarray()
        S = S + S.T
        Q = (
            scipy.sparse.diags(
                functools.reduce(operator.concat, S.sum(axis=1)[:, 0].tolist())
            )
            - S
            + 0.25 * (param_gamma + 1) * scipy.sparse.eye(N, N)
        )  ##################
        R = scipy.linalg.cholesky(
            Q.toarray()
        )  # Cholesky Decomposition of a Sparse Matrix
        invR = scipy.sparse.linalg.inv(csr_matrix(R))  # R:solve(R)
        # invR = np.matrix(invR)
        # invQ = invR*invR.T
        # left = invR.T*np.matrix(Y)
        invQ = invR.dot(invR.T)
        left = invR.T.dot(Y)

        # res = scipy.sparse.linalg.svds(left, k = d)
        res = scipy.linalg.svd(left)
        Lambda = res[1][:d]
        W = res[2][:, :2]
        invQY = invR.dot(left)
        invQYW = invQY.dot(W)

        P = 0.5 * D * invQ + 0.125 * param_gamma ** 2 * invQYW.dot(invQYW.T)
        logdet_Q = 2 * sum(np.log(np.diag(np.linalg.cholesky(Q.toarray()).T)))
        # log(det(Q))
        obj = (
            0.5 * D * logdet_Q
            - scipy.sparse.csr_matrix.sum(scipy.sparse.csr_matrix.multiply(S, dist))
            + 0.25
            / C
            * scipy.sparse.csr_matrix.sum(scipy.sparse.csr_matrix.multiply(S, S))
            - 0.125 * param_gamma ** 2 * sum(np.diag(np.dot(W.T, np.dot(Y.T, invQYW))))
        )  # trace: #sum(diag(m))
        objs[iter] = obj

        if verbose:
            if iter == 0:
                print("i = ", iter + 1, ", obj = ", obj)
            else:
                rel_obj_diff = abs(obj - objs[iter - 1]) / abs(objs[iter - 1])
                print("i = ", iter, ", obj = ", obj, ", rel_obj_diff = ", rel_obj_diff)
        subgrad = np.zeros(m)

        for i in range(len(rows)):
            subgrad[i] = (
                P[rows[i], rows[i]]
                + P[cols[i], cols[i]]
                - P[rows[i], cols[i]]
                - P[cols[i], rows[i]]
                - 1 / C * S[rows[i], cols[i]]
                - 2 * dist[rows[i], cols[i]]
            )

        s = s + 1 / (iter + 1) * subgrad
        s[s < 0] = 0

        # print("print s:",s)
        if param_gamma != 0:
            # print("print invQY:",invQY)
            # print("print W:",W)

            Z = 0.25 * (param_gamma + 1) * np.dot(invQY, W)
        else:
            # centeralized kernel
            A = scipy.sparse.linalg.inv(Q)
            column_sums = np.array(np.sum(A, 0) / N)
            J = np.dot(np.array(np.ones(N))[:, None], np.array(column_sums))
            K = A - J - J.T + sum(column_sums) / N

            # eigendecomposition
            V, U = eigs(K, d)
            v = V[0:d]

            tmp = np.zeros(shape=(d, d))
            np.fill_diagonal(tmp, np.sqrt(v))
            Z = np.dot(U, tmp)

    return (S, Z)


def logdet(A):
    """ Here, A should be a square matrix of double or single class.	
    If A is singular, it will returns -inf.	
    Theoretically, this function should be functionally	
    equivalent to log(det(A)). However, it avoids the	
    overflow/underflow problems that are likely to	
    happen when applying det to large matrices.	
    """
    v = 2 * sum(np.log(np.diag(np.linalg.cholesky(A))))
    return v
