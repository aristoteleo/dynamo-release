# use for convert list of list to a list (https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists)
import functools
import operator
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.spatial as ss
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

from .DDRTree import repmat

# from scikits.sparse.cholmod import cholesky


def psl(
    Y: np.ndarray,
    sG: Optional[csr_matrix] = None,
    dist: Optional[np.ndarray] = None,
    K: int = 10,
    C: int = 1e3,
    param_gamma: float = 1e-3,
    d: int = 2,
    maxIter: int = 10,
    verbose: bool = False
) -> Tuple[csr_matrix, np.ndarray]:
    """This function is a pure Python implementation of the PSL algorithm.

    Reference: 
    Li Wang and Qi Mao, Probabilistic Dimensionality Reduction via Structure Learning. T-PAMI, VOL. 41, NO. 1, JANUARY 2019

    Args:
        Y: The data list.
        sG: A prior kNN graph passed to the algorithm. Defaults to None.
        dist: A dense distance matrix between all vertices. If no distance matrix passed, we will use the kNN based
            algorithm, otherwise we will use the original algorithm reported in the manuscript. Defaults to None.
        K: Number of nearest neighbors used to build the neighborhood graph. Large k can obtain less sparse structures.
            Ignored if sG is used. Defaults to 10.
        C: The penalty parameter for loss term. It controls the preservation of distances. The larger it is, the
            distance is more strictly preserve. If the structure is very clear, a larger C is preferred. Defaults to 
            1e3.
        param_gamma: param_gamma is trying to make a matrix A non-singular, it is like a round-off parameter. 1e-4 or
            1e-5 is good. It corresponds to the variance of prior embedding. Defaults to 1e-3.
        d: the embedding dimension. Defaults to 2.
        maxIter: Number of maximum iterations. Defaults to 10.
        verbose: Whether to print running information. Defaults to False.

    Returns:
        A tuple (S, Z), where S is the adjacency matrix and Z is the reduced low dimension embedding.
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
                rows[location: location + K] = i
                cols[location: location + K] = indices[i]
                dists[location: location + K] = distances[i]
                location = location + K
            sG = csr_matrix((np.array(dists) ** 2, (rows, cols)), shape=(N, N))
            sG = scipy.sparse.csc_matrix.maximum(sG, sG.T)  # symmetrize the matrix
        else:
            N = Y.shape[0]
            sidx = np.argsort(dist)
            # flatten first rows and then cols
            i = repmat(sidx[:, 0][:, None], K, 1).flatten()  # .reshape(1, -1)[0]
            j = sidx[:, 1: K + 1].T.flatten()  # .reshape(1, -1)[0]
            sG = csr_matrix(
                (np.repeat(1, N * K), (i, j)), shape=(N, N)
            )  # [1 for k in range(N * K)]

    if dist is None:
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


def diag_mat(values: List[int]):
    """Returns a diagonal matrix with the given values on the diagonal.

    Args:
        values: A list of values to place on the diagonal of the matrix.

    Returns:
        A diagonal matrix with the given values on the diagonal.
    """

    mat = np.zeros((len(values), len(values)))
    np.fill_diagonal(mat, values)

    return mat
