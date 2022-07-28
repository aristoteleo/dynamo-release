from typing import Callable, Union

import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.neighbors import NearestNeighbors

from ..dynamo_logger import LoggerManager
from .utils import nearest_neighbors, timeit


class TRNET:
    def __init__(self, n_nodes, X, seed=19491001):
        self.n_nodes = n_nodes
        self.n_dims = X.shape[1]
        self.X = X
        self.seed = seed
        self.W = self.draw_sample(self.n_nodes)  # initialize the positions of nodes

    def draw_sample(self, n_samples):
        np.random.seed(self.seed)

        idx = np.random.randint(0, self.X.shape[0], n_samples)
        return self.X[idx]

    def runOnce(self, p, l, ep, c):
        # calc the squared distances ||w - p||^2
        D = p - self.W
        sD = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            sD[i] = D[i].dot(D[i])

        # calc the closeness rank k's
        I = np.argsort(sD)
        K = np.empty_like(I)
        K[I] = np.arange(len(I))

        # move the nodes
        if c == 0:
            self.W += ep * np.exp(-K[:, None] / l) * D
        else:
            # move nodes whose displacements are larger than the cutoff to accelerate the calculation
            kc = -l * np.log(c / ep)
            idx = K < kc
            K = K[:, None]
            self.W[idx, :] += ep * np.exp(-K[idx] / l) * D[idx, :]

    def run(self, tmax=200, li=0.2, lf=0.01, ei=0.3, ef=0.05, c=0):
        tmax = int(tmax * self.n_nodes)
        li = li * self.n_nodes
        P = self.draw_sample(tmax)
        for t in LoggerManager.progress_logger(range(1, tmax + 1), progress_name="Running TRN"):
            # calc the parameters
            tt = t / tmax
            l = li * np.power(lf / li, tt)
            ep = ei * np.power(ef / ei, tt)
            # run once
            self.runOnce(P[t - 1], l, ep, c)

    def run_n_pause(self, k0, k, tmax=200, li=0.2, lf=0.01, ei=0.3, ef=0.05, c=0):
        tmax = int(tmax * self.n_nodes)
        li = li * self.n_nodes
        P = self.draw_sample(tmax)
        for t in range(k0, k + 1):
            # calc the parameters
            tt = t / tmax
            l = li * np.power(lf / li, tt)
            ep = ei * np.power(ef / ei, tt)
            # run once
            self.runOnce(P[t - 1], l, ep, c)

            if t % 1000 == 0:
                print(str(t) + " steps have been run")


@timeit
def trn(X, n, return_index=True, seed=19491001, **kwargs):
    trnet = TRNET(n, X, seed)
    trnet.run(**kwargs)
    if not return_index:
        return trnet.W
    else:
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=1,
                n_jobs=-1,
                random_state=seed,
            )
            idx, _ = nbrs.query(trnet.W, k=1)
        else:
            alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=1, algorithm=alg, n_jobs=-1).fit(X)
            _, idx = nbrs.kneighbors(trnet.W)

        return idx[:, 0]


def sample_by_velocity(V, n, seed=19491001):
    np.random.seed(seed)
    tmp_V = np.linalg.norm(V, axis=1)
    p = tmp_V / np.sum(tmp_V)
    idx = np.random.choice(np.arange(len(V)), size=n, p=p, replace=False)
    return idx


def sample_by_kmeans(X, n, return_index=False):
    C, _ = kmeans2(X, n)
    nbrs = nearest_neighbors(C, X, k=1).flatten()

    if return_index:
        return nbrs
    else:
        X[nbrs]


def lhsclassic(n_samples: int, n_dim: int, bounds=None, seed=19491001):
    """
    Latin Hypercube Sampling method implemented from PyDOE.

    Parameters
    ----------
        n_samples: int
            Number of samples to be generated.
        n_dim: int
            Number of data dimensions.
        bounds: None, list, or :class:`~numpy.ndarray`
            n_dim-by-2 matrix where each row specifies the lower and upper bound for the corresponding dimension.
            If None, it is assumed to be (0, 1) for every dimension.
        seed: int
            Randomization seed.

    Returns
    -------
        H: :class:`~numpy.ndarray`
            The sampled data array (n_samples-by-n_dim).
    """

    # Generate the intervals
    np.random.seed(seed)
    cut = np.linspace(0, 1, n_samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(n_samples, n_dim)
    a = cut[:n_samples]
    b = cut[1 : n_samples + 1]
    rdpoints = np.zeros(u.shape)
    for j in range(n_dim):
        rdpoints[:, j] = u[:, j] * (b - a) + a

    # Make the random pairings
    H = np.zeros(rdpoints.shape)
    for j in range(n_dim):
        order = np.random.permutation(range(n_samples))
        H[:, j] = rdpoints[order, j]

    # Scale according to bounds
    if bounds is not None:
        for i in range(n_dim):
            H[:, i] = H[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    return H


def sample(
    arr: Union[list, np.ndarray],
    n: int,
    method: str = "random",
    X: Union[np.ndarray, None] = None,
    V: Union[np.ndarray, None] = None,
    seed: int = 19491001,
    **kwargs,
):
    """
    A collection of various sampling methods.

    Parameters
    ----------
        arr: list or :class:`~numpy.ndarray`
            The array to be subsampled.
        n: int
            The number of samples.
        method: str
            Sampling method:
            "random": randomly choosing `n` elements from `arr`;
            "velocity": Higher the velocity, higher the chance to be sampled;
            "trn": Topology Representing Network based sampling;
            "kmeans": `n` points that are closest to the kmeans centroids on `X` are chosen.
        X: None or :class:`~numpy.ndarray`
            Coordinates associated to each element in `arr`
        V: None or :class:`~numpy.ndarray`
            Velocity associated to each element in `arr`
        seed: int
            randomization seed

    Returns
    -------
        sub_arr: :class:`~numpy.ndarray`
            The sampled data array.
    """
    if method == "random":
        np.random.seed(seed)
        sub_arr = np.random.choice(arr, size=n, replace=False)
    elif method == "velocity" and V is not None:
        sub_arr = arr[sample_by_velocity(V=V, n=n, seed=seed, **kwargs)]
    elif method == "trn" and X is not None:
        sub_arr = arr[trn(X=X, n=n, return_index=True, seed=seed, **kwargs)]
    elif method == "kmeans":
        sub_arr = arr[sample_by_kmeans(X, n, return_index=True)]
    else:
        raise NotImplementedError(f"The sampling method {method} is not implemented or relevant data are not provided.")
    return sub_arr
