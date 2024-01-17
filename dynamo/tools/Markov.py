# create by Yan Zhang, minor adjusted by Xiaojie Qiu
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numba import jit
from scipy.linalg import eig, null_space
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ..dynamo_logger import LoggerManager, main_warning
from ..simulation.utils import directMethod
from .connectivity import k_nearest_neighbors
from .utils import append_iterative_neighbor_indices, flatten


def prepare_velocity_grid_data(
    X_emb: np.ndarray,
    xy_grid_nums: List,
    density: Optional[int] = None,
    smooth: Optional[float] = None,
    n_neighbors: Optional[int] = None,
) -> Tuple:
    """Prepare the grid of data used to calculate the velocity embedding on grid.

    Args:
        X_emb: The embedded data matrix.
        xy_grid_nums: The number of grid points along each dimension for the velocity grid.
        density: The density of grid points relative to the number of points in each dimension.
        smooth: The smoothing factor for grid points relative to the range of each dimension.
        n_neighbors: The number of neighbors to consider when estimating grid velocities.

    Returns:
        A tuple containing:
            The grid points for the velocity.
            The estimated probability mass for each grid point based on grid velocities.
            The indices of neighbors for each grid point.
            The weights corresponding to the neighbors for each grid point.
    """
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs, scale = [], 0
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, xy_grid_nums[dim_i] * density)
        scale += gr[1] - gr[0]
        grs.append(gr)

    scale = scale / n_dim * smooth

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = np.max([10, int(n_obs / 50)])

    neighs, dists = k_nearest_neighbors(
        X_emb,
        query_X=X_grid,
        k=n_neighbors - 1,
        exclude_self=False,
        pynn_rand_state=19491001,
    )

    weight = norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    return X_grid, p_mass, neighs, weight


def grid_velocity_filter(
    V_emb: np.ndarray,
    neighs: np.ndarray,
    p_mass: np.ndarray,
    X_grid: np.ndarray,
    V_grid: np.ndarray,
    min_mass: Optional[float] = None,
    autoscale: bool = False,
    adjust_for_stream: bool = True,
    V_threshold: Optional[float]=None,
) -> Tuple:
    """Filter the grid velocities, adjusting for streamlines if needed.

    Args:
        V_emb: The velocity matrix which represents the velocity vectors associated with each data point in the embedding.
        neighs: The indices of neighbors for each grid point.
        p_mass: The estimated probability mass for each grid point based on grid velocities.
        X_grid: The grid points for the velocity.
        V_grid: The estimated grid velocities.
        min_mass: The minimum probability mass threshold to filter grid points based on p_mass.
        autoscale: Whether to autoscale the grid velocities based on the grid points and their velocities.
        adjust_for_stream: Whether to adjust the grid velocities to show streamlines.
        V_threshold: The velocity threshold to filter grid points based on velocity magnitude.

    Returns:
        The filtered grid points and the filtered estimated grid velocities.
    """
    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(V_grid.shape[0]))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid**2).sum(0))
        if V_threshold is not None:
            V_grid[0][mass.reshape(V_grid[0].shape) < V_threshold] = np.nan
        else:
            if min_mass is None:
                min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
            cutoff = mass.reshape(V_grid[0].shape) < min_mass

            if neighs is not None:
                length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T.reshape(ns, ns)
                cutoff |= length < np.percentile(length, 5)

            V_grid[0][cutoff] = np.nan
    else:
        from ..plot.utils import quiver_autoscaler

        if p_mass is None:
            p_mass = np.sqrt((V_grid**2).sum(1))
            if min_mass is None:
                min_mass = np.clip(np.percentile(p_mass, 5), 1e-5, None)
        else:
            if min_mass is None:
                min_mass = np.clip(np.percentile(p_mass, 99) / 100, 1e-5, None)
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale:
            V_grid /= 3 * quiver_autoscaler(X_grid, V_grid)

    return X_grid, V_grid


def velocity_on_grid(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    xy_grid_nums: List,
    density: Optional[int] = None,
    smooth: Optional[float] = None,
    n_neighbors: Optional[int] = None,
    min_mass: Optional[float] = None,
    autoscale: bool = False,
    adjust_for_stream: bool = True,
    V_threshold: Optional[float] = None,
    cut_off_velocity: bool = True,
) -> Tuple:
    """Function to calculate the velocity vectors on a grid for grid vector field quiver plot and streamplot, adapted
    from scVelo.

    Args:
        X_emb: The low-dimensional embedding which represents the coordinates of the data points in the embedding space.
        V_emb: The velocity matrix which represents the velocity vectors associated with each data point in the embedding.
        xy_grid_nums: The number of grid points along each dimension of the embedding space.
        density: The number of density grid points for each dimension.
        smooth: The smoothing parameter for grid points along each dimension.
        n_neighbors: The number of neighbors to consider for estimating grid velocities.
        min_mass: The minimum probability mass threshold to filter grid points based on p_mass.
        autoscale: Whether to autoscale the grid velocities based on the grid points and their velocities.
        adjust_for_stream: Whether to adjust the grid velocities to show streamlines.
        V_threshold: The velocity threshold to filter grid points based on velocity magnitude.
        cut_off_velocity: Whether to cut off the grid velocities or return the entire grid.

    Returns:
        A tuple containing the grid points, the filtered estimated grid velocities, the diffusion matrix D of shape.
    """
    from ..vectorfield.stochastic_process import diffusionMatrix2D

    valid_idx = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb, V_emb = X_emb[valid_idx], V_emb[valid_idx]

    X_grid, p_mass, neighs, weight = prepare_velocity_grid_data(
        X_emb,
        xy_grid_nums,
        density=density,
        smooth=smooth,
        n_neighbors=n_neighbors,
    )

    # V_grid = (V_emb[neighs] * (weight / p_mass[:, None])[:, :, None]).sum(1) # / np.maximum(1, p_mass)[:, None]
    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1) / np.maximum(1, p_mass)[:, None]
    # calculate diffusion matrix D
    D = diffusionMatrix2D(V_emb[neighs])

    if cut_off_velocity:
        X_grid, V_grid = grid_velocity_filter(
            V_emb,
            neighs,
            p_mass,
            X_grid,
            V_grid,
            min_mass=min_mass,
            autoscale=autoscale,
            adjust_for_stream=adjust_for_stream,
            V_threshold=V_threshold,
        )
    else:
        X_grid, V_grid = (
            np.array([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])]),
            np.array(
                [
                    V_grid[:, 0].reshape(xy_grid_nums),
                    V_grid[:, 1].reshape(xy_grid_nums),
                ]
            ),
        )

    return X_grid, V_grid, D


class MarkovChain:
    """Base class for all Markov Chain implementation."""
    def __init__(
        self,
        P: Optional[np.ndarray] = None,
        eignum: Optional[int] = None,
        check_norm: bool = True,
        sumto: int = 1,
        tol: float = 1e-3
    ):
        """Initialize the MarkovChain instance.

        Args:
            P: The transition matrix. The elements in the transition matrix P_ij encodes transition probability from j
                to i, i.e.:
                    P_ij = P(j -> i)
                Consequently, each column of P should sum to `sumto`.
            eignum: Number of eigenvalues/eigenvectors to compute. If None, all are solved.
            check_norm:  Whether to check if the input transition matrix is properly normalized.
            sumto: The value that each column of the transition matrix should sum to if 'check_norm' is True.
            tol: The numerical tolerance used for normalization check.

        Returns:
            An instance of MarkovChain.
        """
        if check_norm and not self.is_normalized(P, axis=0, sumto=sumto, tol=tol):
            if self.is_normalized(P, axis=1, sumto=sumto, tol=tol):
                LoggerManager.main_logger.info(
                    f"Column sums of the input matrix are not {sumto} but row sums are. "
                    "Transposing the transition matrix"
                )
                P = P.T
            else:
                raise Exception(f"Neither the row nor the column sums of the input matrix are {sumto}.")
        self.P = P
        self.D = None  # eigenvalues
        self.U = None  # left eigenvectors
        self.W = None  # right eigenvectors
        self.W_inv = None
        self.eignum = eignum  # if None all eigs are solved

    def eigsys(self, eignum: Optional[int] = None) -> None:
        """Compute the eigenvalues and eigenvectors of the transition matrix.

        The eigenvalues and eigenvectors are stored in the class variables D, U, and W. If 'eignum' is None, the
        function uses numpy's eig function to compute all eigenvalues and eigenvectors. Otherwise, it uses
        sparse.linalg.eigs to compute the first 'eignum' eigenvalues and eigenvectors. The eigenvalues are sorted in
        descending order, and the eigenvectors are arranged accordingly.

        Args:
            eignum: Number of eigenvalues/eigenvectors to compute.
        """
        if eignum is not None:
            self.eignum = eignum
        if self.eignum is None:
            D, U, W = eig(self.P, left=True, right=True)
            idx = D.argsort()[::-1]
            self.D = D[idx]
            self.U = U[:, idx]
            self.W = W[:, idx]
            try:
                self.W_inv = np.linalg.inv(self.W)
            except np.linalg.LinAlgError:
                self.W_inv = np.linalg.pinv(self.W)
        else:
            self.D, self.W = sp.linalg.eigs(self.P, k=self.eignum, which="LR")
            self.U = self.right_eigvecs_to_left(self.W, self.W[:, 0])
            self.W_inv = self.U.T.copy()
            self.U /= np.linalg.norm(self.U, axis=0)

    def right_eigvecs_to_left(self, W: np.ndarray, p_st: np.ndarray) -> np.ndarray:
        """Transform right eigenvectors into left eigenvectors.

        Args:
            W: The matrix containing right eigenvectors.
            p_st: A probability distribution vector.

        Returns:
            The matrix containing left eigenvectors.
        """
        U = np.diag(1 / np.abs(p_st)) @ W
        f = np.mean(np.diag(U.T @ U))
        return U / np.sqrt(f)

    def get_num_states(self) -> float:
        """Get the number of states in the Markov chain.

        Returns:
            The number of states in the Markov chain.
        """
        return self.P.shape[0]

    def make_p0(self, init_states: np.ndarray) -> np.ndarray:
        """Create an initial probability distribution vector with probabilities set to 1 at specified initial states.

        Args:
            init_states: A list or array of initial states.

        Returns:
            The initial probability distribution vector.
        """
        p0 = np.zeros(self.get_num_states())
        p0[init_states] = 1
        p0 /= np.sum(p0)
        return p0

    def is_normalized(
        self,
        P: Optional[np.ndarray] = None,
        tol: float = 1e-3,
        sumto: int = 1,
        axis: int = 0,
        ignore_nan: bool = True
    ) -> bool:
        """check if the matrix is properly normalized up to `tol`.

        Args:
            P: The transition matrix. If None, self.P is checked instead.
            tol: The numerical tolerance.
            sumto: The value that each column/row should sum to.
            axis: 0 - Check if the matrix is column normalized; 1 - check if the matrix is row normalized.
            ignore_nan: Whether to ignore NaN values when computing the sum.

        Returns:
            True if the matrix is properly normalized.
        """
        if P is None:
            main_warning("No transition matrix input. Normalization check is skipped.")
            return True
        else:
            sumfunc = np.sum if not ignore_nan else np.nansum
            return np.all(np.abs(sumfunc(P, axis=axis) - sumto) < tol)

    def __reset__(self) -> None:
        """Reset the class variables D, U, W, and W_inv to None."""
        self.D = None
        self.U = None
        self.W = None
        self.W_inv = None


class KernelMarkovChain(MarkovChain):
    """KernelMarkovChain class represents a Markov chain with kernel-based transition probabilities."""
    def __init__(
        self,
        P: Optional[np.ndarray] = None,
        Idx: Optional[np.ndarray] = None,
        n_recurse_neighbors: Optional[int] = None
    ):
        """Initialize the KernelMarkovChain instance.

        Args:
            P: The transition matrix of the Markov chain.
            Idx: The neighbor indices used for kernel computation.
            n_recurse_neighbors: Number of recursive neighbor searches to improve kernel computation. If not None, it
                appends the iterative neighbor indices using the function append_iterative_neighbor_indices().

        Returns:
            An instance of KernelMarkovChain.
        """

        super().__init__(P)
        self.Kd = None
        if n_recurse_neighbors is not None and Idx is not None:
            self.Idx = append_iterative_neighbor_indices(Idx, n_recurse_neighbors)
        else:
            self.Idx = Idx

    def fit(
        self,
        X: np.ndarray,
        V: np.ndarray,
        M_diff: Union[np.ndarray, float],
        neighbor_idx: Optional[np.ndarray] = None,
        n_recurse_neighbors: Optional[int] = None,
        k: int = 30,
        epsilon: Optional[float] = None,
        adaptive_local_kernel: bool = False,
        tol: float = 1e-4,
        sparse_construct: bool = True,
        sample_fraction: float = None,
    ):
        """Fit the KernelMarkovChain model to the given data.

        Args:
            X: The cell data matrix which represents the states of the Markov chain.
            V: The velocity matrix which represents the expected returns of each state in the Markov chain.
            M_diff: The covariance matrix or scalar value representing the diffusion matrix. It is used for
                computing transition probabilities.
            neighbor_idx: The neighbor indices used for kernel computation. If None, it is computed using k-NN.
            n_recurse_neighbors: Number of recursive neighbor searches to improve kernel computation. If not None, it
                appends the iterative neighbor indices using the function append_iterative_neighbor_indices().
            k: The number of nearest neighbors used for k-NN down sampling.
            epsilon: The bandwidth parameter used for computing density kernel if not None.
            adaptive_local_kernel: Whether to use adaptive local kernel computation for transition probabilities.
            tol: The numerical tolerance used for transition probability computation. Default is 1e-4.
            sparse_construct: Whether construct sparse matrices for transition matrix and density kernel.
            sample_fraction: The fraction of neighbors used for k-NN down sampling if not None.
        """
        # compute connectivity
        if neighbor_idx is None:
            neighbor_idx, _ = k_nearest_neighbors(
                X,
                k=k-1,
                exclude_self=False,
                pynn_rand_state=19491001,
            )

        if n_recurse_neighbors is not None:
            self.Idx = append_iterative_neighbor_indices(neighbor_idx, n_recurse_neighbors)
        else:
            self.Idx = neighbor_idx

        # apply kNN downsampling to accelerate calculation (adapted from velocyto)
        if sample_fraction is not None:
            neighbor_idx = self.Idx
            p = np.linspace(0.5, 1, neighbor_idx.shape[1])
            p = p / p.sum()

            sampling_ixs = np.stack(
                (
                    np.random.choice(
                        np.arange(1, neighbor_idx.shape[1] - 1),
                        size=int(sample_fraction * (neighbor_idx.shape[1] + 1)),
                        replace=False,
                        p=p,
                    )
                    for i in range(neighbor_idx.shape[0])
                ),
                0,
            )
            self.Idx = self.Idx[np.arange(neighbor_idx.shape[0])[:, None], sampling_ixs]

        n = X.shape[0]
        if sparse_construct:
            self.P = sp.lil_matrix((n, n))
        else:
            self.P = np.zeros((n, n))

        # compute density kernel
        if epsilon is not None:
            if sparse_construct:
                self.Kd = sp.lil_matrix((n, n))
            else:
                self.Kd = np.zeros((n, n))
            inv_eps = 1 / epsilon
            for i in range(n):
                self.Kd[i, self.Idx[i]] = compute_density_kernel(X[i], X[self.Idx[i]], inv_eps)
            self.Kd = sp.csc_matrix(self.Kd)
            D = np.sum(self.Kd, 0)

        # compute transition prob.
        if np.isscalar(M_diff):
            inv_s = 1 / M_diff
        else:
            inv_s = np.linalg.inv(M_diff)
        for i in tqdm(range(n), desc="compute transiton matrix"):
            y = X[i]
            v = V[i]
            Y = X[self.Idx[i]]
            if adaptive_local_kernel:
                k = compute_drift_local_kernel(y, v, Y, inv_s)
            else:
                k = compute_drift_kernel(y, v, Y, inv_s)
            if epsilon is not None:
                k = k / D[0, self.Idx[i]]
            else:
                k = np.matrix(k)
            p = k / np.sum(k) if np.sum(k) > 0 else np.ones_like(k) / n
            p[p <= tol] = 0  # tolerance check
            p = p / np.sum(p)
            self.P[self.Idx[i], i] = p.A[0]

        self.P = sp.csc_matrix(self.P)

    def propagate_P(self, num_prop: int) -> sp.csc_matrix:
        """Propagate the transition matrix 'P' for a given number of steps.

        Args:
            num_prop: The number of propagation steps.

        Returns:
            The propagated transition matrix after 'num_prop' steps.
        """
        ret = sp.csc_matrix(self.P, copy=True)
        for i in range(num_prop - 1):
            ret = self.P * ret  # sparse matrix (ret) is a `np.matrix`
        return ret

    def compute_drift(self, X: np.ndarray, num_prop: int = 1, scale: bool = True) -> np.ndarray:
        """Compute the drift for each state in the Markov chain.

        Args:
            X: The cell data matrix which represents the states of the Markov chain.
            num_prop: The number of propagation steps used for drift computation. Default is 1.
            scale: Whether to scale the result.

        Returns:
            The computed drift values for each state in the Markov chain.
        """
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(int(num_prop))
        for i in tqdm(range(n), desc="compute drift"):
            V[i] = (X - X[i]).T.dot(P[:, i].A.flatten())
        return V * 1 / V.max() if scale else V

    def compute_density_corrected_drift(
        self,
        X: np.ndarray,
        neighbor_idx: Optional[np.ndarray] = None,
        k: Optional[int] = None,
        num_prop: int = 1,
        normalize_vector: bool = False,
        correct_by_mean: bool = True,
        scale: bool = True,
    ) -> np.ndarray:
        """Compute density-corrected drift for each state in the Markov chain.

        Args:
            X: The cell data matrix which represents the states of the Markov chain.
            neighbor_idx: The neighbor indices used for density-corrected drift computation.
            k: The number of nearest neighbors used for computing the mean of kernel probabilities.
            num_prop: The number of propagation steps used for drift computation. Default is 1.
            normalize_vector: Whether to normalize the drift vector for each state.
            correct_by_mean: Whether to correct the drift by subtracting the mean kernel probability from each state's drift.
            scale: Whether to scale the result.

        Returns:
            The computed density-corrected drift values for each state in the Markov chain.
        """
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(num_prop)
        if neighbor_idx is None:
            neighbor_idx = self.Idx
        for i in tqdm(range(n), desc="calculating density corrected drift"):
            Idx = neighbor_idx[i]
            D = X[Idx] - X[i]
            if normalize_vector:
                D = D / np.linalg.norm(D, 1)
            p = P[Idx, i].A.flatten()
            if k is None:
                if not correct_by_mean:
                    k_inv = 1 / len(Idx)
                else:
                    k_inv = np.mean(p)
            else:
                k_inv = 1 / k
            p -= k_inv
            V[i] = D.T.dot(p)
        return V * 1 / V.max() if scale else V

    def compute_stationary_distribution(self) -> np.ndarray:
        """Compute the stationary distribution of the Markov chain.

        Returns:
            The computed stationary distribution as a probability vector.
        """
        # if self.W is None:
        # self.eigsys()
        _, vecs = sp.linalg.eigs(self.P, k=1, which="LR")
        p = np.abs(np.real(vecs[:, 0]))
        p = p / np.sum(p)
        return p

    def diffusion_map_embedding(self, n_dims: int = 2, t: int = 1) -> np.ndarray:
        """Perform diffusion map embedding for the Markov chain.

        Args:
            n_dims: The number of dimensions for the diffusion map embedding.
            t: The diffusion time parameter used in the embedding.

        Returns:
            The diffusion map embedding of the Markov chain as a matrix.
        """
        # if self.W is None:
        #    self.eigsys()
        vals, vecs = sp.linalg.eigs(self.P.T, k=n_dims + 1, which="LR")
        ind = np.range(1, n_dims + 1)
        Y = np.real(vals[ind] ** t) * np.real(vecs[:, ind])
        return Y

    def compute_theta(self, p_st: Optional[np.ndarray] = None) -> sp.csr_matrix:
        """Compute the matrix 'Theta' for the Markov chain.

        Args:
            p_st: The stationary distribution of the Markov chain.

        Returns:
            The computed matrix 'Theta' as a sparse matrix.
        """
        p_st = self.compute_stationary_distribution() if p_st is None else p_st
        Pi = sp.csr_matrix(np.diag(np.sqrt(p_st)))
        Pi_right = sp.csc_matrix(np.diag(np.sqrt(p_st)))
        # Pi_inv = sp.csr_matrix(np.linalg.pinv(Pi))
        # Pi_inv_right = sp.csc_matrix(np.linalg.pinv(Pi))
        Pi_inv = sp.csr_matrix(np.diag(np.sqrt(1 / p_st)))
        Pi_inv_right = sp.csc_matrix(np.diag(np.sqrt(1 / p_st)))
        Theta = 0.5 * (Pi @ self.P @ Pi_inv_right + Pi_inv @ self.P.T @ Pi_right)

        return Theta


class DiscreteTimeMarkovChain(MarkovChain):
    """DiscreteTimeMarkovChain class represents a discrete-time Markov chain."""
    def __init__(self, P: Optional[np.ndarray] = None, eignum: Optional[int] = None, sumto: int = 1, **kwargs):
        """Initialize the DiscreteTimeMarkovChain instance.

        Args:
            P: The transition matrix of the Markov chain.
            eignum: Number of eigenvalues/eigenvectors to compute.
            sumto: The value that each column of the transition matrix should sum to.
            **kwargs: Additional keyword arguments to be passed to the base class MarkovChain's constructor.

        Returns:
            An instance of DiscreteTimeMarkovChain.
        """
        super().__init__(P, eignum=eignum, sumto=sumto, **kwargs)
        # self.Kd = None

    """def fit(self, X, V, k, s=None, method="qp", eps=None, tol=1e-4):  # pass index
        # the parameter k will be replaced by a connectivity matrix in the future.
        self.__reset__()
        # knn clustering
        if X.shape[0] > 200000 and X.shape[1] > 2:
            from pynndescent import NNDescent

            nbrs = NNDescent(
                X,
                metric="euclidean",
                n_neighbors=k,
                n_jobs=-1,
                random_state=19491001,
            )
            Idx, _ = nbrs.query(X, k=k)
        else:
            alg = "ball_tree" if X.shape[1] > 10 else "kd_tree"
            nbrs = NearestNeighbors(n_neighbors=k, algorithm=alg, n_jobs=-1).fit(X)
            _, Idx = nbrs.kneighbors(X)
        # compute transition prob.
        n = X.shape[0]
        self.P = np.zeros((n, n))
        if method == "kernel":
            inv_s = np.linalg.inv(s)
            # compute density kernel
            if eps is not None:
                self.Kd = np.zeros((n, n))
                inv_eps = 1 / eps
                for i in range(n):
                    self.Kd[i, Idx[i]] = compute_density_kernel(X[i], X[Idx[i]], inv_eps)
                D = np.sum(self.Kd, 0)
        for i in range(n):
            y = X[i]
            v = V[i]
            if method == "qp":
                Y = X[Idx[i, 1:]]
                p = compute_markov_trans_prob(y, v, Y, s)
                p[p <= tol] = 0  # tolerance check
                self.P[Idx[i, 1:], i] = p
                self.P[i, i] = 1 - np.sum(p)
            else:
                Y = X[Idx[i]]
                # p = compute_kernel_trans_prob(y, v, Y, inv_s)
                k = compute_drift_kernel(y, v, Y, inv_s)
                if eps is not None:
                    k /= D[Idx[i]]
                p = k / np.sum(k)
                p[p <= tol] = 0  # tolerance check
                p = p / np.sum(p)
                self.P[Idx[i], i] = p"""

    def propagate_P(self, num_prop: int) -> np.ndarray:
        """Propagate the transition matrix 'P' for a given number of steps.

        Args:
            num_prop: The number of propagation steps.

        Returns:
            The propagated transition matrix after 'num_prop' steps.
        """
        ret = np.array(self.P, copy=True)
        for _ in range(num_prop - 1):
            ret = self.P @ ret
        return ret

    def compute_drift(self, X: np.ndarray, num_prop: int = 1) -> np.ndarray:
        """Compute the drift for each state in the Markov chain.

        Args:
            X: The data matrix which represents the states of the Markov chain.
            num_prop: The number of propagation steps used for drift computation. Default is 1.

        Returns:
            The computed drift values for each state in the Markov chain.
        """
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.propagate_P(num_prop)
        for i in range(n):
            V[i] = (X - X[i]).T.dot(P[:, i])
        return V

    def compute_density_corrected_drift(
        self, X: np.ndarray, k: Optional[int] = None, normalize_vector: bool = False
    ) -> np.ndarray:
        """Compute density-corrected drift for each state in the Markov chain.

        Args:
            X: The data matrix whicj represents the states of the Markov chain.
            k: The number of nearest neighbors used for computing the mean of kernel probabilities.
            normalize_vector: whether to normalize the drift vector for each state.

        Returns:
            The computed density-corrected drift values for each state in the Markov chain.
        """
        n = self.get_num_states()
        if k is None:
            k = n
        V = np.zeros_like(X)
        for i in range(n):
            d = X - X[i]  # no self.nbrs_idx[i] is here.... may be wrong?
            if normalize_vector:
                d /= np.linalg.norm(d)
            V[i] = d.T.dot(self.P[:, i] - 1 / k)
        return V

    def solve_distribution(self, p0: np.ndarray, n: int, method: str = "naive") -> np.ndarray:
        """Solve the distribution for the Markov chain.

        Args:
            p0: The initial probability distribution vector.
            n: The number of steps for distribution propagation.
            method: The method used for solving the distribution.

        Returns:
            The computed probability distribution vector after 'n' steps.
        """
        if method == "naive":
            p = p0
            for _ in range(n):
                p = self.P.dot(p)
        else:
            if self.D is None:
                self.eigsys()
            p = np.real(self.W @ np.diag(self.D**n) @ np.linalg.inv(self.W)).dot(p0)
        return p

    def compute_stationary_distribution(self, method: str = "eig") -> np.ndarray:
        """Compute the stationary distribution of the Markov chain.

        Args:
            method: The method used for computing the stationary distribution.

        Returns:
            The computed stationary distribution as a probability vector.
        """
        if method == "solve":
            p = np.real(null_space(self.P - np.eye(self.P.shape[0])[:, 0])[:, 0].flatten())
        else:
            if self.W is None:
                self.eigsys()
            p = np.abs(np.real(self.W[:, 0]))
        p = p / np.sum(p)
        return p

    def lump(self, labels: np.ndarray, M_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Markov chain lumping based on:
            K. Hoffmanna and P. Salamon, Bounding the lumping error in Markov chain dynamics, Appl Math Lett, (2009)

        Args:
            labels: The lumping labels.
            M_weight: The weighting matrix. If None, it is computed using the stationary distribution.

        Returns:
            The lumped transition matrix after the lumping operation.
        """
        k = len(labels)
        M_part = np.zeros((k, self.get_num_states()))

        for i in range(len(labels)):
            M_part[labels[i], i] = 1

        n_node = self.get_num_states()
        if M_weight is None:
            p_st = self.compute_stationary_distribution()
            M_weight = np.multiply(M_part, p_st)
            M_weight = np.divide(M_weight.T, M_weight @ np.ones(n_node))
        P_lumped = M_part @ self.P @ M_weight

        return P_lumped

    def naive_lump(self, x: np.ndarray, grp: np.ndarray) -> np.ndarray:
        """Perform naive Markov chain lumping based on given data and group labels.

        Args:
            x: The data matrix.
            grp: The group labels.

        Returns:
            The lumped transition matrix after the lumping operation.
        """
        k = len(np.unique(grp))
        y = np.zeros((k, k))
        for i in range(len(y)):
            for j in range(len(y)):
                y[i, j] = x[grp == i, :][:, grp == j].mean()

        return y

    def diffusion_map_embedding(self, n_dims: int = 2, t: int = 1) -> np.ndarray:
        """Perform diffusion map embedding for the Markov chain.

        Args:
            n_dims: The number of dimensions for the diffusion map embedding.
            t: The diffusion time parameter used in the embedding.

        Returns:
            The diffusion map embedding of the Markov chain as a matrix of shape (n_states, n_dims).
        """
        if self.W is None:
            self.eigsys()

        ind = np.arange(1, n_dims + 1)
        Y = np.real(self.D[ind] ** t) * np.real(self.U[:, ind])
        return Y

    def simulate_random_walk(self, init_idx: int, num_steps: int) -> np.ndarray:
        """Simulate a random walk on the Markov chain from a given initial state.

        Args:
            init_idx: The index of the initial state for the random walk.
            num_steps: The number of steps for the random walk.

        Returns:
            The sequence of state indices resulting from the random walk.
        """
        P = self.P.copy()

        seq = np.ones(num_steps + 1, dtype=int) * -1
        seq[0] = init_idx
        for i in range(1, num_steps + 1):
            cur_state = seq[i - 1]
            r = np.random.rand()
            seq[i] = np.cumsum(P[:, cur_state]).searchsorted(r)

        return seq


class ContinuousTimeMarkovChain(MarkovChain):
    """ContinuousTimeMarkovChain class represents a continuous-time Markov chain."""
    def __init__(self, P: Optional[np.ndarray] = None, eignum: Optional[int] = None, **kwargs):
        """Initialize the ContinuousTimeMarkovChain instance.

        Args:
            P: The transition matrix of the Markov chain.
            eignum: Number of eigenvalues/eigenvectors to compute.
            **kwargs: Additional keyword arguments to be passed to the base class MarkovChain's constructor.

        Returns:
            An instance of ContinuousTimeMarkovChain.
        """
        super().__init__(P, eignum=eignum, sumto=0, **kwargs)
        self.Q = None  # embedded markov chain transition matrix
        self.Kd = None  # density kernel for density adjustment
        self.p_st = None  # stationary distribution

    def check_transition_rate_matrix(self, P: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Check if the input matrix is a valid transition rate matrix.

        Args:
            P: The transition rate matrix to be checked.
            tol: Tolerance threshold for zero row- or column-sum check.

        Returns:
            The checked transition rate matrix.

        Raises:
            ValueError: If the input transition rate matrix has non-zero row- and column-sums.
        """
        if np.any(flatten(np.abs(np.sum(P, 0))) <= tol):
            return P
        elif np.any(flatten(np.abs(np.sum(P, 1))) <= tol):
            return P.T
        else:
            raise ValueError("The input transition rate matrix must have a zero row- or column-sum.")

    def compute_drift(self, X: np.ndarray, t: float, n_top: int = 5, normalize_vector: bool = False) -> np.ndarray:
        """Compute the drift for each state in the continuous-time Markov chain.

        Args:
            X: The data matrix of shape which represents the states of the Markov chain.
            t: The time at which the drift is computed.
            n_top: The number of top states to consider for drift computation.
            normalize_vector: Whether to normalize the drift vector for each state.

        Returns:
            The computed drift values for each state in the continuous-time Markov chain.
        """
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.compute_transition_matrix(t)

        for i in range(n):
            if n_top is None:
                d = (X - X[i]).T
                if normalize_vector:
                    d = d / np.linalg.norm(d, axis=0)
                V[i] = d.dot(P[:, i])
            else:
                idx = np.argsort(P[:, i])[-n_top:]
                d = (X[idx] - X[i]).T
                if normalize_vector:
                    d = d / np.linalg.norm(d, axis=0)
                V[i] = d.dot(P[idx, i])
                # q = P[idx, i] / np.sum(P[idx, i])
                # V[i] = d.dot(q)
        return V

    def compute_density_corrected_drift(
        self, X: np.ndarray, t: float, k: Optional[int] = None, normalize_vector: bool = False
    ) -> np.ndarray:
        """Compute density-corrected drift for each state in the continuous-time Markov chain.

        Args:
            X: The data matrix of shape which represents the states of the Markov chain.
            t: The time at which the density-corrected drift is computed.
            k: The number of nearest neighbors used for computing the correction term.
            normalize_vector: Whether to normalize the drift vector for each state.

        Returns:
            The computed density-corrected drift values for each state in the continuous-time Markov chain.
        """
        n = self.get_num_states()
        V = np.zeros_like(X)
        P = self.compute_transition_matrix(t)
        for i in range(n):
            P[i, i] = 0
            P[:, i] /= np.sum(P[:, i])
            d = X - X[i]
            if normalize_vector:
                d /= np.linalg.norm(d)
            correction = 1 / k if k is not None else np.mean(P[:, i])
            V[i] = d.T.dot(P[:, i] - correction)
        return V

    def compute_transition_matrix(self, t: float) -> np.ndarray:
        """Compute the transition matrix for a given time.

        Args:
            t: The time at which the transition matrix is computed.

        Returns:
            The computed transition matrix.
        """

        if self.D is None:
            self.eigsys()
        P = np.real(self.W @ np.diag(np.exp(self.D * t)) @ self.W_inv)
        return P

    def compute_embedded_transition_matrix(self) -> np.ndarray:
        """Compute the embedded Markov chain transition matrix.

        Returns:
            The computed embedded Markov chain transition matrix.
        """
        self.Q = np.array(self.P, copy=True)
        for i in range(self.Q.shape[1]):
            self.Q[i, i] = 0
            self.Q[:, i] /= np.sum(self.Q[:, i])
        return self.Q

    def solve_distribution(self, p0: np.ndarray, t: float) -> np.ndarray:
        """Solve the distribution for the continuous-time Markov chain at a given time.

        Args:
            p0: The initial probability distribution vector.
            t: The time at which the distribution is solved.

        Returns:
            The computed probability distribution vector at time.
        """
        P = self.compute_transition_matrix(t)
        p = P @ p0
        p = p / np.sum(p)
        return p

    def compute_stationary_distribution(self, method: str = "eig"):
        """Compute the stationary distribution of the continuous-time Markov chain.

        Args:
            method: The method used for computing the stationary distribution.

        Returns:
            The computed stationary distribution of the continuous-time Markov chain.
        """
        if self.p_st is None:
            if method == "null":
                p = np.abs(np.real(null_space(self.P)[:, 0].flatten()))
                p = p / np.sum(p)
                self.p_st = p
            else:
                if self.D is None:
                    self.eigsys()
                p = np.abs(np.real(self.W[:, 0]))
                p = p / np.sum(p)
                self.p_st = p
        return self.p_st

    def simulate_random_walk(self, init_idx: int, tspan: np.ndarray) -> Tuple:
        """Simulate a random walk on the continuous-time Markov chain from a given initial state.

        Args:
            init_idx: The index of the initial state for the random walk.
            tspan: The time span for the random walk as a 1D array.

        Returns:
            A tuple containing two arrays:
                The value at each time point.
                The corresponding time points during the random walk.
        """
        P = self.P.copy()

        def prop_func(c):
            a = P[:, c[0]]
            a[c[0]] = 0
            return a

        def update_func(c, mu):
            return np.array([mu])

        T, C = directMethod(prop_func, update_func, tspan=tspan, C0=np.array([init_idx]))

        return C, T

    def compute_mean_exit_time(self, p0: np.ndarray, sinks: np.ndarray) -> float:
        """Compute the mean exit time given a initial distribution p0 and a set of sink nodes using:
            met = inv(K) @ p0_
        where K is the transition rate matrix (P) where the columns and rows corresponding to the sinks are removed,
        and p0_ the initial distribution w/o the sinks.

        Args:
            p0: The initial probability distribution vector.
            sinks: The indices of the sink nodes.

        Returns:
            The computed mean exit time.
        """
        states = []
        for i in range(self.get_num_states()):
            if not i in sinks:
                states.append(i)
        K = self.P[states][:, states]  # submatrix of P excluding the sinks
        p0_ = p0[states]
        met = np.sum(-np.linalg.inv(K) @ p0_)
        return met

    def compute_mean_first_passage_time(self, p0: np.ndarray, target: int, sinks: np.ndarray) -> float:
        """Compute the mean first passage time given an initial distribution, a target node, and a set of sink nodes.

        Args:
            p0: The initial probability distribution vector of shape (n_states,).
            target: The index of the target node.
            sinks: The indices of the sink nodes.

        Returns:
            The computed mean first passage time.
        """
        states = []
        all_sinks = np.hstack((target, sinks))
        for i in range(self.get_num_states()):
            if not i in all_sinks:
                states.append(i)
        K = self.P[states][:, states]  # submatrix of P excluding the sinks
        p0_ = p0[states]

        # find transition prob. from states to target
        k = np.zeros(len(states))
        for i, state in enumerate(states):
            k[i] = np.sum(self.P[target, state])

        K_inv = np.linalg.inv(K)
        mfpt = -(k @ (K_inv @ K_inv @ p0_)) / (k @ (K_inv @ p0_))
        return mfpt

    def compute_hitting_time(self, p_st: Optional[np.ndarray] = None, return_Z: bool = False) -> Union[Tuple, np.ndarray]:
        """Compute the hitting time of the continuous-time Markov chain.

        Args:
            p_st: The stationary distribution of the continuous-time Markov chain.
            return_Z: Whether to return the matrix Z in addition to the hitting time matrix.

        Returns:
            The computed hitting time matrix.
        """
        p_st = self.compute_stationary_distribution() if p_st is None else p_st
        n_nodes = len(p_st)
        W = np.ones((n_nodes, 1)) * p_st
        Z = np.linalg.inv(-self.P.T + W)
        H = np.ones((n_nodes, 1)) * np.diag(Z).T - Z
        H = H / W
        H = H.T
        if return_Z:
            return H, Z
        else:
            return H

    def diffusion_map_embedding(self, n_dims: int = 2, t: Union[int, float] = 1, n_pca_dims: Optional[int] = None) -> np.ndarray:
        """Perform diffusion map embedding for the continuous-time Markov chain.

        Args:
            n_dims: The number of dimensions for the diffusion map embedding.
            t: The diffusion time parameter used in the embedding.
            n_pca_dims: The number of dimensions for PCA before diffusion map embedding.

        Returns:
            The diffusion map embedding of the continuous-time Markov chain.
        """
        if self.D is None:
            self.eigsys()

        ind = np.arange(1, n_dims + 1)
        Y = np.real(np.exp(self.D[ind] ** t)) * np.real(self.U[:, ind])
        if n_pca_dims is not None:
            pca = PCA(n_components=n_pca_dims)
            Y = pca.fit_transform(Y)
        return Y

    """def fit(self, X, V, k, s=None, tol=1e-4):
        self.__reset__()
        # knn clustering
        if self.nbrs_idx is None:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
            _, Idx = nbrs.kneighbors(X)
            self.nbrs_idx = Idx
        else:
            Idx = self.nbrs_idx
        # compute transition prob.
        n = X.shape[0]
        self.P = np.zeros((n, n))
        for i in range(n):
            y = X[i]
            v = V[i]
            Y = X[Idx[i, 1:]]
            p = compute_markov_trans_prob(y, v, Y, s, cont_time=True)
            p[p<=tol] = 0       # tolerance check
            self.P[Idx[i, 1:], i] = p
            self.P[i, i] = - np.sum(p)"""


def markov_combination(x: np.ndarray, v: np.ndarray, X: np.ndarray) -> Tuple:
    """Calculate the Markov combination by solving a 'cvxopt' library quadratic programming (QP) problem, which is
    defined as:
        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h

    Args:
        x: The cell data matrix.
        v: The velocity data matrix.
        X: The neighbors data matrix.

    Returns:
        A tuple containing the results of QP problem.
    """
    from cvxopt import matrix, solvers

    n = X.shape[0]
    R = matrix(X - x).T
    H = R.T * R
    f = matrix(v).T * R
    G = np.vstack((-np.eye(n), np.ones(n)))
    h = np.zeros(n + 1)
    h[-1] = 1
    p = solvers.qp(H, -f.T, G=matrix(G), h=matrix(h))["x"]
    u = R * p
    return p, u


def compute_markov_trans_prob(
    x: np.ndarray, v: np.ndarray, X: np.ndarray, s: Optional[np.ndarray] = None, cont_time: bool = False
) -> np.ndarray:
    """Calculate the Markov transition probabilities by solving a 'cvxopt' library quadratic programming (QP) problem,
    which is defined as:
        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h

    Args:
        x: The cell data matrix.
        v: The velocity data matrix.
        X: The neighbors data matrix.
        s: Extra constraints added to the `q` in QP problem.
        cont_time: Whether is continuous-time or not.

    Returns:
        An array containing the optimal Markov transition probabilities computed by QP problem.
    """
    from cvxopt import matrix, solvers

    n = X.shape[0]
    R = X - x
    # normalize R, v, and s
    Rn = np.array(R, copy=True)
    vn = np.array(v, copy=True)
    scale = np.abs(np.max(R, 0) - np.min(R, 0))
    Rn = Rn / scale
    vn = vn / scale
    if s is not None:
        sn = np.array(s, copy=True)
        sn = sn / scale
        A = np.hstack((Rn, 0.5 * Rn * Rn))
        b = np.hstack((vn, 0.5 * sn * sn))
    else:
        A = Rn
        b = vn

    H = A.dot(A.T)
    f = b.dot(A.T)
    if cont_time:
        G = -np.eye(n)
        h = np.zeros(n)
    else:
        G = np.vstack((-np.eye(n), np.ones(n)))
        h = np.zeros(n + 1)
        h[-1] = 1
    p = solvers.qp(matrix(H), matrix(-f), G=matrix(G), h=matrix(h))["x"]
    p = np.array(p).flatten()
    return p


@jit(nopython=True)
def compute_kernel_trans_prob(
    x: np.ndarray, v: np.ndarray, X: np.ndarray, inv_s: Union[np.ndarray, float], cont_time: bool = False
) -> np.ndarray:
    """Calculate the transition probabilities.

    Args:
        x: The cell data matrix representing current state.
        v: The velocity data matrix.
        X: An array of data points representing the neighbors.
        inv_s: The inverse of the diffusion matrix or a scalar value.
        cont_time: Whether to use continuous-time kernel computation.

    Returns:
        The computed transition probabilities for each state in the Markov chain.
    """
    n = X.shape[0]
    p = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        p[i] = np.exp(-0.25 * (d - v) @ inv_s @ (d - v).T)
    p /= np.sum(p)
    return p


# @jit(nopython=True)
def compute_drift_kernel(x: np.ndarray, v: np.ndarray, X: np.ndarray, inv_s: Union[np.ndarray, float]) -> np.ndarray:
    """Compute the kernal representing the drift based on input data and parameters.

    Args:
        x: The cell data matrix representing current state.
        v: The velocity data matrix.
        X: An array of data points representing the neighbors.
        inv_s: The inverse of the diffusion matrix or a scalar value.

    Returns:
        The computed drift kernel values for each state in the Markov chain.
    """
    n = X.shape[0]
    k = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        if np.isscalar(inv_s):
            k[i] = np.exp(-0.25 * inv_s * (d - v).dot(d - v))
        else:
            k[i] = np.exp(-0.25 * (d - v) @ inv_s @ (d - v).T)
    return k


"""def compute_drift_local_kernel(x, v, X, inv_s):
    n = X.shape[0]
    k = np.zeros(n)
    # compute tau
    D = X - x
    dists = np.zeros(n)
    vds = np.zeros(n)
    for (i, d) in enumerate(D):
        dists[i] = np.linalg.norm(d)
        if dists[i] > 0:
            vds[i] = v.dot(d) / dists[i]
    i_dir = np.logical_and(vds >= np.quantile(vds, 0.7), vds > 0)
    tau = np.mean(dists[i_dir] / vds[i_dir])
    if np.isnan(tau): tau = 1
    if tau > 1e2: tau = 1e2

    tau_v = tau * v
    tau_invs = (1 / (tau * np.linalg.norm(v))) * inv_s
    for i in range(n):
        d = D[i]
        k[i] = np.exp(-0.25 * (d-tau_v) @ tau_invs @ (d-tau_v).T)
    return k, tau_invs"""


# @jit(nopython=True)
def compute_drift_local_kernel(x: np.ndarray, v: np.ndarray, X: np.ndarray, inv_s: Union[np.ndarray, float]) -> np.ndarray:
    """Compute a local kernel representing the drift based on input data and parameters.

    Args:
        x: The cell data matrix representing current state.
        v: The velocity data matrix.
        X: An array of data points representing the neighbors.
        inv_s: The inverse of the diffusion matrix or a scalar value.

    Returns:
        The computed drift kernel values.
    """
    n = X.shape[0]
    k = np.zeros(n)
    # compute tau
    D = X - x
    dists = np.zeros(n)
    vds = np.zeros(n)
    for (i, d) in enumerate(D):
        dists[i] = np.linalg.norm(d)
        if dists[i] > 0:
            vds[i] = v.dot(d) / dists[i]
    i_dir = np.logical_and(vds >= np.quantile(vds, 0.7), vds > 0)
    if np.any(i_dir):
        tau = np.mean(dists[i_dir] / vds[i_dir])
        if tau > 1e2:
            tau = 1e2
        tau_v = tau * v
        tau_invs = (1 / (tau * v.dot(v))) * inv_s
    else:
        tau_v = 0
        tau_invs = (1 / (1e2 * v.dot(v))) * inv_s
    for i in range(n):
        d = D[i]
        if np.isscalar(tau_invs):
            k[i] = np.exp(-0.25 * tau_invs * (d - tau_v).dot(d - tau_v))
        else:
            k[i] = np.exp(-0.25 * (d - tau_v) @ tau_invs @ (d - tau_v).T)
    return k


@jit(nopython=True)
def compute_density_kernel(x: np.ndarray, X: np.ndarray, inv_eps: float) -> np.ndarray:
    """Compute the density kernel values.

    Args:
        x: The cell data matrix representing current state.
        X: An array of data points representing the neighbors.
        inv_eps: The inverse of the epsilon.

    Returns:
        The computed density kernel values for each state.
    """
    n = X.shape[0]
    k = np.zeros(n)
    for i in range(n):
        d = X[i] - x
        k[i] = np.exp(-0.25 * inv_eps * d.dot(d))
    return k


@jit(nopython=True)
def makeTransitionMatrix(Qnn: np.ndarray, I_vec: np.ndarray, tol: float = 0.0) -> np.ndarray:  # Qnn, I, tol=0.0
    """Create the transition matrix based on the transition rate matrix `Qnn` and the indexing vector `I_vec`.

    Args:
        Qnn: The matrix which represents the transition rates between different states.
        I_vec: The indexing vector to map the rows to the appropriate positions in the transition matrix.
        tol: A numerical tolerance value to consider rate matrix elements as zero.

    Returns:
        The computed transition matrix based on `Qnn` and `I_vec`.
    """
    n = Qnn.shape[0]
    M = np.zeros((n, n))

    for i in range(n):
        q = Qnn[i]
        q[q < tol] = 0
        M[I_vec[i], i] = q
        M[i, i] = 1 - np.sum(q)
    return M


@jit(nopython=True)
def compute_tau(X: np.ndarray, V: np.ndarray, k: int = 100, nbr_idx: Optional[np.ndarray] = None) -> Tuple:
    """Compute the tau values for each state in `X` based on the local density and velocity magnitudes.

    Args:
        X: The data matrix which represents the states of the system.
        V: The velocity matrix which represents the velocity vectors associated with each state in `X`.
        k: The number of neighbors to consider when estimating local density. Default is 100.
        nbr_idx: The indices of neighbors for each state in `X`.

    Returns:
        The computed tau values representing the timescale of transitions for each state in `X`. The computed velocity
        magnitudes for each state in `X`.
    """

    if nbr_idx is None:
        _, dists = k_nearest_neighbors(
            X,
            k=k - 1,
            exclude_self=False,
            pynn_rand_state=19491001,
            n_jobs=-1,
        )
    else:
        dists = np.zeros(nbr_idx.shape)
        for i in range(nbr_idx.shape[0]):
            for j in range(nbr_idx.shape[1]):
                x = X[i]
                y = X[nbr_idx[i, j]]
                dists[i, j] = np.sqrt((x - y).dot(x - y))
    d = np.mean(dists[:, 1:], 1)
    v = np.linalg.norm(V, axis=1)
    tau = d / v
    return tau, v


# we might need a separate module/file for discrete vector field and markovian methods in the future
def graphize_velocity(
    V: np.ndarray,
    X: np.ndarray,
    nbrs_idx: Optional[list] = None,
    k: int = 30,
    normalize_v: bool = False,
    E_func: Optional[Union[Callable, str]] = None
) -> Tuple:
    """The function generates a graph based on the velocity data. The flow from i- to j-th
    node is returned as the edge matrix E[i, j], and E[i, j] = -E[j, i].

    Args:
        V: The velocities for all cells.
        X: The coordinates for all cells.
        nbrs_idx: A list of neighbor indices for each cell. If None a KNN will be performed instead.
        k: The number of neighbors for the KNN search.
        normalize_v: Whether to normalize the velocity vectors.
        E_func: A variance stabilizing function for reducing the variance of the flows.
            If a string is passed, there are two options:
                'sqrt': the `numpy.sqrt` square root function;
                'exp': the `numpy.exp` exponential function.

    Returns:
        The edge matrix and the neighbor indices.
    """
    n, d = X.shape

    nbrs = None
    if nbrs_idx is None:
        nbrs_idx, _ = k_nearest_neighbors(
            X,
            k=k,
            exclude_self=False,
            pynn_rand_state=19491001,
        )

    if type(E_func) is str:
        if E_func == "sqrt":
            E_func = np.sqrt
        elif E_func == "exp":
            E_func = np.exp
        else:
            raise NotImplementedError("The specified edge function is not implemented.")

    # E = sp.csr_matrix((n, n))      # Making E a csr_matrix will slow down this process. Try lil_matrix maybe?
    E = np.zeros((n, n))
    for i in range(n):
        x = flatten(X[i])
        idx = nbrs_idx[i]
        if len(idx) > 0 and idx[0] == i:  # excluding the node itself from the neighbors
            idx = idx[1:]
        vi = flatten(V[i])
        if normalize_v:
            vi_norm = np.linalg.norm(vi)
            if vi_norm > 0:
                vi /= vi_norm

        # normalized differences
        U = X[idx] - x
        U_norm = np.linalg.norm(U, axis=1)
        U_norm[U_norm == 0] = 1
        U /= U_norm[:, None]

        for jj, j in enumerate(idx):
            vj = flatten(V[j])
            if normalize_v:
                vj_norm = np.linalg.norm(vj)
                if vj_norm > 0:
                    vj /= vj_norm
            u = flatten(U[jj])
            v = np.mean((vi.dot(u), vj.dot(u)))

            if E_func is not None:
                v = np.sign(v) * E_func(np.abs(v))
            E[i, j] = v
            E[j, i] = -v

    return E, nbrs_idx


def calc_Laplacian(E: np.ndarray, convention: str = "graph") -> np.ndarray:
    """Calculate the Laplacian matrix of a given matrix of edge weights.

    Args:
        E: The matrix of edge weights which represents the weights of edges in a graph.
        convention: The convention used to compute the Laplacian matrix.
            If "graph", the Laplacian matrix will be calculated as the diagonal matrix of node degrees minus the adjacency matrix.
            If "diffusion", the Laplacian matrix will be calculated as the negative of the graph Laplacian.
            Default is "graph".

    Returns:
        The Laplacian matrix.
    """
    A = np.abs(np.sign(E))
    L = np.diag(np.sum(A, 0)) - A

    if convention == "diffusion":
        L = -L

    return L


def fp_operator(E: np.ndarray, D: Union[int, float]) -> np.ndarray:
    """Calculate the Fokker-Planck operator based on the given matrix of edge weights (E) and diffusion coefficient (D).

    Args:
        E: The matrix of edge weights.
        D: The diffusion coefficient used in the Fokker-Planck operator.

    Returns:
        The Fokker-Planck operator matrix.
    """
    # drift
    Mu = E.T.copy()
    Mu[Mu < 0] = 0
    Mu = np.diag(np.sum(Mu, 0)) - Mu
    # diffusion
    L = calc_Laplacian(E, convention="diffusion")

    return -Mu + D * L


def divergence(E: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Calculate the divergence for each node in a given matrix of edge weights.

    Args:
        E: The matrix of edge weights.
        tol: The tolerance value. Edge weights below this value will be treated as zero.

    Returns:
        The divergence values for each node.
    """
    n = E.shape[0]
    div = np.zeros(n)
    # optimize for sparse matrices later...
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(E[i, j]) > tol:
                div[i] += E[i, j] - E[j, i]

    return div
