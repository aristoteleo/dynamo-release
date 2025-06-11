from typing import Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm



from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.sparse import issparse

from ..dynamo_logger import LoggerManager
from ..tools.utils import fetch_states, nearest_neighbors
from ..utils import pca_to_expr
from ..vectorfield import SvcVectorField
from ..vectorfield.utils import (
    vecfld_from_adata,
    vector_field_function_transformation,
    vector_transformation,
)
from .trajectory import GeneTrajectory, Trajectory, arclength_sampling_n
from .utils import find_elbow


from ..pl import kinetic_heatmap,multiplot
from ..vf import rank_genes
from ..tools.utils import nearest_neighbors


class LeastActionPath(Trajectory):
    """A class for computing the Least Action Path for a given function and initial conditions.

    Args:
        X: The initial conditions as a 2D array of shape (n, m), where n is the number of
                     points in the trajectory and m is the dimension of the system.
        vf_func: The vector field function that governs the system.
        D: The diffusion constant of the system. Defaults to 1.
        dt: The time step for the simulation. Defaults to 1.

    Attributes:
        func: The vector field function that governs the system.
        D: The diffusion constant of the system.
        _action: The Least Action Path action values for each point in the trajectory.

    Methods:
        get_t(): Returns the time points of the least action path.
        get_dt(): Returns the time step of the least action path.
        action(t=None, **interp_kwargs): Returns the Least Action Path action values at time t.
                                         If t is None, returns the action values for all time points.
                                         **interp_kwargs are passed to the interp1d function.
        mfpt(action=None): Returns the mean first passage time using the action values.
                           If action is None, uses the action values stored in the _action attribute.
        optimize_dt(): Optimizes the time step of the simulation to minimize the Least Action Path action.
    """

    def __init__(self, X: np.ndarray, vf_func: Callable, D: float = 1, dt: float = 1) -> None:
        """Initializes the LeastActionPath class instance with the given initial conditions, vector field function,
        diffusion constant and time step.

        Args:
            X: The initial conditions as a 2D array of shape (n, m), where n is the number of
                         points in the trajectory and m is the dimension of the system.
            vf_func: The vector field function that governs the system.
            D: The diffusion constant of the system. Defaults to 1.
            dt: The time step for the simulation. Defaults to 1.
        """
        super().__init__(X, t=np.arange(X.shape[0]) * dt)
        self.func = vf_func
        self.D = D
        self._action = np.zeros(X.shape[0])
        for i in range(1, len(self._action)):
            self._action[i] = action(self.X[: i + 1], self.func, self.D, dt)

    def get_t(self) -> np.ndarray:
        """Returns the time points of the trajectory.

        Returns:
            The time points of the trajectory.
        """
        return self.t

    def get_dt(self) -> float:
        """Returns the time step of the trajectory.

        Returns:
            The time step of the trajectory.
        """
        return np.mean(np.diff(self.t))

    def action_t(self, t: Optional[float] = None, **interp_kwargs) -> np.ndarray:
        """Returns the Least Action Path action values at time t.

        Args:
            t: The time point(s) to return the action value(s) for.
                                 If None, returns the action values for all time points.
                                 Defaults to None.
            **interp_kwargs: Additional keyword arguments to pass to the interp1d function.

        Returns:
            The Least Action Path action value(s).
        """
        if t is None:
            return self._action
        else:
            return interp1d(self.t, self._action, **interp_kwargs)(t)

    def mfpt(self, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Eqn. 7 of Epigenetics as a first exit problem.

        Args:
            action: The action values. If None, uses the action values stored in the _action attribute.

        Returns:
            The mean first passage time.
        """
        action = self._action if action is None else action
        return 1 / np.exp(-action)

    def optimize_dt(self) -> float:
        """Optimizes the time step of the simulation to minimize the Least Action Path action.

        Returns:
            Optimal time step
        """
        dt_0 = self.get_dt()
        t_dict = minimize(lambda t: action(self.X, self.func, D=self.D, dt=t), dt_0)
        LoggerManager.main_logger.info(t_dict["message"])
        LoggerManager.main_logger.info("optimal action: %f" % t_dict["fun"])
        dt_sol = t_dict["x"][0]
        self.t = np.arange(self.X.shape[0]) * dt_sol
        return dt_sol


class GeneLeastActionPath(GeneTrajectory):
    """A class for computing the least action path trajectory and action for a gene expression dataset.
    Inherits from GeneTrajectory class.

    Attributes:
        adata: AnnData object containing the gene expression dataset.
        X: Expression data.
        to_pca: Transformation matrix from gene expression space to PCA space.
        from_pca: Transformation matrix from PCA space to gene expression space.
        PCs: Principal components from PCA analysis.
        func: Vector field function reconstructed within the PCA space.
        D: Diffusivity value.
        t: Array of time values.
        action: Array of action values.
    """

    def __init__(
        self,
        adata: AnnData,
        lap: LeastActionPath = None,
        X_pca: Optional[np.ndarray] = None,
        vf_func: Optional[Callable] = None,
        D: float = 1,
        dt: float = 1,
        **kwargs,
    ) -> None:
        """Initializes the GeneLeastActionPath class instance.

        Args:
            adata: AnnData object containing the gene expression dataset.
            lap: LeastActionPath object. Defaults to None.
            X_pca: PCA transformed expression data. Defaults to None.
            vf_func: Vector field function. Defaults to None.
            D: Diffusivity value. Defaults to 1.
            dt: Time step size. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the GeneTrajectory class.
        """
        if lap is not None:
            self.from_lap(adata, lap, **kwargs)
        else:
            super().__init__(X_pca=X_pca, t=np.arange(X_pca.shape[0]) * dt, **kwargs)
            self.func = vector_field_function_transformation(vf_func, self.PCs, self.to_pca)
            self.D = D

        self.adata = adata
        if self.X is not None and self.func is not None:
            self.action = self.genewise_action()

    def from_lap(self, adata: AnnData, lap: LeastActionPath, **kwargs):
        """
        Initializes class from a LeastActionPath object.

        Args:
            adata: AnnData object containing the gene expression dataset.
            lap: LeastActionPath object.
            **kwargs: Additional keyword arguments passed to the GeneTrajectory class.
        """
        super().__init__(adata, X_pca=lap.X, t=lap.t, **kwargs)
        self.func = vector_field_function_transformation(lap.func, self.PCs, self.to_pca)
        self.D = lap.D

    def get_t(self) -> np.ndarray:
        """Returns the array of time values.

        Returns:
            Array of time values.
        """
        return self.t

    def get_dt(self) -> float:
        """Returns the average time step size.

        Returns:
            Average time step size.
        """
        return np.mean(np.diff(self.t))

    def genewise_action(self) -> np.ndarray:
        """Calculates the genewise action values.

        Returns:
            Array of genewise action values.
        """
        dt = self.get_dt()
        x = (self.X[:-1] + self.X[1:]) * 0.5
        v = np.diff(self.X, axis=0) / dt

        s = v - self.func(x)
        s = 0.5 * np.sum(s * s, axis=0) * dt / self.D

        return s

    def select_genewise_action(self, genes: Union[str, List[str]]) -> np.ndarray:
        """Returns the genewise action values for the specified genes.

        Args:
            genes (Union[str, List[str]]): List of gene names or a single gene name.

        Returns:
            Array of genewise action values.
        """
        return super().select_gene(genes, arr=self.action)


def action(path: np.ndarray, vf_func: Callable[[np.ndarray], np.ndarray], D: float = 1, dt: float = 1) -> float:
    """The action function calculates the action (or functional) of a path in space, given a velocity field function
    and diffusion coefficient.

    The path is represented as an array of points in space, and the velocity field is given by vf_func. The function
    first calculates the centers of the segments between each point in the path, and then calculates the velocity at
    each of these centers by taking the average of the velocities at the two neighboring points. The difference
    between the actual velocity and the velocity field at each center is then calculated and flattened into a
    one-dimensional array. The action is then calculated by taking the dot product of this array with itself,
    multiplying by a factor of 0.5*dt/D, where dt is the time step used to define the path, and D is the diffusion
    coefficient.

    Args:
        path: An array of shape (N, d) containing the coordinates of a path with N points in d dimensions.
        vf_func: A callable that takes an array of shape (d,) as input and returns a vector field at that point.
        D: A scalar representing the diffusion coefficient.
        dt: A scalar representing the time step size.

    Returns:
        A scalar representing the action along the path.
    """
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt

    s = (v - vf_func(x)).flatten()
    s = 0.5 * s.dot(s) * dt / D

    return s


def action_aux(
    path_flatten: np.ndarray,
    vf_func: Callable,
    dim: int,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Auxiliary function for computing the action of a path.

    Args:
        path_flatten: A 1D array representing the flattened path.
        vf_func: A function that computes the velocity field vf(x) for a given position x.
        dim: The dimension of the path.
        start: The starting point of the path.
        end: The end point of the path.
        **kwargs: Additional keyword arguments to pass to the action function.

    Returns:
        The action of the path.
    """
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action(path, vf_func, **kwargs)


def action_grad(path: np.ndarray, vf_func: Callable, jac_func: Callable, D: float = 1.0, dt: float = 1.0) -> np.ndarray:
    """Computes the gradient of the action functional with respect to the path.

    Args:
        path: A 2D array of shape (n+1,d) representing the path, where n is the number of time steps and d is the
            dimension of the path.
        vf_func: A function that computes the velocity field vf(x) for a given position x.
        jac_func: A function that computes the Jacobian matrix of the velocity field at a given position.
        D: The diffusion constant (default is 1).
        dt: The time step (default is 1).

    Returns:
        The gradient of the action functional with respect to the path, as a 2D array of shape (n,d).
    """
    x = (path[:-1] + path[1:]) * 0.5
    v = np.diff(path, axis=0) / dt

    dv = v - vf_func(x)
    J = jac_func(x)
    z = np.zeros(dv.shape)
    for s in range(dv.shape[0]):
        z[s] = dv[s] @ J[:, :, s]
    grad = (dv[:-1] - dv[1:]) / D - dt / (2 * D) * (z[:-1] + z[1:])
    return grad


def action_grad_aux(
    path_flatten: np.ndarray,
    vf_func: Callable,
    jac_func: Callable,
    dim: int,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Auxiliary function for computing the gradient of the action functional with respect to the path.

    Args:
        path_flatten: A 1D array representing the flattened path.
        vf_func: A function that computes the velocity field vf(x) for a given position x.
        jac_func: A function that computes the Jacobian matrix of the velocity field at a given position.
        dim: The dimension of the path.
        start: The starting point of the path.
        end: The end point of the path.
        **kwargs: Additional keyword arguments to pass to the action_grad function.

    Returns:
        The gradient of the action functional with respect to the path.
    """
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action_grad(path, vf_func, jac_func, **kwargs).flatten()


def reshape_path(
    path_flatten: np.ndarray,
    dim: int,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reshapes a flattened path into a 2D array.

    Args:
        path_flatten: A 1D array representing the flattened path.
        dim: The dimension of the path.
        start: The starting point of the path.
        end: The end point of the path.

    Returns:
        A 2D array representing the path.
    """
    path = path_flatten.reshape(int(len(path_flatten) / dim), dim)
    if start is not None:
        path = np.vstack((start, path))
    if end is not None:
        path = np.vstack((path, end))
    return path


def lap_T(
    path_0: List[np.ndarray],
    T: float,
    vf_func: Callable[[np.ndarray], np.ndarray],
    jac_func: Callable[[np.ndarray], np.ndarray],
    D: float = 1,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute a time-optimal path between two points with a given velocity field.

    Args:
        path_0: An array of points representing the initial path, where each point is a list of floats.
        T: A float representing the maximum time to reach the end of the path.
        vf_func: A function that takes a point and returns a velocity vector as an array.
        jac_func: A function that takes a point and returns the Jacobian matrix of the velocity field
            at that point as an array.
        D: A float representing the cost per unit of time.

    Returns:
        A tuple containing the following elements:
        - path_sol: An array of points representing the optimized path.
        - dt_sol: A float representing the time step used to compute the optimized path.
        - action_opt: A float representing the minimum action (cost) of the optimized path.
    """
    n = len(path_0)
    dt = T / (n - 1)
    dim = len(path_0[0])

    def fun(x):
        return action_aux(x, vf_func, dim, start=path_0[0], end=path_0[-1], D=D, dt=dt)

    def jac(x):
        return action_grad_aux(x, vf_func, jac_func, dim, start=path_0[0], end=path_0[-1], D=D, dt=dt)

    sol_dict = minimize(fun, path_0[1:-1].flatten(), jac=jac)
    path_sol = reshape_path(sol_dict["x"], dim, start=path_0[0], end=path_0[-1])

    # further optimization by varying dt
    t_dict = minimize(lambda t: action(path_sol, vf_func, D=D, dt=t), dt)
    action_opt = t_dict["fun"]
    dt_sol = t_dict["x"][0]

    return path_sol, dt_sol, action_opt


def least_action_path(
    start: np.ndarray,
    end: np.ndarray,
    vf_func: Callable[[np.ndarray], np.ndarray],
    jac_func: Callable[[np.ndarray], np.ndarray],
    n_points: int = 20,
    init_path: Optional[np.ndarray] = None,
    D: float = 1,
    dt_0: float = 1,
    EM_steps: int = 2,
) -> Tuple[np.ndarray, float, float]:
    """
    Computes the least action path between two points in a given vector field.

    Args:
        start: The starting point of the path.
        end: The end point of the path.
        vf_func: A function that computes the vector field at a given point.
        jac_func: A function that computes the Jacobian of the vector field at a given point.
        n_points: The number of points to use in the initial path.
        init_path: An optional initial path to use instead of the default linear path.
        D: The diffusion coefficient.
        dt_0: The initial time step to use.
        EM_steps: The number of expectation-maximization steps to use in the Laplace method.

    Returns:
        A tuple containing the least action path, the optimal time step, and the minimum action value.
    """
    if init_path is None:
        path = (
            np.tile(start, (n_points + 1, 1))
            + (np.linspace(0, 1, n_points + 1, endpoint=True) * np.tile(end - start, (n_points + 1, 1)).T).T
        )
    else:
        path = np.array(init_path, copy=True)

    # initial dt estimation:
    t_dict = minimize(lambda t: action(path, vf_func, D=D, dt=t), dt_0)
    dt = t_dict["x"][0]

    while EM_steps > 0:
        EM_steps -= 1
        path, dt, action_opt = lap_T(path, dt * len(path), vf_func, jac_func, D=D)

    return path, dt, action_opt


def minimize_lap_time(
    path_0: np.ndarray,
    t0: float,
    t_min: float,
    vf_func: Callable,
    jac_func: Callable,
    D: Union[float, int, np.ndarray] = 1,
    num_t: int = 20,
    elbow_method: str = "hessian",
    hes_tol=3,
) -> Tuple[int, List[np.ndarray], np.ndarray, np.ndarray]:
    """Minimize the least action path time.

    Args:
        path_0: The initial path.
        t0: The initial time to start the minimization.
        t_min: The minimum time to consider.
        vf_func: The vector field function.
        jac_func: The Jacobian function.
        D: The diffusion constant or matrix.
        num_t: The number of time steps.
        elbow_method: The method to use to find the elbow in the action vs time plot.
        hes_tol: The tolerance to use for the elbow method.

    Returns:
        A tuple containing the following elements:
            - i_elbow: The index of the elbow in the action vs time plot.
            - laps: A list of the least action paths for each time step.
            - A: An array of action values for each time step.
    """
    T = np.linspace(t_min, t0, num_t)
    A = np.zeros(num_t)
    opt_T = np.zeros(num_t)
    laps = []

    for i, t in enumerate(T):
        path, dt, action = lap_T(path_0, t, vf_func, jac_func, D=D)
        A[i] = action
        opt_T[i] = dt * (len(path_0) - 1)
        laps.append(path)

    i_elbow = find_elbow(opt_T, A, method=elbow_method, order=-1, tol=hes_tol)

    return i_elbow, laps, A, opt_T


def get_init_path(
    G: nx.Graph,
    start: np.ndarray,
    end: np.ndarray,
    coords: np.ndarray,
    interpolation_num: int = 20,
) -> np.ndarray:
    """Get the initial path for the least action path calculation.

    Args:
        G: A networkx graph representing the cell state space.
        start: The starting point of the path.
        end: The end point of the path.
        coords: The coordinates of the cell states.
        interpolation_num: The number of points to use in the initial path.

    Returns:
        The initial path for the least action path calculation.
    """
    source_ind = nearest_neighbors(start, coords, k=1)[0][0]
    target_ind = nearest_neighbors(end, coords, k=1)[0][0]

    path = nx.shortest_path(G, source_ind, target_ind)
    init_path = coords[path, :]

    # _, arclen, _ = remove_redundant_points_trajectory(init_path, tol=1e-4, output_discard=True)
    # arc_stepsize = arclen / (interpolation_num - 1)
    # init_path_final, _, _ = arclength_sampling(init_path, step_length=arc_stepsize, t=np.arange(len(init_path)))
    init_path_final, _, _ = arclength_sampling_n(init_path, interpolation_num, t=np.arange(len(init_path)))

    # add the beginning and end point
    init_path_final = np.vstack((start, init_path_final, end))

    return init_path_final


def least_action(
    adata: AnnData,
    init_cells: Union[str, list],
    target_cells: Union[str, list],
    init_states: Union[None, np.ndarray] = None,
    target_states: Union[None, np.ndarray] = None,
    paired: bool = True,
    min_lap_t=False,
    elbow_method="hessian",
    num_t=20,
    basis: str = "pca",
    vf_key: str = "VecFld",
    vecfld: Union[None, Callable] = None,
    adj_key: str = "pearson_transition_matrix",
    n_points: int = 25,
    init_paths: Union[None, np.ndarray, list] = None,
    D: int = 10,
    PCs: Union[None, str] = None,
    expr_func: callable = np.expm1,
    add_key: Union[None, str] = None,
    **kwargs,
) -> Union[LeastActionPath, List[LeastActionPath]]:
    """Calculate the optimal paths between any two cell states.

    Args:
        adata: Anndata object that has vector field function computed.
        init_cells: Cell name or indices of the initial cell states, that will be used to find the initial cell state when
          optimizing for the least action paths. If the names in init_cells not found in the adata.obs_name, it will
          be treated as cell indices and must be integers.
        target_cells: Cell name or indices of the terminal cell states , that will be used to find the target cell state when
          optimizing for the least action paths. If the names in init_cells not found in the adata.obs_name, it will
          be treated as cell indices and must be integers.
        init_states: Initial cell states for least action path calculation.
        target_states: Target cell states for least action path calculation.
        paired: Whether the initial states and target states will be treated as pairs; if not, all combination of intial and
          target states will be used to find the least action paths between. When paired is used, the long list is
          trimmed to the same length of the shorter one.
        basis: The embedding data to use for predicting the least action path. If `basis` is `pca`, the identified least action paths will be projected back to high dimensional space.
        vf_key: A key to the vector field functions in adata.uns.
        vecfld: A function of vector field.
        adj_key: The key to the adjacency matrix in adata.obsp
        n_points: Number of way points on the least action paths.
        init_paths: Initial paths that will be used for optimization.
        D: The diffusion constant. In theory, the diffusion matrix needs to be a function of cell states but we use a
            constant for the purpose of simplicity.
        PCs: The key to the PCs loading matrix in adata.uns.
        expr_func: The function that is applied before performing PCA analysis.
        add_key: The key name that will be used to store the calculated least action path information.
        kwargs: Additional argument passed to least_action_path function.

    Returns:
        A trajectory class containing the least action paths information. Meanwhile, the anndata object with be updated
        with newly calculated least action paths information.
    """

    logger = LoggerManager.gen_logger("dynamo-least-action-path")

    if vecfld is None:
        vf_dict, func = vecfld_from_adata(adata, basis=basis, vf_key=vf_key)
        if vf_dict["method"] == "dynode":
            vf = vf_dict["dynode_object"]
        else:
            vf = SvcVectorField()
            vf.from_adata(adata, basis=basis, vf_key=vf_key)
    else:
        vf = vecfld

    coords = adata.obsm["X_" + basis]

    T = adata.obsp[adj_key]
    G = nx.convert_matrix.from_scipy_sparse_array(T)

    init_states, _, _, _ = fetch_states(
        adata,
        init_states,
        init_cells,
        basis,
        "X",
        False,
        None,
    )
    target_states, _, _, valid_genes = fetch_states(
        adata,
        target_states,
        target_cells,
        basis,
        "X",
        False,
        None,
    )

    logger.info("searching for the least action path...", indent_level=1)
    logger.log_time()

    init_states = np.atleast_2d(init_states)
    target_states = np.atleast_2d(target_states)

    if paired:
        if init_states.shape[0] != target_states.shape[0]:
            logger.warning("The numbers of initial and target states are not equal. The longer one is trimmed")
            num = min(init_states.shape[0], target_states.shape[0])
            init_states = init_states[:num]
            target_states = target_states[:num]
        pairs = [(init_states[i], target_states[i]) for i in range(init_states.shape[0])]
    else:
        pairs = [(pi, pt) for pi in init_states for pt in target_states]
        logger.warning(
            f"A total of {len(pairs)} pairs of initial and target states will be calculated."
            "To reduce the number of LAP calculations, please use the `paired` mode."
        )

    t, prediction, action, exprs, mftp, trajectory = [], [], [], [], [], []
    if min_lap_t:
        i_elbow = []
        laps = []
        opt_T = []
        A = []

    path_ind = 0
    for init_state, target_state in LoggerManager.progress_logger(
        pairs, progress_name=f"iterating through {len(pairs)} pairs"
    ):
        logger.info(
            "initializing path with the shortest path in the graph built from the velocity transition matrix...",
            indent_level=2,
        )
        if init_paths is None:
            init_path = get_init_path(G, init_state, target_state, coords, interpolation_num=n_points)
        else:
            init_path = init_paths if type(init_paths) == np.ndarray else init_paths[path_ind]

        path_ind += 1
        logger.info(
            "optimizing for least action path...",
            indent_level=2,
        )
        path_sol, dt_sol, action_opt = least_action_path(
            init_state, target_state, vf.func, vf.get_Jacobian(), n_points=n_points, init_path=init_path, D=D, **kwargs
        )

        n_points = len(path_sol)  # the actual #points due to arclength resampling

        if min_lap_t:
            t_sol = dt_sol * (n_points - 1)
            t_min = 0.3 * t_sol
            i_elbow_, laps_, A_, opt_T_ = minimize_lap_time(
                path_sol, t_sol, t_min, vf.func, vf.get_Jacobian(), D=D, num_t=num_t, elbow_method=elbow_method
            )
            if i_elbow_ is None:
                i_elbow_ = 0
            path_sol = laps_[i_elbow_]
            dt_sol = opt_T_[i_elbow_] / (n_points - 1)

            i_elbow.append(i_elbow_)
            laps.append(laps_)
            A.append(A_)
            opt_T.append(opt_T_)

        traj = LeastActionPath(X=path_sol, vf_func=vf.func, D=D, dt=dt_sol)
        trajectory.append(traj)
        t.append(np.arange(path_sol.shape[0]) * dt_sol)
        prediction.append(path_sol)
        action.append(traj.action_t())
        mftp.append(traj.mfpt())

        if basis == "pca":
            pc_keys = "PCs" if PCs is None else PCs
            if pc_keys not in adata.uns.keys():
                logger.warning("Expressions along the trajectories cannot be retrieved, due to lack of `PCs` in .uns.")
            else:
                if "pca_mean" not in adata.uns.keys():
                    pca_mean = None
                else:
                    pca_mean = adata.uns["pca_mean"]
                exprs.append(pca_to_expr(traj.X, adata.uns["PCs"], pca_mean, func=expr_func))

        # logger.info(sol_dict["message"], indent_level=1)
        logger.info("optimal action: %f" % action_opt, indent_level=1)

    if add_key is None:
        LAP_key = "LAP" if basis is None else "LAP_" + basis
    else:
        LAP_key = add_key

    adata.uns[LAP_key] = {
        "init_states": init_states,
        "init_cells": list(init_cells),
        "t": t,
        "mftp": mftp,
        "prediction": prediction,
        "action": action,
        "genes": adata.var_names[adata.var.use_for_pca],
        "exprs": exprs,
        "vf_key": vf_key,
    }

    if min_lap_t:
        adata.uns[LAP_key]["min_t"] = {"A": A, "T": opt_T, "i_elbow": i_elbow, "paths": laps, "method": elbow_method}

    logger.finish_progress(progress_name="least action path")

    return trajectory[0] if len(trajectory) == 1 else trajectory




def select_marginal_cells_advanced(
    adata,
    cell_type_column='cell_type',
    embedding_key='X_umap',
    potential_key='umap_ddhodge_potential',
    graph_key='connectivities',
    method='combined',
    n_candidates=50,
    return_scores=False
):
    """
    Select marginal cells using multiple criteria to find truly peripheral cells
    """
    
    cell_types = adata.obs[cell_type_column].unique()
    cells_indices_dict = {}
    scores_dict = {}
    
    # Get embedding coordinates
    if embedding_key in adata.obsm:
        coords = adata.obsm[embedding_key]
        if issparse(coords):
            coords = coords.toarray()
    else:
        raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm")
    
    print(f"Using method: {method}")
    print(f"Analyzing {len(cell_types)} cell types...")
    
    for ct in cell_types:
        print(f"\nProcessing {ct}...")
        
        # Get cells of current type
        ct_mask = adata.obs[cell_type_column] == ct
        ct_indices = np.where(ct_mask)[0]
        ct_coords = coords[ct_mask]
        
        if len(ct_indices) == 0:
            print(f"Warning: No cells found for type {ct}")
            continue
        
        # Get coordinates of other cell types
        other_mask = ~ct_mask
        other_coords = coords[other_mask]
        
        scores = {}
        
        # Method 1: Distance-based marginality
        if method in ['distance', 'combined']:
            if len(other_coords) > 0:
                distances = cdist(ct_coords, other_coords)
                min_distances = np.min(distances, axis=1)
                scores['distance'] = min_distances
                print(f"  Distance scores range: {min_distances.min():.3f} - {min_distances.max():.3f}")
            else:
                scores['distance'] = np.ones(len(ct_indices))
        
        # Method 2: Graph degree-based marginality  
        if method in ['degree', 'combined']:
            if graph_key in adata.obsp:
                graph = adata.obsp[graph_key]
                if issparse(graph):
                    degrees = np.array(graph.sum(axis=1)).flatten()
                    ct_degrees = degrees[ct_indices]
                    scores['degree'] = 1.0 / (ct_degrees + 1e-6)
                    print(f"  Degree scores range: {ct_degrees.min():.1f} - {ct_degrees.max():.1f}")
                else:
                    scores['degree'] = np.ones(len(ct_indices))
            else:
                print(f"  Warning: Graph key '{graph_key}' not found")
                scores['degree'] = np.ones(len(ct_indices))
        
        # Method 3: Potential-based marginality
        if method in ['potential', 'combined']:
            if potential_key in adata.obs.columns:
                ct_potentials = adata.obs.loc[ct_mask, potential_key].values
                
                if ct == 'HSC':
                    scores['potential'] = -ct_potentials
                else:
                    scores['potential'] = ct_potentials
                
                print(f"  Potential range: {ct_potentials.min():.3f} - {ct_potentials.max():.3f}")
            else:
                scores['potential'] = np.ones(len(ct_indices))
        
        # Combine scores
        if method == 'combined':
            normalized_scores = {}
            for score_name, score_values in scores.items():
                if len(score_values) > 0 and np.std(score_values) > 0:
                    normalized_scores[score_name] = (score_values - np.min(score_values)) / (np.max(score_values) - np.min(score_values))
                else:
                    normalized_scores[score_name] = np.ones_like(score_values)
            
            weights = {'distance': 0.4, 'degree': 0.2, 'potential': 0.4}
            combined_score = np.zeros(len(ct_indices))
            
            for score_name, weight in weights.items():
                if score_name in normalized_scores:
                    combined_score += weight * normalized_scores[score_name]
            
            final_score = combined_score
        else:
            final_score = scores[method]
        
        # Select top candidates
        n_select = min(n_candidates, len(ct_indices))
        top_indices = np.argsort(final_score)[-n_select:][::-1]
        selected_global_idx = ct_indices[top_indices]
        
        print(f"  Selected top {len(selected_global_idx)} candidates")
        print(f"  Best score: {final_score[top_indices[0]]:.3f}")
        
        # Use nearest_neighbors to get the final selection
        best_cell_coord = coords[selected_global_idx[0]].reshape(1, -1)
        cells_indices = nearest_neighbors(best_cell_coord, coords)
        cells_indices_dict[ct] = cells_indices
        
        # Store detailed scores if requested
        if return_scores:
            scores_dict[ct] = {
                'indices': ct_indices,
                'scores': scores,
                'final_score': final_score,
                'selected_idx': selected_global_idx[0],
                'selected_coord': coords[selected_global_idx[0]]
            }
        
        print(f"  Final selected cell: {adata.obs_names[selected_global_idx[0]]}")
    
    if return_scores:
        return cells_indices_dict, scores_dict
    else:
        return cells_indices_dict


def visualize_marginal_cell_selection(
    adata,
    cells_indices_dict,
    scores_dict=None,
    embedding_key='X_umap',
    cell_type_column='cell_type',
    figsize=(12, 8),
    save_path=None
):
    """
    Visualize the selected marginal cells on the embedding
    """
    
    coords = adata.obsm[embedding_key]
    if issparse(coords):
        coords = coords.toarray()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all cells
    cell_types = adata.obs[cell_type_column].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_types)))
    
    for i, ct in enumerate(cell_types):
        ct_mask = adata.obs[cell_type_column] == ct
        ct_coords = coords[ct_mask]
        ax.scatter(ct_coords[:, 0], ct_coords[:, 1], 
                  c=[colors[i]], alpha=0.6, s=20, label=ct)
    
    # Highlight selected marginal cells
    for ct, cell_indices in cells_indices_dict.items():
        if len(cell_indices) > 0 and len(cell_indices[0]) > 0:
            selected_idx = cell_indices[0][0]
            selected_coord = coords[selected_idx]
            ax.scatter(selected_coord[0], selected_coord[1], 
                      c='red', s=200, marker='*', 
                      edgecolors='black', linewidth=2)
            
            # Add text annotation
            ax.annotate(f'{ct}', 
                       (selected_coord[0], selected_coord[1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Selected Marginal Cells (Red Stars)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except:
            print(f"Could not save to {save_path}")
    
    plt.show()


def select_marginal_cells_simple(
    adata,
    source_cell_type='HSC',
    target_cell_types=['Meg','Ery','Bas','Mon','Neu'],
    method='combined'
):
    """
    Simplified function to select marginal cells with optimized parameters
    """
    
    all_cell_types = [source_cell_type] + target_cell_types
    
    # Filter adata to only include specified cell types
    cell_mask = adata.obs['cell_type'].isin(all_cell_types)
    adata_subset = adata[cell_mask].copy()
    
    cells_indices_dict, scores_dict = select_marginal_cells_advanced(
        adata_subset,
        method=method,
        return_scores=True
    )
    
    # Visualize results
    visualize_marginal_cell_selection(
        adata_subset,
        cells_indices_dict,
        scores_dict,
        save_path='marginal_cells_selection.png'
    )
    
    # Convert indices back to original adata
    original_indices_dict = {}
    original_obs_names = adata.obs_names
    subset_obs_names = adata_subset.obs_names
    
    for ct, indices_list in cells_indices_dict.items():
        if len(indices_list) > 0:
            # Map subset indices back to original indices
            subset_indices = indices_list[0]
            original_indices = []
            for subset_idx in subset_indices:
                subset_cell_name = subset_obs_names[subset_idx]
                original_idx = np.where(original_obs_names == subset_cell_name)[0]
                if len(original_idx) > 0:
                    original_indices.append(original_idx[0])
            
            original_indices_dict[ct] = [original_indices]
        else:
            original_indices_dict[ct] = [[]]
    
    return original_indices_dict, scores_dict


def compute_cell_type_transitions(
    adata,
    cell_types,
    potential_column='umap_ddhodge_potential',
    cell_type_column='cell_type',
    reference_cell_types=None,
    basis_list=['umap', 'pca'],
    umap_adj_key='X_umap_distances',
    pca_adj_key='cosine_transition_matrix',
    EM_steps=2,
    top_genes=5,
    enable_plotting=True,
    enable_gene_analysis=True,
    marginal_method='combined',
    verify_selection=False,
    manual_cell_indices=None,
    manual_source_indices=None,
    manual_target_indices=None
):
    """
    Compute transition paths between cell types and analyze gene dynamics
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing single-cell data
    cell_types : list
        List of cell types to analyze
    potential_column : str, default='umap_ddhodge_potential'
        Column name for potential values in adata.obs
    cell_type_column : str, default='cell_type'
        Column name for cell type annotations in adata.obs
    reference_cell_types : list, optional
        List of reference cell types (defaults to first cell type in cell_types)
    basis_list : list, default=['umap', 'pca']
        List of basis to use for transition path calculation
    umap_adj_key : str, default='X_umap_distances'
        Key for adjacency matrix in UMAP space
    pca_adj_key : str, default='cosine_transition_matrix'
        Key for adjacency matrix in PCA space
    EM_steps : int, default=2
        Number of expectation-maximization steps for optimization
    top_genes : int, default=5
        Number of top genes to analyze
    enable_plotting : bool, default=True
        Whether to generate plots
    enable_gene_analysis : bool, default=True
        Whether to perform gene trajectory analysis
    marginal_method : str, default='combined'
        Method for selecting marginal cells: 'distance', 'degree', 'potential', 'combined'
    verify_selection : bool, default=False
        Whether to show marginal cell selection for verification before proceeding
    manual_cell_indices : dict, optional
        Dictionary mapping cell types to specific cell obs indices. 
        Format: {'HSC': [123], 'Meg': [456], ...}
        If provided, overrides automatic marginal cell selection.
    manual_source_indices : dict, optional
        Dictionary mapping cell types to specific source cell obs indices.
        Format: {'HSC': [123, 124], 'Meg': [456], ...}
        Only used if manual_cell_indices is None.
    manual_target_indices : dict, optional  
        Dictionary mapping cell types to specific target cell obs indices.
        Format: {'HSC': [789], 'Meg': [101112], ...}
        Only used if manual_cell_indices is None and for asymmetric transitions.
        
    Returns:
    --------
    transition_graph : dict
        Dictionary containing transition path results for each cell type pair
    cells_indices_dict : dict
        Dictionary mapping cell types to selected cell indices
        
    Examples:
    ---------
    # Example 1: Automatic marginal cell selection
    >>> transition_graph, cells_dict = compute_cell_type_transitions(
    ...     adata, 
    ...     cell_types=['HSC', 'Meg', 'Ery', 'Bas'],
    ...     marginal_method='combined'
    ... )
    
    # Example 2: Manual cell specification using cell names
    >>> cell_names = ['AAACCTGAGAAGGCCT-1', 'AAACCTGAGCAGGTCA-1', 'AAACCTGCAAGCCGAT-1']
    >>> indices = get_cell_indices_by_name(adata, cell_names)
    >>> manual_indices = {
    ...     'HSC': [indices[0]], 
    ...     'Meg': [indices[1]], 
    ...     'Ery': [indices[2]]
    ... }
    >>> transition_graph, cells_dict = compute_cell_type_transitions(
    ...     adata,
    ...     cell_types=['HSC', 'Meg', 'Ery'],
    ...     manual_cell_indices=manual_indices
    ... )
    
    # Example 3: Manual cell specification using criteria-based selection
    >>> hsc_indices = find_cells_by_criteria(adata, 'HSC', {'umap_ddhodge_potential': 'max'})
    >>> meg_indices = find_cells_by_criteria(adata, 'Meg', {'GATA1': 'max'}) 
    >>> manual_indices = {'HSC': hsc_indices, 'Meg': meg_indices}
    >>> transition_graph, cells_dict = compute_cell_type_transitions(
    ...     adata,
    ...     cell_types=['HSC', 'Meg'],
    ...     manual_cell_indices=manual_indices
    ... )
    
    # Example 4: Asymmetric source/target specification
    >>> source_indices = {'HSC': [100, 101]}  # Multiple source cells for HSC
    >>> target_indices = {'Meg': [500], 'Ery': [600]}  # Different target cells
    >>> transition_graph, cells_dict = compute_cell_type_transitions(
    ...     adata,
    ...     cell_types=['HSC', 'Meg', 'Ery'],
    ...     manual_source_indices=source_indices,
    ...     manual_target_indices=target_indices
    ... )
        
    # Example 5: Mixed manual and automatic selection
    >>> # Manually specify some cell types, auto-select others
    >>> manual_indices = {'HSC': [123]}  # Manual HSC
    >>> transition_graph, cells_dict = compute_cell_type_transitions(
    ...     adata,
    ...     cell_types=['HSC', 'Meg', 'Ery', 'Bas'],  # Auto-select Meg, Ery, Bas
    ...     manual_cell_indices=manual_indices,
    ...     marginal_method='combined'
    ... )

    # Example 6: Avoiding duplicate calculations
    >>> # If you specify HSC manually, only Meg and Ery will be auto-calculated
    >>> # HSC will NOT be recalculated automatically
    >>> manual_indices = {'HSC': [100]}  
    >>> transition_graph, cells_dict = compute_cell_type_transitions(
    ...     adata,
    ...     cell_types=['HSC', 'Meg', 'Ery'],  # Only Meg, Ery auto-calculated
    ...     manual_cell_indices=manual_indices,
    ...     marginal_method='combined'
    ... )

    Helper Functions:
    """
    
    # Set reference cell types
    if reference_cell_types is None:
        reference_cell_types = [cell_types[0]]
    
    print("=" * 60)
    print("CELL TYPE TRANSITION ANALYSIS")
    print("=" * 60)
    
    # Step 1: Determine cell selection method and get cell indices
    if manual_cell_indices is not None:
        print("Step 1: Using manually specified cell indices...")
        cells_indices_dict = {}
        
        # Validate and convert manual indices to the expected format
        for ct in cell_types:
            if ct in manual_cell_indices:
                # Convert to the nested list format expected by the rest of the code
                if isinstance(manual_cell_indices[ct], (list, tuple)):
                    cells_indices_dict[ct] = [list(manual_cell_indices[ct])]
                else:
                    cells_indices_dict[ct] = [[manual_cell_indices[ct]]]
                
                print(f"  {ct}: manual indices {manual_cell_indices[ct]}")
            else:
                print(f"  {ct}: will use automatic selection")
        
        # Fill in missing cell types with automatic selection if needed
        missing_cell_types = [ct for ct in cell_types if ct not in cells_indices_dict]
        if missing_cell_types:
            print(f"  Automatically selecting cells for: {missing_cell_types}")
            
            # Only call automatic selection for missing cell types
            if reference_cell_types[0] in missing_cell_types:
                # If reference cell type is missing, include it in automatic selection
                source_ct = reference_cell_types[0]
                target_cts = [ct for ct in missing_cell_types if ct != source_ct]
            else:
                # Reference cell type is manually specified, only auto-select targets
                source_ct = reference_cell_types[0]  # This will be used for graph initialization
                target_cts = missing_cell_types
            
            if target_cts:  # Only run if there are target cell types to auto-select
                auto_cells_dict, _ = select_marginal_cells_simple(
                    adata=adata,
                    source_cell_type=source_ct,
                    target_cell_types=target_cts,
                    method=marginal_method
                )
                # Only update with missing cell types, not all auto-calculated ones
                for ct in missing_cell_types:
                    if ct in auto_cells_dict:
                        cells_indices_dict[ct] = auto_cells_dict[ct]
            
            # Handle case where reference cell type needs auto-selection
            if reference_cell_types[0] in missing_cell_types and reference_cell_types[0] not in cells_indices_dict:
                ref_auto_dict, _ = select_marginal_cells_simple(
                    adata=adata,
                    source_cell_type=reference_cell_types[0],
                    target_cell_types=[],  # Empty target list for reference cell
                    method=marginal_method
                )
                if reference_cell_types[0] in ref_auto_dict:
                    cells_indices_dict[reference_cell_types[0]] = ref_auto_dict[reference_cell_types[0]]
        
        scores_dict = None  # No scores for manual selection
        
    elif manual_source_indices is not None or manual_target_indices is not None:
        print("Step 1: Using manually specified source/target cell indices...")
        cells_indices_dict = {}
        
        # Handle source indices
        if manual_source_indices is not None:
            for ct, indices in manual_source_indices.items():
                if isinstance(indices, (list, tuple)):
                    cells_indices_dict[ct] = [list(indices)]
                else:
                    cells_indices_dict[ct] = [[indices]]
                print(f"  {ct} (source): manual indices {indices}")
        
        # Handle target indices (if different from source)
        if manual_target_indices is not None:
            for ct, indices in manual_target_indices.items():
                if ct not in cells_indices_dict:
                    if isinstance(indices, (list, tuple)):
                        cells_indices_dict[ct] = [list(indices)]
                    else:
                        cells_indices_dict[ct] = [[indices]]
                    print(f"  {ct} (target): manual indices {indices}")
        
        # Fill in missing cell types with automatic selection
        missing_cell_types = [ct for ct in cell_types if ct not in cells_indices_dict]
        if missing_cell_types:
            print(f"  Automatically selecting cells for: {missing_cell_types}")
            
            # Determine source and target for automatic selection
            available_source = reference_cell_types[0] if reference_cell_types[0] in cells_indices_dict else None
            if available_source is None and missing_cell_types:
                # If reference cell is also missing, include it
                available_source = reference_cell_types[0] if reference_cell_types[0] in missing_cell_types else missing_cell_types[0]
            
            missing_targets = [ct for ct in missing_cell_types if ct != available_source]
            
            if missing_targets or available_source in missing_cell_types:
                auto_cells_dict, _ = select_marginal_cells_simple(
                    adata=adata,
                    source_cell_type=available_source,
                    target_cell_types=missing_targets,
                    method=marginal_method
                )
                # Only update with missing cell types
                for ct in missing_cell_types:
                    if ct in auto_cells_dict:
                        cells_indices_dict[ct] = auto_cells_dict[ct]
        
        scores_dict = None  # No scores for manual selection
        
    else:
        print("Step 1: Selecting marginal cells automatically...")
        
        cells_indices_dict, scores_dict = select_marginal_cells_simple(
            adata=adata,
            source_cell_type=reference_cell_types[0],
            target_cell_types=[ct for ct in cell_types if ct not in reference_cell_types],
            method=marginal_method
        )
    
    print(f"\nSelected representative cells:")
    for ct, indices in cells_indices_dict.items():
        if len(indices) > 0 and len(indices[0]) > 0:
            selected_idx = indices[0][0]
            selected_name = adata.obs_names[selected_idx]
            print(f"  {ct}: {selected_name} (index: {selected_idx})")
    
    # Verification step (only for automatic selection)
    if verify_selection and manual_cell_indices is None and manual_source_indices is None and manual_target_indices is None:
        print("\n" + "=" * 60)
        print("VERIFICATION: Please check the red stars in the plot above.")
        print("Do the selected cells look like they are on the periphery of each cell type?")
        
        user_input = input("Continue with these selections? (y/n): ").lower().strip()
        
        if user_input != 'y':
            print("Analysis interrupted by user. Please adjust parameters and try again.")
            return None, cells_indices_dict

    # Step 2: Calculate transition paths
    print("\n" + "=" * 60)
    print("Step 2: Calculating transition paths...")
    print("=" * 60)
    
    transition_graph = {}
    start_cell_indices = [cells_indices_dict[ct] for ct in cell_types]
    end_cell_indices = start_cell_indices
    
    for i, start in enumerate(tqdm(start_cell_indices, desc="Source cell types")):
        for j, end in enumerate(tqdm(end_cell_indices, desc="Target cell types", leave=False)):
            if i != j:  # Skip self-to-self transitions
                start_ct = cell_types[i]
                end_ct = cell_types[j]
                transition_name = f"{start_ct}->{end_ct}"
                
                print(f"\nComputing {transition_name}")
                
                # Determine if source is a reference cell type
                min_lap_t = start_ct in reference_cell_types
                
                # Get cell names
                start_cell_name = adata.obs_names[start[0][0]]
                end_cell_name = adata.obs_names[end[0][0]]
                
                # Calculate least action path in different basis
                lap_results = {}
                
                for basis in basis_list:
                    adj_key = umap_adj_key if basis == 'umap' else pca_adj_key
                    
                    # Compute least action path
                    lap = least_action(
                        adata,
                        [start_cell_name],
                        [end_cell_name],
                        basis=basis,
                        adj_key=adj_key,
                        min_lap_t=min_lap_t,
                        EM_steps=EM_steps,
                    )
                    
                    lap_results[basis] = lap
                    
                    # Plotting
                    if enable_plotting:
                        from ..pl import least_action as pl_least_action
                        pl_least_action(adata, basis=basis)
                
                # Gene trajectory analysis
                gene_analysis_results = {}
                if enable_gene_analysis and 'pca' in lap_results:
                    print(f"  Performing gene trajectory analysis...")
                    
                    # Plot kinetic heatmap
                    if enable_plotting:
                        kinetic_heatmap(
                            adata,
                            basis="pca",
                            mode="lap",
                            genes=adata.var_names[adata.var.use_for_transition],
                            project_back_to_high_dim=True,
                        )
                    
                    # Gene trajectory analysis
                    gtraj = GeneTrajectory(adata)
                    gtraj.from_pca(lap_results['pca'].X, t=lap_results['pca'].t)
                    gtraj.calc_msd()
                    
                    # Gene ranking
                    ranking = rank_genes(adata, "traj_msd")
                    genes = ranking[:top_genes]["all"].to_list()
                    
                    # Plot gene trajectories
                    if enable_plotting:
                        arr = gtraj.select_gene(genes)
                        multiplot(
                            lambda k: [plt.plot(arr[k, :]), plt.title(genes[k])], 
                            np.arange(len(genes))
                        )
                    
                    gene_analysis_results = {
                        "ranking": ranking,
                        "gtraj": gtraj,
                        "top_genes": genes
                    }
                
                # Save results
                transition_graph[transition_name] = {
                    "lap_results": lap_results,
                    **{f"LAP_{basis}": adata.uns[f"LAP_{basis}"] for basis in basis_list if f"LAP_{basis}" in adata.uns},
                    **gene_analysis_results
                }
    
    print(f"\n" + "=" * 60)
    print(f"COMPLETED! Calculated {len(transition_graph)} transition paths")
    print("=" * 60)
    
    return transition_graph, cells_indices_dict



def extract_transition_metrics(
    transition_graph,
    cells_indices_dict,
    cell_types,
    transcription_factors,
    top_tf_genes=10,
    lap_method='action'  # 'action' or 'action_t'
):
    """
    Extract transition metrics with error handling and method fallback
    
    Parameters:
    -----------
    lap_method : str, default='action'
        Method to call on lap object: 'action' or 'action_t'
    """
    
    action_df = pd.DataFrame(index=cell_types, columns=cell_types)
    t_df = pd.DataFrame(index=cell_types, columns=cell_types)
    tf_genes_results = {}
    
    # Create list of cell indices
    cell_indices_list = [cells_indices_dict[ct] for ct in cell_types]
    
    for i, start in enumerate(cell_indices_list):
        for j, end in enumerate(cell_indices_list):
            if start is not end:
                transition_name = cell_types[i] + "->" + cell_types[j]
                print(transition_name, end=",")
                
                try:
                    # Get data from transition_graph
                    transition_data = transition_graph[transition_name]
                    
                    # Try different keys for lap data
                    if "lap" in transition_data:
                        lap = transition_data["lap"]
                    elif "lap_results" in transition_data and "pca" in transition_data["lap_results"]:
                        lap = transition_data["lap_results"]["pca"] 
                    else:
                        print(f"\nWarning: Could not find lap data for {transition_name}")
                        continue
                    
                    gtraj = transition_data["gtraj"]
                    ranking = transition_data["ranking"].copy()
                    
                    # Filter for transcription factors
                    ranking["TF"] = [gene in transcription_factors for gene in list(ranking["all"])]
                    genes = ranking.query("TF == True").head(top_tf_genes)["all"].to_list()
                    
                    # Fallback to top genes if no TF genes found
                    if len(genes) == 0:
                        genes = ranking.head(top_tf_genes)["all"].to_list()
                    
                    tf_genes_results[transition_name] = genes
                    
                    # Select gene array
                    arr = gtraj.select_gene(genes)
                    
                    # Extract action value with method selection
                    if lap_method == 'action_t':
                        action_value = lap.action_t()[-1]
                    else:
                        action_value = lap.action()[-1]
                    
                    action_df.loc[cell_types[i], cell_types[j]] = action_value
                    t_df.loc[cell_types[i], cell_types[j]] = lap.t[-1]
                    
                except Exception as e:
                    print(f"\nError processing {transition_name}: {str(e)}")
                    continue
    
    print("\nExtraction completed!")
    return action_df, t_df, tf_genes_results



import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd

def plot_kinetic_heatmap(
    adata,
    cells_indices_dict,
    source_cell_type,
    target_cell_type,
    transcription_factors,
    basis="pca",
    adj_key="cosine_transition_matrix",
    figsize=(16, 8),
    color_map="bwr",
    font_scale=0.8,
    scaler=0.6,
    save_path=None,
    show_plot=True,
    return_data=False
):
    """
    Plot kinetic heatmap showing gene expression dynamics along least action path
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    cells_indices_dict : dict
        Dictionary containing cell indices for each cell type
    source_cell_type : str
        Source cell type (e.g., 'HSC')
    target_cell_type : str
        Target cell type (e.g., 'Bas')
    transcription_factors : list
        List of transcription factor gene names
    basis : str, default="pca"
        Basis for least action path computation
    adj_key : str, default="cosine_transition_matrix"
        Adjacency matrix key
    figsize : tuple, default=(16, 8)
        Figure size for heatmap
    color_map : str, default="bwr" 
        Color map for heatmap
    font_scale : float, default=0.8
        Font scale for seaborn
    scaler : float, default=0.6
        Publication style scaler for dynamo
    save_path : str, optional
        Full path to save figure (e.g., 'HSC_to_Bas_kinetics.pdf')
    show_plot : bool, default=True
        Whether to display plot
    return_data : bool, default=False
        Whether to return LAP object and heatmap data
        
    Returns:
    --------
    If return_data=True:
        dict containing LAP object and heatmap data
    """
    
    # Set up cells
    init_cells = [adata.obs_names[cells_indices_dict[source_cell_type][0][0]]]
    target_cells = [adata.obs_names[cells_indices_dict[target_cell_type][0][0]]]
    
    print(f"Transition: {source_cell_type} → {target_cell_type}")
    print(f"Initial cells: {init_cells}")
    print(f"Target cells: {target_cells}")
    
    # Filter for transcription factors
    is_human_tfs = [gene in transcription_factors for gene in adata.var_names[adata.var.use_for_transition]]
    human_genes = adata.var_names[adata.var.use_for_transition][is_human_tfs]
    
    print(f"Using {len(human_genes)} transcription factors for analysis")
    
    # Set plotting style
    #dyn.configuration.set_pub_style(scaler=scaler)
    sns.set(font_scale=font_scale)
    
    # Compute LAP
    print(f"\nComputing least action path...")
    lap = least_action(
        adata,
        init_cells=init_cells,
        target_cells=target_cells,
        basis=basis,
        adj_key=adj_key,
    )
    
    # Plot heatmap
    print(f"Generating kinetic heatmap...")
    plt.figure(figsize=figsize)
    sns_heatmap = kinetic_heatmap(
        adata,
        basis=basis,
        mode="lap",
        figsize=figsize,
        genes=human_genes,
        project_back_to_high_dim=True,
        save_show_or_return="return",
        color_map=color_map,
        transpose=True,
        xticklabels=True,
        yticklabels=False,
    )
    
    # Customize plot
    plt.setp(sns_heatmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.title(f'{source_cell_type} → {target_cell_type}\nTranscription Factor Expression Kinetics', 
              fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
        
    
    print(f"Completed kinetic heatmap for {source_cell_type} → {target_cell_type}")
    
    if return_data:
        return {
            'lap': lap,
            'heatmap': sns_heatmap,
            'transcription_factors': human_genes
        }
    else:
        return sns_heatmap


def analyze_kinetic_genes(
    adata,
    cells_indices_dict,
    source_cell_type,
    target_cell_type,
    transcription_factors,
    top_genes=20,
    basis="pca",
    adj_key="cosine_transition_matrix"
):
    """
    Analyze top dynamic genes along the transition path
    
    Returns:
    --------
    analysis_df : pd.DataFrame
        DataFrame with gene rankings and TF information
    """
    
    # Get cells
    init_cells = [adata.obs_names[cells_indices_dict[source_cell_type][0][0]]]
    target_cells = [adata.obs_names[cells_indices_dict[target_cell_type][0][0]]]
    
    print(f"Analyzing gene dynamics for {source_cell_type} → {target_cell_type}")
    
    # Compute LAP
    lap = least_action(
        adata, init_cells=init_cells, target_cells=target_cells,
        basis=basis, adj_key=adj_key
    )
    
    # Gene trajectory analysis
    gtraj = GeneTrajectory(adata)
    gtraj.from_pca(lap.X, t=lap.t)
    gtraj.calc_msd()
    ranking = rank_genes(adata, "traj_msd")
    
    # Add TF information
    ranking["TF"] = [gene in transcription_factors for gene in ranking["all"]]
    
    # Get top TF genes
    tf_genes = ranking.query("TF == True").head(top_genes)
    
    print(f"Top {len(tf_genes)} transcription factors:")
    for i, (idx, row) in enumerate(tf_genes.iterrows(), 1):
        # Find the score column (usually first numeric column)
        score_value = None
        for col in tf_genes.columns:
            if col != 'all' and col != 'TF':
                try:
                    score_value = row[col]
                    if isinstance(score_value, (int, float)):
                        print(f"  {i}. {row['all']} (score: {score_value:.3f})")
                        break
                except:
                    continue
        if score_value is None:
            print(f"  {i}. {row['all']}")
    
    return ranking, tf_genes


def conversions_heatmap(
    action_df,
    figsize=(3,3),
    color_dict=None,
    cmap='magma'
):
    import marsilea as ma
    import marsilea.plotter as mp
    action_df = action_df.fillna(0)
    sc_color=[
     '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10', '#EF7B77', '#279AD7','#F0EEF0',
     '#EAEFC5', '#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
     '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

    if color_dict is None:
        color_dict=dict(zip(
            action_df.index,
            sc_color[:len(action_df.index)]
        ))
    m = ma.Heatmap(
        action_df.values, 
        cmap=cmap, 
        height=figsize[1], width=figsize[0],
        linewidth=0.5,
        linecolor="lightgray",
        label='Action',
    )
    m.group_rows(pd.DataFrame(action_df.index)[0], 
    order=action_df.index)
    
    
    # Create plotters
    chunk = mp.Chunk(action_df.index, rotation=0, 
                     align="center",padding=12,size=10)
    colors_left = mp.Colors(action_df.index, palette=color_dict)
    colors_top = mp.Colors(action_df.index, palette=color_dict)
    #label_markers = mp.Labels(markers)
    
    # Add to the heatmap
    m.add_left(colors_left, size=0.1, pad=0.1,legend=False)
    m.add_left(chunk)
    
    m.add_bottom(colors_top, size=0.1, pad=0.1,legend=False)
    
    text_mask=ma.plotter.TextMesh(action_df.fillna(0).values.round(2),
                                 color='white',size=8)
    m.add_layer(text_mask)
    #cell_count = mp.Numbers(t_df.fillna(0).values, color="#fac858", label="Cell Count")
    #m.add_left(mp.Labels('Source'))
    title_left = mp.Title(f'Source', align='center')
    m.add_left(title_left)
    title_top = mp.Title(f'Target', align='center')
    m.add_bottom(title_top)
    m.add_legends()
    m.render()
    return m


def get_cell_indices_by_name(adata, cell_names):
    """
    Helper function to get cell obs indices from cell names
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    cell_names : list or str
        Cell names to find indices for
        
    Returns:
    --------
    indices : list
        List of obs indices corresponding to the cell names
        
    Examples:
    ---------
    >>> indices = get_cell_indices_by_name(adata, ['CELL_001', 'CELL_002'])
    >>> manual_indices = {'HSC': indices[:1], 'Meg': indices[1:]}
    """
    if isinstance(cell_names, str):
        cell_names = [cell_names]
    
    indices = []
    for name in cell_names:
        if name in adata.obs_names:
            idx = adata.obs_names.get_loc(name)
            indices.append(idx)
        else:
            print(f"Warning: Cell name '{name}' not found in adata.obs_names")
    
    return indices


def find_cells_by_criteria(adata, cell_type, criteria_dict, top_n=1):
    """
    Helper function to find cells by specific criteria
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    cell_type : str
        Cell type to filter for
    criteria_dict : dict
        Dictionary of criteria to filter cells
        e.g., {'umap_ddhodge_potential': 'max'} or {'gene_expression': ('GATA1', 'min')}
    top_n : int, default=1
        Number of top cells to return
        
    Returns:
    --------
    indices : list
        List of obs indices of selected cells
        
    Examples:
    ---------
    >>> # Find HSC cell with highest potential
    >>> indices = find_cells_by_criteria(adata, 'HSC', {'umap_ddhodge_potential': 'max'})
    >>> 
    >>> # Find Meg cell with lowest GATA1 expression
    >>> indices = find_cells_by_criteria(adata, 'Meg', {'GATA1': 'min'})
    """
    # Filter for cell type
    cell_mask = adata.obs['cell_type'] == cell_type
    cell_indices = np.where(cell_mask)[0]
    
    if len(cell_indices) == 0:
        print(f"Warning: No cells found for type {cell_type}")
        return []
    
    # Apply criteria
    scores = np.ones(len(cell_indices))
    
    for criterion, direction in criteria_dict.items():
        if criterion in adata.obs.columns:
            values = adata.obs.loc[cell_mask, criterion].values
        elif criterion in adata.var_names:
            # Gene expression
            gene_idx = adata.var_names.get_loc(criterion)
            if hasattr(adata.X, 'toarray'):
                values = adata.X[cell_mask, gene_idx].toarray().flatten()
            else:
                values = adata.X[cell_mask, gene_idx]
        else:
            print(f"Warning: Criterion '{criterion}' not found")
            continue
        
        if direction == 'max':
            scores *= values
        elif direction == 'min':
            scores *= (1.0 / (values + 1e-6))
    
    # Get top cells
    top_indices = np.argsort(scores)[-top_n:][::-1]
    selected_indices = cell_indices[top_indices]
    
    return selected_indices.tolist()