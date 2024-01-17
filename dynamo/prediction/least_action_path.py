from typing import Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from anndata import AnnData
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from ..dynamo_logger import LoggerManager
from ..tools.utils import fetch_states, nearest_neighbors
from ..utils import pca_to_expr
from ..vectorfield import SvcVectorField
from ..vectorfield.utils import (
    vecfld_from_adata,
    vector_field_function_transformation,
    vector_transformation,
)
from .trajectory import GeneTrajectory, Trajectory
from .utils import arclength_sampling_n, find_elbow


class LeastActionPath(Trajectory):
    """
    A class for computing the Least Action Path for a given function and initial conditions.

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
        """
        Initializes the LeastActionPath class instance with the given initial conditions, vector field function,
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
        """
        Returns the time points of the trajectory.

        Returns:
            ndarray: The time points of the trajectory.
        """
        return self.t

    def get_dt(self) -> float:
        """
        Returns the time step of the trajectory.

        Returns:
            float: The time step of the trajectory.
        """
        return np.mean(np.diff(self.t))

    def action_t(self, t: Optional[float] = None, **interp_kwargs) -> np.ndarray:
        """
        Returns the Least Action Path action values at time t.

        Args:
            t: The time point(s) to return the action value(s) for.
                                 If None, returns the action values for all time points.
                                 Defaults to None.
            **interp_kwargs: Additional keyword arguments to pass to the interp1d function.

        Returns:
            ndarray: The Least Action Path action value(s).
        """
        if t is None:
            return self._action
        else:
            return interp1d(self.t, self._action, **interp_kwargs)(t)

    def mfpt(self, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Eqn. 7 of Epigenetics as a first exit problem."""
        action = self._action if action is None else action
        return 1 / np.exp(-action)

    def optimize_dt(self) -> float:
        """Optimizes the time step of the simulation to minimize the Least Action Path action.

        Returns:
            float: optimal time step
        """
        dt_0 = self.get_dt()
        t_dict = minimize(lambda t: action(self.X, self.func, D=self.D, dt=t), dt_0)
        LoggerManager.main_logger.info(t_dict["message"])
        LoggerManager.main_logger.info("optimal action: %f" % t_dict["fun"])
        dt_sol = t_dict["x"][0]
        self.t = np.arange(self.X.shape[0]) * dt_sol
        return dt_sol


class GeneLeastActionPath(GeneTrajectory):
    def __init__(self, adata, lap: LeastActionPath = None, X_pca=None, vf_func=None, D=1, dt=1, **kwargs) -> None:
        """
        Calculates the least action path trajectory and action for a gene expression dataset.
        Inherits from GeneTrajectory class.

        Args:
            adata: AnnData object containing the gene expression dataset.
            lap: LeastActionPath object. Defaults to None.
            X_pca: PCA transformed expression data. Defaults to None.
            vf_func: Vector field function. Defaults to None.
            D: Diffusivity value. Defaults to 1.
            dt: Time step size. Defaults to 1.
            **kwargs: Additional keyword arguments passed to the GeneTrajectory class.

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
        """
        Returns the array of time values.

        Returns:
            np.ndarray: Array of time values.
        """
        return self.t

    def get_dt(self) -> float:
        """
        Returns the average time step size.

        Returns:
            float: Average time step size.
        """
        return np.mean(np.diff(self.t))

    def genewise_action(self) -> np.ndarray:
        """
        Calculates the genewise action values.

        Returns:
            np.ndarray: Array of genewise action values.
        """
        dt = self.get_dt()
        x = (self.X[:-1] + self.X[1:]) * 0.5
        v = np.diff(self.X, axis=0) / dt

        s = v - self.func(x)
        s = 0.5 * np.sum(s * s, axis=0) * dt / self.D

        return s

    def select_genewise_action(self, genes: Union[str, List[str]]) -> np.ndarray:
        """
        Returns the genewise action values for the specified genes.

        Args:
            genes (Union[str, List[str]]): List of gene names or a single gene name.

        Returns:
            np.ndarray: Array of genewise action values.
        """
        return super().select_gene(genes, arr=self.action)


def action(path: np.ndarray, vf_func: Callable[[np.ndarray], np.ndarray], D: float = 1, dt: float = 1) -> float:
    # centers
    """The action function calculates the action (or functional) of a path in space, given a velocity field function and diffusion coefficient. The path is represented as an array of points in space, and the velocity field is given by vf_func.

    The function first calculates the centers of the segments between each point in the path, and then calculates the velocity at each of these centers by taking the average of the velocities at the two neighboring points. The difference between the actual velocity and the velocity field at each center is then calculated and flattened into a one-dimensional array.

    The action is then calculated by taking the dot product of this array with itself, multiplying by a factor of 0.5*dt/D, where dt is the time step used to define the path, and D is the diffusion coefficient.

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


def action_aux(path_flatten, vf_func, dim, start=None, end=None, **kwargs):
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action(path, vf_func, **kwargs)


def action_grad(path: np.ndarray, vf_func: Callable, jac_func: Callable, D: float = 1.0, dt: float = 1.0) -> np.ndarray:
    """
    Computes the gradient of the action functional with respect to the path.

    Args:
        path: A 2D array of shape (n+1,d) representing the path, where n is the number of time steps and d is the dimension of the path.
        vf_func: A function that computes the velocity field vf(x) for a given position x.
        jac_func: A function that computes the Jacobian matrix of the velocity field at a given position.
        D: The diffusion constant (default is 1).
        dt: The time step (default is 1).

    Returns:
        np.ndarray: The gradient of the action functional with respect to the path, as a 2D array of shape (n,d).
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


def action_grad_aux(path_flatten, vf_func, jac_func, dim, start=None, end=None, **kwargs):
    path = reshape_path(path_flatten, dim, start=start, end=end)
    return action_grad(path, vf_func, jac_func, **kwargs).flatten()


def reshape_path(path_flatten, dim, start=None, end=None):
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


def minimize_lap_time(path_0, t0, t_min, vf_func, jac_func, D=1, num_t=20, elbow_method="hessian", hes_tol=3):
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


def get_init_path(G, start, end, coords, interpolation_num=20):
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
    for (init_state, target_state) in LoggerManager.progress_logger(
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
