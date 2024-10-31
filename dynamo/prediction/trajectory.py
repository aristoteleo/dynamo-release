from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy
from anndata import AnnData
from scipy.interpolate import interp1d

from ..dynamo_logger import LoggerManager
from ..tools.utils import flatten
from ..utils import expr_to_pca, pca_to_expr
from ..vectorfield.scVectorField import DifferentiableVectorField
from ..vectorfield.topography import dup_osc_idx_iter
from ..vectorfield.utils import angle, normalize_vectors


class Trajectory:
    """Base class for handling trajectory interpolation, resampling, etc."""

    def __init__(self, X: np.ndarray, t: Union[None, np.ndarray] = None, sort: bool = True) -> None:
        """Initializes a Trajectory object.

        Args:
            X: trajectory positions, shape (n_points, n_dimensions)
            t: trajectory times, shape (n_points,). Defaults to None.
            sort: whether to sort the time stamps. Defaults to True.
        """
        self.X = X
        if t is None:
            self.t = None
        else:
            self.set_time(t, sort=sort)

    def __len__(self) -> int:
        """Returns the number of points in the trajectory.

        Returns:
            number of points in the trajectory
        """
        return self.X.shape[0]

    def set_time(self, t: np.ndarray, sort: bool = True) -> None:
        """Set the time stamps for the trajectory. Sorts the time stamps if requested.

        Args:
            t: trajectory times, shape (n_points,)
            sort: whether to sort the time stamps. Defaults to True.
        """
        if sort:
            I = np.argsort(t)
            self.t = t[I]
            self.X = self.X[I]
        else:
            self.t = t

    def dim(self) -> int:
        """Returns the number of dimensions in the trajectory.

        Returns:
            number of dimensions in the trajectory
        """
        return self.X.shape[1]

    def calc_tangent(self, normalize: bool = True):
        """Calculate the tangent vectors of the trajectory.

        Args:
            normalize: whether to normalize the tangent vectors. Defaults to True.

        Returns:
            tangent vectors of the trajectory, shape (n_points-1, n_dimensions)
        """
        tvec = self.X[1:] - self.X[:-1]
        if normalize:
            tvec = normalize_vectors(tvec)
        return tvec

    def calc_arclength(self) -> float:
        """Calculate the arc length of the trajectory.

        Returns:
            arc length of the trajectory
        """
        tvec = self.calc_tangent(normalize=False)
        norms = np.linalg.norm(tvec, axis=1)
        return np.sum(norms)

    def calc_curvature(self) -> np.ndarray:
        """Calculate the curvature of the trajectory.

        Returns:
            curvature of the trajectory, shape (n_points,)
        """
        tvec = self.calc_tangent(normalize=False)
        kappa = np.zeros(self.X.shape[0])
        for i in range(1, self.X.shape[0] - 1):
            # ref: http://www.cs.jhu.edu/~misha/Fall09/1-curves.pdf (p. 55)
            kappa[i] = angle(tvec[i - 1], tvec[i]) / (np.linalg.norm(tvec[i - 1] / 2) + np.linalg.norm(tvec[i] / 2))
        return kappa

    def resample(self, n_points: int, tol: float = 1e-4, inplace: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the curve with the specified number of points.

        Args:
            n_points: An integer specifying the number of points in the resampled curve.
            tol: A float specifying the tolerance for removing redundant points. Default is 1e-4.
            inplace: A boolean flag indicating whether to modify the curve object in place. Default is True.

        Returns:
            A tuple containing the resampled curve coordinates and time values (if available).

        Raises:
            ValueError: If the specified number of points is less than 2.

        TODO:
            Decide whether the tol argument should be included or not during the code refactoring and optimization.
        """
        # remove redundant points
        """if tol is not None:
            X, arclen, discard = remove_redundant_points_trajectory(self.X, tol=tol, output_discard=True)
            if self.t is not None:
                t = np.array(self.t[~discard], copy=True)
            else:
                t = None
        else:
            X = np.array(self.X, copy=True)
            t = np.array(self.t, copy=True) if self.t is not None else None
            arclen = self.calc_arclength()"""

        # resample using the arclength sampling
        # ret = arclength_sampling(X, arclen / n_points, t=t)
        ret = arclength_sampling_n(self.X, n_points, t=self.t)
        X = ret[0]
        if self.t is not None:
            t = ret[2]

        if inplace:
            self.X, self.t = X, t

        return X, t

    def archlength_sampling(
        self,
        sol: scipy.integrate._ivp.common.OdeSolution,
        interpolation_num: int,
        integration_direction: str,
    ) -> None:
        """Sample the curve using archlength sampling.

        Args:
            sol: The ODE solution from scipy.integrate.solve_ivp.
            interpolation_num: The number of points to interpolate the curve at.
            integration_direction: The direction to integrate the curve in. Can be "forward", "backward", or "both".
        """
        tau, x = self.t, self.X.T
        idx = dup_osc_idx_iter(x, max_iter=100, tol=x.ptp(0).mean() / 1000)[0]

        # idx = dup_osc_idx_iter(x)
        x = x[:idx]
        _, arclen, _ = remove_redundant_points_trajectory(x, tol=1e-4, output_discard=True)
        cur_Y, alen, self.t = arclength_sampling_n(x, num=interpolation_num + 1, t=tau[:idx])
        self.t = self.t[1:]
        cur_Y = cur_Y[:, 1:]

        if integration_direction == "both":
            neg_t_len = sum(np.array(self.t) < 0)

        self.X = (
            sol(self.t)
            if integration_direction != "both"
            else np.hstack(
                (
                    sol[0](self.t[:neg_t_len]),
                    sol[1](self.t[neg_t_len:]),
                )
            )
        )

    def logspace_sampling(
        self,
        sol: scipy.integrate._ivp.common.OdeSolution,
        interpolation_num: int,
        integration_direction: str,
    ) -> None:
        """Sample the curve using logspace sampling.

        Args:
            sol: The ODE solution from scipy.integrate.solve_ivp.
            interpolation_num: The number of points to interpolate the curve at.
            integration_direction: The direction to integrate the curve in. Can be "forward", "backward", or "both".
        """
        tau, x = self.t, self.X.T
        neg_tau, pos_tau = tau[tau < 0], tau[tau >= 0]

        if len(neg_tau) > 0:
            t_0, t_1 = (
                -(
                    np.logspace(
                        0,
                        np.log10(abs(min(neg_tau)) + 1),
                        interpolation_num,
                    )
                )
                - 1,
                np.logspace(0, np.log10(max(pos_tau) + 1), interpolation_num) - 1,
            )
            self.t = np.hstack((t_0[::-1], t_1))
        else:
            self.t = np.logspace(0, np.log10(max(tau) + 1), interpolation_num) - 1

        if integration_direction == "both":
            neg_t_len = sum(np.array(self.t) < 0)

        self.X = (
            sol(self.t)
            if integration_direction != "both"
            else np.hstack(
                (
                    sol[0](self.t[:neg_t_len]),
                    sol[1](self.t[neg_t_len:]),
                )
            )
        )

    def interpolate(self, t: np.ndarray, **interp_kwargs) -> np.ndarray:
        """Interpolate the curve at new time values.

        Args:
            t: The new time values at which to interpolate the curve.
            **interp_kwargs: Additional arguments to pass to `scipy.interpolate.interp1d`.

        Returns:
            The interpolated values of the curve at the specified time values.

        Raises:
            Exception: If `self.t` is `None`, which is needed for interpolation.
        """
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return interp1d(self.t, self.X, axis=0, **interp_kwargs)(t)

    def interp_t(self, num: int = 100) -> np.ndarray:
        """Interpolates the `t` parameter linearly.

        Args:
            num: Number of interpolation points.

        Returns:
            The array of interpolated `t` values.
        """
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return np.linspace(self.t[0], self.t[-1], num=num)

    def interp_X(self, num: int = 100, **interp_kwargs) -> np.ndarray:
        """Interpolates the curve at `num` equally spaced points in `t`.

        Args:
            num: The number of points to interpolate the curve at.
            **interp_kwargs: Additional keyword arguments to pass to `scipy.interpolate.interp1d`.

        Returns:
            The interpolated curve at `num` equally spaced points in `t`.
        """
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return self.interpolate(self.interp_t(num=num), **interp_kwargs)

    def integrate(self, func: Callable) -> np.ndarray:
        """Calculate the integral of a function along the curve.

        Args:
            func: A function to integrate along the curve.

        Returns:
            The integral of func along the discrete curve.
        """
        F = np.zeros(func(self.X[0]).shape)
        tvec = self.calc_tangent(normalize=False)
        for i in range(1, self.X.shape[0] - 1):
            # ref: http://www.cs.jhu.edu/~misha/Fall09/1-curves.pdf P. 47
            F += func(self.X[i]) * (np.linalg.norm(tvec[i - 1]) + np.linalg.norm(tvec[i])) / 2
        return F

    def calc_msd(self, decomp_dim: bool = True, ref: int = 0) -> Union[float, np.ndarray]:
        """Calculate the mean squared displacement (MSD) of the curve with respect to a reference point.

        Args:
            decomp_dim: If True, return the MSD of each dimension separately. If False, return the total MSD.
            ref: Index of the reference point. Default is 0.

        Returns:
            The MSD of the curve with respect to the reference point.
        """
        S = (self.X - self.X[ref]) ** 2
        if decomp_dim:
            S = S.sum(axis=0)
        else:
            S = S.sum()
        S /= len(self)
        return S


class VectorFieldTrajectory(Trajectory):
    """Class for handling trajectory data with a differentiable vector field."""

    def __init__(self, X: np.ndarray, t: np.ndarray, vecfld: DifferentiableVectorField) -> None:
        """Initializes a VectorFieldTrajectory object.

        Args:
            X: The trajectory data as a numpy array of shape (n, d).
            t: The time data as a numpy array of shape (n,).
            vecfld: The differentiable vector field that describes the trajectory.
        """
        super().__init__(X, t=t)
        self.vecfld = vecfld
        self.data = {"velocity": None, "acceleration": None, "curvature": None, "divergence": None}
        self.Js = None

    def get_velocities(self) -> np.ndarray:
        """Calculates and returns the velocities along the trajectory.

        Returns:
            The velocity data as a numpy array of shape (n, d).
        """
        if self.data["velocity"] is None:
            self.data["velocity"] = self.vecfld.func(self.X)
        return self.data["velocity"]

    def get_jacobians(self, method: Optional[str] = None) -> np.ndarray:
        """Calculates and returns the Jacobians of the vector field along the trajectory.

        Args:
            method: The method used to compute the Jacobians.

        Returns:
            The Jacobian data as a numpy array of shape (n, d, d).
        """
        if self.Js is None:
            fjac = self.vecfld.get_Jacobian(method=method)
            self.Js = fjac(self.X)
        return self.Js

    def get_accelerations(self, method: Optional[str] = None, **kwargs) -> np.ndarray:
        """Calculates and returns the accelerations along the trajectory.

        Args:
            method: The method used to compute the Jacobians.
            **kwargs: Additional keyword arguments to be passed to the acceleration computation method.

        Returns:
            The acceleration data as a numpy array of shape (n, d).
        """
        if self.data["acceleration"] is None:
            if self.Js is None:
                self.Js = self.get_jacobians(method=method)
            self.data["acceleration"] = self.vecfld.compute_acceleration(self.X, Js=self.Js, **kwargs)
        return self.data["acceleration"]

    def get_curvatures(self, method: Optional[str] = None, **kwargs) -> np.ndarray:
        """Calculates and returns the curvatures along the trajectory.

        Args:
            method: The method used to compute the Jacobians.
            **kwargs: Additional keyword arguments to be passed to the curvature computation method.

        Returns:
            The curvature data as a numpy array of shape (n,).
        """
        if self.data["curvature"] is None:
            if self.Js is None:
                self.Js = self.get_jacobians(method=method)
            self.data["curvature"] = self.vecfld.compute_curvature(self.X, Js=self.Js, **kwargs)
        return self.data["curvature"]

    def get_divergences(self, method: Optional[str] = None, **kwargs) -> np.ndarray:
        """Calculates and returns the divergences along the trajectory.

        Args:
            method: The method used to compute the Jacobians.
            **kwargs: Additional keyword arguments to be passed to the divergence computation method.

        Returns:
            The divergence data as a numpy array of shape (n,).
        """
        if self.data["divergence"] is None:
            if self.Js is None:
                self.Js = self.get_jacobians(method=method)
            self.data["divergence"] = self.vecfld.compute_divergence(self.X, Js=self.Js, **kwargs)
        return self.data["divergence"]

    def calc_vector_msd(self, key: str, decomp_dim: bool = True, ref: int = 0) -> Union[np.ndarray, float]:
        """Calculate and return the mean squared displacement of a given vector field attribute in the trajectory.

        Args:
            key: The key for the vector field attribute in self.data to compute the mean squared displacement of.
            decomp_dim: Whether to decompose the MSD by dimension. Defaults to True.
            ref: The index of the reference point to use for computing the MSD. Defaults to 0.

        Returns:
            The mean squared displacement of the specified vector component in the trajectory.

        TODO:
            Discuss should we also calculate other quantities during the code refactoring and
                optimization phase (e.g. curl, hessian, laplacian, etc).

        """
        V = self.data[key]
        S = (V - V[ref]) ** 2
        if decomp_dim:
            S = S.sum(axis=0)
        else:
            S = S.sum()
        S /= len(self)
        return S


class GeneTrajectory(Trajectory):
    """Class for handling gene expression trajectory data."""

    def __init__(
        self,
        adata: AnnData,
        X: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        X_pca: Optional[np.ndarray] = None,
        PCs: str = "PCs",
        mean: str = "pca_mean",
        genes: str = "use_for_pca",
        expr_func: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Initializes a GeneTrajectory object.

        Args:
            adata: Anndata object containing the gene expression data.
            X: The gene expression data as a numpy array of shape (n, d). Defaults to None.
            t: The time data as a numpy array of shape (n,). Defaults to None.
            X_pca: The PCA-transformed gene expression data as a numpy array of shape (n, d). Defaults to None.
            PCs: The key in adata.uns to use for the PCA components. Defaults to "PCs".
            mean: The key in adata.uns to use for the PCA mean. Defaults to "pca_mean".
            genes: The key in adata.var to use for the genes. Defaults to "use_for_pca".
            expr_func: A function to transform the PCA-transformed gene expression data back to the original space.
                Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the superclass initializer.
        """
        self.adata = adata
        if type(PCs) is str:
            PCs = self.adata.uns[PCs]
        self.PCs = PCs

        if type(mean) is str:
            mean = self.adata.uns[mean]
        self.mean = mean

        self.expr_func = expr_func

        if type(genes) is str:
            genes = adata.var_names[adata.var[genes]].to_list()
        self.genes = np.array(genes)

        if X_pca is not None:
            self.from_pca(X_pca, t=t, **kwargs)

        if X is not None:
            super().__init__(X, t=t)

    def from_pca(self, X_pca: np.ndarray, t: Optional[np.ndarray] = None) -> None:
        """Converts PCA-transformed gene expression data to gene expression data.

        Args:
            X_pca: The PCA-transformed gene expression data as a numpy array of shape (n, d).
            t: The time data as a numpy array of shape (n,). Defaults to None.
        """
        X = pca_to_expr(X_pca, self.PCs, mean=self.mean, func=self.expr_func)
        super().__init__(X, t=t)

    def to_pca(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """Converts gene expression data to PCA-transformed gene expression data.

        Args:
            x: The gene expression data as a numpy array of shape (n, d). Defaults to None.

        Returns:
            The PCA-transformed gene expression data as a numpy array of shape (n, d).
        """
        if x is None:
            x = self.X
        return expr_to_pca(x, self.PCs, mean=self.mean, func=self.expr_func)

    def genes_to_mask(self) -> np.ndarray:
        """Returns a boolean mask for the genes in the trajectory.

        Returns:
            A boolean mask for the genes in the trajectory.
        """
        mask = np.zeros(self.adata.n_vars, dtype=np.bool_)
        for g in self.genes:
            mask[self.adata.var_names == g] = True
        return mask

    def calc_msd(self, save_key: str = "traj_msd", **kwargs) -> Union[float, np.ndarray]:
        """Calculate the mean squared displacement (MSD) of the gene expression trajectory.

        Args:
            save_key: The key to save the MSD data to in adata.var. Defaults to "traj_msd".
            **kwargs: Additional keyword arguments to be passed to the superclass method.

        Returns:
            The mean squared displacement of the gene expression trajectory.
        """
        msd = super().calc_msd(**kwargs)

        LoggerManager.main_logger.info_insert_adata(save_key, "var")
        self.adata.var[save_key] = np.ones(self.adata.n_vars) * np.nan
        self.adata.var[save_key][self.genes_to_mask()] = msd

        return msd

    def save(self, save_key: str = "gene_trajectory") -> None:
        """Save the gene expression trajectory to adata.var.

        Args:
            save_key: The key to save the gene expression trajectory to in adata.var. Defaults to "gene_trajectory".
        """
        LoggerManager.main_logger.info_insert_adata(save_key, "varm")
        self.adata.varm[save_key] = np.ones((self.adata.n_vars, self.X.shape[0])) * np.nan
        self.adata.varm[save_key][self.genes_to_mask(), :] = self.X.T

    def select_gene(
        self,
        genes: Union[np.ndarray, list],
        arr: Optional[np.ndarray] = None,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        """Selects the gene expression data for the specified genes.

        Args:
            genes: The genes to select the expression data for.
            arr: The array to select the genes from. Defaults to None.
            axis: The axis to select the genes along. Defaults to None.

        Returns:
            The gene expression data for the specified genes.
        """
        if arr is None:
            arr = self.X
        if arr.ndim == 1:
            axis = 0
        else:
            if axis is None:
                axis = 1
        y = []
        if self.genes is not None:
            for g in genes:
                if g not in self.genes:
                    LoggerManager.main_logger.warning(f"{g} is not in `self.genes`.")
                else:
                    if axis == 0:
                        y.append(flatten(arr[self.genes == g]))
                    elif axis == 1:
                        y.append(flatten(arr[:, self.genes == g]))
        else:
            raise Exception("Cannot select genes since `self.genes` is `None`.")

        return np.array(y)


def arclength_sampling_n(
    X: np.ndarray,
    num: int,
    t: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """Uniformly sample data points on an arc curve that generated from vector field predictions.

    Args:
        X: The data points to sample from.
        num: The number of points to sample.
        t: The time values for the data points. Defaults to None.

    Returns:
        The sampled data points and the arc length of the curve.
    """
    arclen = np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))
    arclen = np.hstack((0, arclen))

    z = np.linspace(arclen[0], arclen[-1], num)
    X_ = interp1d(arclen, X, axis=0)(z)
    if t is not None:
        t_ = interp1d(arclen, t)(z)
        return X_, arclen[-1], t_
    else:
        return X_, arclen[-1]


def remove_redundant_points_trajectory(
    X: np.ndarray,
    tol: float = 1e-4,
    output_discard: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """Remove consecutive data points that are too close to each other.

    Args:
        X: The data points to remove redundant points from.
        tol: The tolerance for removing redundant points. Defaults to 1e-4.
        output_discard: Whether to output the discarded points. Defaults to False.

    Returns:
        The data points with redundant points removed and the arc length of the curve.
    """
    X = np.atleast_2d(X)
    discard = np.zeros(len(X), dtype=bool)
    if X.shape[0] > 1:
        for i in range(len(X) - 1):
            dist = np.linalg.norm(X[i + 1] - X[i])
            if dist < tol:
                discard[i + 1] = True
        X = X[~discard]

    arclength = 0

    x0 = X[0]
    for i in range(1, len(X)):
        tangent = X[i] - x0 if i == 1 else X[i] - X[i - 1]
        d = np.linalg.norm(tangent)

        arclength += d

    if output_discard:
        return (X, arclength, discard)
    else:
        return (X, arclength)


def arclength_sampling(X: np.ndarray, step_length: float, n_steps: int, t: Optional[np.ndarray] = None) -> np.ndarray:
    """Uniformly sample data points on an arc curve that generated from vector field predictions.

    Args:
        X: The data points to sample from.
        step_length: The length of each step.
        n_steps: The number of steps to sample.
        t: The time values for the data points. Defaults to None.

    Returns:
        The sampled data points and the arc length of the curve.
    """
    Y = []
    x0 = X[0]
    T = [] if t is not None else None
    t0 = t[0] if t is not None else None
    i = 1
    terminate = False
    arclength = 0

    def _calculate_new_point():
        x = x0 if j == i else X[j - 1]
        cur_y = x + (step_length - L) * tangent / d

        if t is not None:
            cur_tau = t0 if j == i else t[j - 1]
            cur_tau += (step_length - L) / d * (t[j] - cur_tau)
            T.append(cur_tau)
        else:
            cur_tau = None

        Y.append(cur_y)

        return cur_y, cur_tau

    while i < len(X) - 1 and not terminate:
        L = 0
        for j in range(i, len(X)):
            tangent = X[j] - x0 if j == i else X[j] - X[j - 1]
            d = np.linalg.norm(tangent)
            if L + d >= step_length:
                y, tau = _calculate_new_point()
                t0 = tau if t is not None else None
                x0 = y
                i = j
                break
            else:
                L += d
        if j == len(X) - 1:
            i += 1
        arclength += step_length
        if L + d < step_length:
            terminate = True

    if len(Y) < n_steps:
        _, _ = _calculate_new_point()

    if T is not None:
        return np.array(Y), arclength, T
    else:
        return np.array(Y), arclength
