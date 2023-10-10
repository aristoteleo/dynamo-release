from typing import Callable, List, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from ..dynamo_logger import LoggerManager
from ..tools.utils import flatten
from ..utils import expr_to_pca, pca_to_expr
from ..vectorfield.scVectorField import DifferentiableVectorField
from ..vectorfield.utils import angle, normalize_vectors
from .utils import arclength_sampling_n


class Trajectory:
    def __init__(self, X: np.ndarray, t: Union[None, np.ndarray] = None, sort: bool = True) -> None:
        """
        Base class for handling trajectory interpolation, resampling, etc.

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
        """
        Set the time stamps for the trajectory. Sorts the time stamps if requested.

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
        """
        Returns the number of dimensions in the trajectory.

        Returns:
            number of dimensions in the trajectory
        """
        return self.X.shape[1]

    def calc_tangent(self, normalize: bool = True):
        """
        Calculate the tangent vectors of the trajectory.

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
        """
        Calculate the arc length of the trajectory.

        Returns:
            arc length of the trajectory
        """
        tvec = self.calc_tangent(normalize=False)
        norms = np.linalg.norm(tvec, axis=1)
        return np.sum(norms)

    def calc_curvature(self) -> np.ndarray:
        """
        Calculate the curvature of the trajectory.

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
        """
        Resample the curve with the specified number of points.

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
        """
        Interpolates the curve at `num` equally spaced points in `t`.

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
    def __init__(self, X: np.ndarray, t: np.ndarray, vecfld: DifferentiableVectorField) -> None:
        """
        Initializes a VectorFieldTrajectory object.

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
        """
        Calculates and returns the velocities along the trajectory.

        Returns:
            The velocity data as a numpy array of shape (n, d).
        """
        if self.data["velocity"] is None:
            self.data["velocity"] = self.vecfld.func(self.X)
        return self.data["velocity"]

    def get_jacobians(self, method=None) -> np.ndarray:
        """
        Calculates and returns the Jacobians of the vector field along the trajectory.

        Args:
            method: The method used to compute the Jacobians.

        Returns:
            The Jacobian data as a numpy array of shape (n, d, d).
        """
        if self.Js is None:
            fjac = self.vecfld.get_Jacobian(method=method)
            self.Js = fjac(self.X)
        return self.Js

    def get_accelerations(self, method=None, **kwargs) -> np.ndarray:
        """
        Calculates and returns the accelerations along the trajectory.

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

    def get_curvatures(self, method=None, **kwargs) -> np.ndarray:
        """
        Calculates and returns the curvatures along the trajectory.

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

    def get_divergences(self, method=None, **kwargs) -> np.ndarray:
        """
        Calculates and returns the divergences along the trajectory.

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
    def __init__(
        self,
        adata,
        X=None,
        t=None,
        X_pca=None,
        PCs="PCs",
        mean="pca_mean",
        genes="use_for_pca",
        expr_func=None,
        **kwargs,
    ) -> None:
        """
        This class is not fully functional yet.
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

    def from_pca(self, X_pca, t=None):
        X = pca_to_expr(X_pca, self.PCs, mean=self.mean, func=self.expr_func)
        super().__init__(X, t=t)

    def to_pca(self, x=None):
        if x is None:
            x = self.X
        return expr_to_pca(x, self.PCs, mean=self.mean, func=self.expr_func)

    def genes_to_mask(self):
        mask = np.zeros(self.adata.n_vars, dtype=np.bool_)
        for g in self.genes:
            mask[self.adata.var_names == g] = True
        return mask

    def calc_msd(self, save_key="traj_msd", **kwargs):
        msd = super().calc_msd(**kwargs)

        LoggerManager.main_logger.info_insert_adata(save_key, "var")
        self.adata.var[save_key] = np.ones(self.adata.n_vars) * np.nan
        self.adata.var[save_key][self.genes_to_mask()] = msd

        return msd

    def save(self, save_key="gene_trajectory"):
        LoggerManager.main_logger.info_insert_adata(save_key, "varm")
        self.adata.varm[save_key] = np.ones((self.adata.n_vars, self.X.shape[0])) * np.nan
        self.adata.varm[save_key][self.genes_to_mask(), :] = self.X.T

    def select_gene(self, genes, arr=None, axis=None):
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
