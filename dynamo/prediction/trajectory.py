import numpy as np
from scipy.interpolate import interp1d
from ..vectorfield.utils import normalize_vectors, angle
from ..vectorfield.scVectorField import DifferentiableVectorField
from .utils import arclength_sampling_n, remove_redundant_points_trajectory, pca_to_expr, expr_to_pca
from ..tools.utils import flatten
from ..dynamo_logger import LoggerManager


class Trajectory:
    def __init__(self, X, t=None) -> None:
        """
        Base class for handling trajectory interpolation, resampling, etc.
        """
        self.X = X
        self.t = t

    def __len__(self):
        return self.X.shape[0]

    def dim(self):
        return self.X.shape[1]

    def calc_tangent(self, normalize=True):
        tvec = self.X[1:] - self.X[:-1]
        if normalize:
            tvec = normalize_vectors(tvec)
        return tvec

    def calc_arclength(self):
        tvec = self.calc_tangent(normalize=False)
        norms = np.linalg.norm(tvec, axis=1)
        return np.sum(norms)

    def calc_curvature(self):
        tvec = self.calc_tangent(normalize=False)
        kappa = np.zeros(self.X.shape[0])
        for i in range(1, self.X.shape[0] - 1):
            # ref: http://www.cs.jhu.edu/~misha/Fall09/1-curves.pdf (p. 55)
            kappa[i] = angle(tvec[i - 1], tvec[i]) / (np.linalg.norm(tvec[i - 1] / 2) + np.linalg.norm(tvec[i] / 2))
        return kappa

    def resample(self, n_points, tol=1e-4, inplace=True):
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

    def interpolate(self, t, **interp_kwargs):
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return interp1d(self.t, self.X, axis=0, **interp_kwargs)(t)

    def interp_t(self, num=100):
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return np.linspace(self.t[0], self.t[-1], num=num)

    def interp_X(self, num=100, **interp_kwargs):
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return self.interpolate(self.interp_t(num=num), **interp_kwargs)

    def integrate(self, func):
        """ Calculate the integral of func along the curve. The first and last points are omitted. """
        F = np.zeros(func(self.X[0]).shape)
        tvec = self.calc_tangent(normalize=False)
        for i in range(1, self.X.shape[0] - 1):
            # ref: http://www.cs.jhu.edu/~misha/Fall09/1-curves.pdf P. 47
            F += func(self.X[i]) * (np.linalg.norm(tvec[i - 1]) + np.linalg.norm(tvec[i])) / 2
        return F

    def calc_msd(self, decomp_dim=True, ref=0):
        S = (self.X - self.X[ref]) ** 2
        if decomp_dim:
            S = S.sum(axis=0)
        else:
            S = S.sum()
        S /= len(self)
        return S


class VectorFieldTrajectory(Trajectory):
    def __init__(self, X, t, vecfld: DifferentiableVectorField) -> None:
        super().__init__(X, t=t)
        self.vecfld = vecfld
        self.data = {"velocity": None, "acceleration": None, "curvature": None, "divergence": None}
        self.Js = None

    def get_velocities(self):
        if self.data["velocity"] is None:
            self.data["velocity"] = self.vecfld.func(self.X)
        return self.data["velocity"]

    def get_jacobians(self, method=None):
        if self.Js is None:
            fjac = self.vecfld.get_Jacobian(method=method)
            self.Js = fjac(self.X)
        return self.Js

    def get_acclerations(self, method=None, **kwargs):
        if self.data["acceleration"] is None:
            if self.Js is None:
                self.Js = self.get_jacobians(method=method)
            self.data["acceleration"] = self.vecfld.compute_acceleration(self.X, Js=self.Js, **kwargs)
        return self.data["acceleration"]

    def get_curvatures(self, method=None, **kwargs):
        if self.data["curvature"] is None:
            if self.Js is None:
                self.Js = self.get_jacobians(method=method)
            self.data["curvature"] = self.vecfld.compute_curvature(self.X, Js=self.Js, **kwargs)
        return self.data["curvature"]

    def get_divergences(self, method=None, **kwargs):
        if self.data["divergence"] is None:
            if self.Js is None:
                self.Js = self.get_jacobians(method=method)
            self.data["divergence"] = self.vecfld.compute_divergence(self.X, Js=self.Js, **kwargs)
        return self.data["divergence"]

    def calc_vector_msd(self, key, decomp_dim=True, ref=0):
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
        self, adata, X=None, t=None, X_pca=None, PCs="PCs", mean="pca_mean", genes="use_for_pca", **kwargs
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

        if type(genes) is str:
            genes = adata.var_names[adata.var[genes]].to_list()
        self.genes = np.array(genes)

        if X_pca is not None:
            self.from_pca(X_pca, **kwargs)

        if X is not None:
            super().__init__(X, t=t)

    def from_pca(self, X_pca, func=None, t=None):
        X = pca_to_expr(X_pca, self.PCs, mean=self.mean, func=func)
        super().__init__(X, t=t)

    def to_pca(self, func=None):
        return expr_to_pca(self.X, self.PCs, mean=self.mean, func=func)

    def genes_to_mask(self):
        mask = np.zeros(self.adata.n_vars, dtype=np.bool_)
        for g in self.genes:
            mask[self.adata.var_names == g] = True
        return mask

    def calc_msd(self, save_key="lap_msd", **kwargs):
        msd = super().calc_msd(**kwargs)

        LoggerManager.main_logger.info_insert_adata(save_key, "var")
        self.adata.var[save_key] = np.ones(self.adata.n_vars) * np.nan
        self.adata.var[save_key][self.genes_to_mask()] = msd

        return msd

    def save(self, save_key="gene_trajectory"):
        LoggerManager.main_logger.info_insert_adata(save_key, "varm")
        self.adata.varm[save_key] = np.ones((self.adata.n_vars, self.X.shape[0])) * np.nan
        self.adata.varm[save_key][self.genes_to_mask(), :] = self.X.T

    def select_gene(self, genes):
        y = []
        if self.genes is not None:
            for g in genes:
                if g not in self.genes:
                    LoggerManager.main_logger.warning(f"{g} is not in `self.genes`.")
                else:
                    y.append(flatten(self.X[:, self.genes == g]))
        else:
            raise Exception("Cannot select genes since `self.genes` is `None`.")

        return np.array(y)
