import numpy as np
from scipy.interpolate import interp1d
from ..vectorfield.utils import normalize_vectors, angle
from .utils import arclength_sampling, remove_redundant_points_trajectory


class Trajectory:
    def __init__(self, X, t=None) -> None:
        """
        Base class for handling trajectory interpolation, resampling, etc.
        """
        self.X = X
        self.t = t

    def __len__(self):
        return self.X.shape[0]

    def __dim__(self):
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

    def resample(self, n_points, tol=1e-4, overwrite=True):
        # remove redundant points
        if tol is not None:
            X, arclen, discard = remove_redundant_points_trajectory(self.X, tol=tol, output_discard=True)
            if self.t is not None:
                t = np.array(self.t[~discard], copy=True)
            else:
                t = None
        else:
            X = np.array(self.X, copy=True)
            t = np.array(self.t, copy=True) if self.t is not None else None
            arclen = self.calc_arclength()

        # resample using the arclength sampling
        ret = arclength_sampling(X, arclen / n_points, t=t)
        X = ret[0]
        if t is not None:
            t = ret[2]

        if overwrite:
            self.X, self.t = X, t

        return X, t

    def interpolate(self, t, **interp_kwargs):
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return interp1d(self.t, self.X, axis=0, **interp_kwargs)(t)

    def integrate(self, func):
        """ Calculate the integral of func along the curve. The first and last points are omitted. """
        F = np.zeros(func(self.X[0]).shape)
        tvec = self.calc_tangent(normalize=False)
        for i in range(1, self.X.shape[0] - 1):
            # ref: http://www.cs.jhu.edu/~misha/Fall09/1-curves.pdf P. 47
            F += func(self.X[i]) * (np.linalg.norm(tvec[i - 1]) + np.linalg.norm(tvec[i])) / 2
        return F
