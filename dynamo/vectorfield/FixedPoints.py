import numpy as np
from scipy.linalg import eig


class FixedPoints:
    def __init__(self, X=None, J=None):
        self.X = X if X is not None else []
        self.J = J if J is not None else []
        self.eigvals = []

    def get_X(self):
        return np.array(self.X)

    def get_J(self):
        return np.array(self.J)

    def add_fixed_points(self, X, J, tol_redundant=1e-4):
        for i, x in enumerate(X):
            redundant = False
            if tol_redundant is not None and len(self.X) > 0:
                for y in self.X:
                    if np.linalg.norm(x - y) <= tol_redundant:
                        redundant = True
            if not redundant:
                self.X.append(x)
                self.J.append(J[i])

    def compute_eigvals(self):
        self.eigvals = []
        for i in range(len(self.J)):
            if self.J[i] is None or np.isnan(self.J[i]).any():
                w = np.nan
            else:
                w, _ = eig(self.J[i])
            self.eigvals.append(w)

    def is_stable(self):
        if len(self.eigvals) != len(self.X):
            self.compute_eigvals()

        stable = np.ones(len(self.eigvals), dtype=bool)
        for i, w in enumerate(self.eigvals):
            if w is None or np.isnan(w).any():
                stable[i] = np.nan
            else:
                if np.any(np.real(w) >= 0):
                    stable[i] = False
        return stable

    def is_saddle(self):
        is_stable = self.is_stable()
        saddle = np.zeros(len(self.eigvals), dtype=bool)
        for i, w in enumerate(self.eigvals):
            if w is None or np.isnan(w).any():
                saddle[i] = np.nan
            else:
                if not is_stable[i] and np.any(np.real(w) < 0):
                    saddle[i] = True
        return saddle, is_stable

    def get_fixed_point_types(self):
        is_saddle, is_stable = self.is_saddle()
        # -1 -- stable, 0 -- saddle, 1 -- unstable
        ftype = np.ones(len(self.X))
        for i in range(len(ftype)):
            if self.X[i] is None or np.isnan((self.X[i])).any():
                ftype[i] = np.nan
            else:
                if is_saddle[i]:
                    ftype[i] = 0
                elif is_stable[i]:
                    ftype[i] = -1
        return ftype
