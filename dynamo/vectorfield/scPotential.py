from typing import Callable, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import scipy as sp
import scipy.optimize
from anndata._core.anndata import AnnData

from ..tools.sampling import lhsclassic
from .Ao import Ao_pot_map, construct_Ao_potential_grid
from .Bhattacharya import alignment, path_integral
from .Tang import Tang_action
from .topography import FixedPoints
from .utils import is_outside_domain, vecfld_from_adata
from .Wang import Wang_action, Wang_LAP


def Potential(
    adata: AnnData,
    basis: str = "umap",
    x_lim: List = [0, 40],
    y_lim: List = [0, 40],
    DiffMat: Optional[Callable] = None,
    method: str = "Ao",
    **kwargs
) -> AnnData:
    """Function to map out the pseudo-potential landscape.

    Although it is appealing to define “potential” for biological systems as it is intuitive and familiar from other
    fields, it is well-known that the definition of a potential function in open biological systems is controversial
    (Ping Ao 2009). In the conservative system, the negative gradient of potential function is relevant to the velocity
    vector by ma = −Δψ (where m, a, are the mass and acceleration of the object, respectively). However, a biological
    system is massless, open and nonconservative, thus methods that directly learn potential function assuming a gradient
    system are not directly applicable. In 2004, Ao first proposed a framework that decomposes stochastic differential
    equations into either the gradient or the dissipative part and uses the gradient part to define a physical equivalent
    of potential in biological systems (P. Ao 2004). Later, various theoretical studies have been conducted towards
    this very goal (Xing 2010; Wang et al. 2011; J. X. Zhou et al. 2012; Qian 2013; P. Zhou and Li 2016). Bhattacharya
    and others also recently provided a numeric algorithm to approximate the potential landscape.

    This function implements the Ao, Bhattacharya method and Ying method and will also support other methods shortly.

    Args:
        adata: AnnData object that contains embedding and velocity data.
        basis: the basis of vector field function.
        x_lim: lower or upper limit of x-axis.
        y_lim: lower or upper limit of y-axis.
        DiffMat: The function which returns the diffusion matrix which can variable (for example, gene) dependent.
        method: Method to map the potential landscape.
        kwargs: Additional parameters for the method.

    Returns:
        The `AnnData` object that is updated with the `Pot` dictionary in the `uns` attribute.

    """

    _, Function = vecfld_from_adata(adata, basis=basis)
    DiffMat = DiffusionMatrix if DiffMat is None else DiffMat
    pot = Pot(Function, DiffMat, **kwargs)
    pot.fit(adata=adata, x_lim=x_lim, y_lim=y_lim, method=method)

    return adata


class Pot:
    def __init__(
        self,
        Function: Callable = None,
        DiffMat: Callable = None,
        boundary: List = None,
        n_points: int = 25,
        fixed_point_only: bool = False,
        find_fixed_points: bool = False,
        refpoint: Optional[np.ndarray] = None,
        stable: Optional[np.ndarray] = None,
        saddle: Optional[np.ndarray] = None,
    ):
        """It implements the least action method to calculate the potential values of fixed points for a given SDE
        (stochastic differential equation) model. The function requires the vector field function and a diffusion
        matrix. This code is based on the MATLAB code from Ruoshi Yuan and Ying Tang. Potential landscape of high
        dimensional nonlinear stochastic dynamics with large noise. Y Tang, R Yuan, G Wang, X Zhu, P Ao - Scientific
        reports, 2017

        Args:
            Function: The (reconstructed) vector field function.
            DiffMat: The function that returns the diffusion matrix which can variable (for example, gene) dependent.
            boundary: The range of variables (genes).
            n_points: The number of points along the least action path.
            fixed_point_only: The logic flag to determine whether only the potential
                for fixed point or entire space should be mapped.
            find_fixed_points: The logic flag to determine whether only the gen_fixed_points function
                should be run to identify fixed points.
            refpoint: The reference point to define the potential.
            stable: The matrix for storing the coordinates (gene expression configuration)
                of the stable fixed point (characteristic state of a particular cell type).
            saddle: The matrix for storing the coordinates (gene expression configuration)
                of the unstable fixed point (characteristic state of cells prime to bifurcation).
        """

        self.VecFld = {
            "Function": Function,
            "DiffusionMatrix": DiffMat,
        }  # should we use annadata here?

        self.parameters = {
            "boundary": boundary,
            "n_points": n_points,
            "fixed_point_only": fixed_point_only,
            "find_fixed_points": find_fixed_points,
            "refpoint": refpoint,
            "stable": stable,
            "saddle": saddle,
        }

    def fit(
        self,
        adata: AnnData,
        x_lim: List,
        y_lim: List,
        basis: str = "umap",
        method: str = "Ao",
        xyGridSpacing: int = 2,
        dt: float = 1e-2,
        tol: float = 1e-2,
        numTimeSteps: int = 1400,
    ) -> AnnData:
        """Function to map out the pseudo-potential landscape.

        Although it is appealing to define “potential” for biological systems as it is intuitive and familiar from other
        fields, it is well-known that the definition of a potential function in open biological systems is controversial
        (Ping Ao 2009). In the conservative system, the negative gradient of potential function is relevant to the velocity
        vector by ma = −Δψ (where m, a, are the mass and acceleration of the object, respectively). However, a biological
        system is massless, open and nonconservative, thus methods that directly learn potential function assuming a gradient
        system are not directly applicable. In 2004, Ao first proposed a framework that decomposes stochastic differential
        equations into either the gradient or the dissipative part and uses the gradient part to define a physical equivalent
        of potential in biological systems (P. Ao 2004). Later, various theoretical studies have been conducted towards
        this very goal (Xing 2010; Wang et al. 2011; J. X. Zhou et al. 2012; Qian 2013; P. Zhou and Li 2016). Bhattacharya
        and others also recently provided a numeric algorithm to approximate the potential landscape.

        This function implements the Ao, Bhattacharya method and Ying method and will also support other methods shortly.

        Args:
            adata: AnnData object that contains U_grid and V_grid data.
            x_lim: Lower or upper limit of x-axis.
            y_lim: Lower or upper limit of y-axis.
            basis: The dimension reduction method to use.
            method: Method used to map the pseudo-potential landscape. By default, it is Bhattacharya (A deterministic map of
                Waddington’s epigenetic landscape for cell fate specification. Sudin Bhattacharya, Qiang Zhang and Melvin
                E. Andersen). Other methods will be supported include: Tang (), Ping (), Wang (), Zhou ().
            xyGridSpacing: Grid spacing for "starting points" for each "path" on the potential surface
            dt: Time step for the path integral.
            tol: Tolerance to test for convergence.
            numTimeSteps: A high-enough number for convergence with given dt.

        Returns:
            The AnnData object updated with the following values:
                for all methods:
                    Xgrid: The X grid to visualize "potential surface"
                    Ygrid: The Y grid to visualize "potential surface"
                    Zgrid: The interpolated potential corresponding to the X,Y grids. In Tang method, this is the action
                        value for the learned least action path.
                if Ao method is used:
                    P: Steady state distribution or the Boltzmann-Gibbs distribution for the state variable.
                    S: List of constant symmetric and semi-positive matrix or friction (dissipative) matrix,
                        corresponding to the divergence part, at each position from X.
                    A: List of constant antisymmetric matrix or transverse (non-dissipative) matrix, corresponding to
                        the curl part, at each position from X.
                if Tang method is used:
                    LAP: The least action path learned.
        """

        if method == "Ao":
            X = adata.obsm["X_" + basis]
            X, U, P, vecMat, S, A = Ao_pot_map(self.VecFld["Function"], X, D=self.VecFld["DiffusionMatrix"](X.T))

            Xgrid, Ygrid, Zgrid = construct_Ao_potential_grid(X=X, U=U)

            adata.uns["grid_Pot_" + basis] = {
                "Xgrid": Xgrid,
                "Ygrid": Ygrid,
                "Zgrid": Zgrid,
                "P": P,
                "S": S,
                "A": A,
            }
        elif method == "Bhattacharya":
            (_, _, _, _, numPaths, numTimeSteps, pot_path, path_tag, attractors_pot, x_path, y_path,) = path_integral(
                self.VecFld["Function"],
                x_lim=x_lim,
                y_lim=y_lim,
                xyGridSpacing=xyGridSpacing,
                dt=dt,
                tol=tol,
                numTimeSteps=numTimeSteps,
            )

            Xgrid, Ygrid, Zgrid = alignment(
                numPaths,
                numTimeSteps,
                pot_path,
                path_tag,
                attractors_pot,
                x_path,
                y_path,
            )

            adata.uns["grid_Pot_" + basis] = {
                "Xgrid": Xgrid,
                "Ygrid": Ygrid,
                "Zgrid": Zgrid,
            }

            return adata
        # make sure to also obtain Xgrid, Ygrid, Zgrid, etc.
        elif method == "Tang":
            Function, DiffusionMatrix = (
                self.VecFld["Function"],
                self.VecFld["DiffusionMatrix"],
            )
            (boundary, n_points, fixed_point_only, find_fixed_points, refpoint, stable, saddle,) = (
                self.parameters["boundary"],
                self.parameters["n_points"],
                self.parameters["fixed_point_only"],
                self.parameters["find_fixed_points"],
                self.parameters["refpoint"],
                self.parameters["stable"],
                self.parameters["saddle"],
            )

            X, Y, retmat, LAP = Tang_action(
                Function=self.VecFld["Function"],
                DiffusionMatrix=self.VecFld["DiffusionMatrix"],
                boundary=self.parameters["boundary"],
                n_points=self.parameters["n_points"],
                fixed_point_only=self.parameters["fixed_point_only"],
                find_fixed_points=self.parameters["find_fixed_points"],
                refpoint=self.parameters["refpoint"],
                stable=self.parameters["stable"],
                saddle=self.parameters["saddle"],
            )

            adata.uns["grid_Pot_" + basis] = {"Xgrid": X, "Ygrid": Y, "Zgrid": retmat}

            return adata
            # return retmat, LAP


def search_fixed_points(
    func: Callable,
    domain: np.ndarray,
    x0: np.ndarray,
    x0_method: str = "lhs",
    reverse: bool = False,
    return_x0: bool = False,
    fval_tol: float = 1e-8,
    remove_outliers: bool = True,
    ignore_fsolve_err: bool = False,
    **fsolve_kwargs
) -> Union[FixedPoints, Tuple[FixedPoints, np.ndarray]]:
    """Search the fixed points of (learned) vector field function in a given domain.

    The initial points are sampled by given methods. Then the function uses the fsolve function
    from SciPy to find the fixed points and Numdifftools to compute the Jacobian matrix of the function.

    Args:
        func: The function of the (learned) vector field function that are required to fixed points for.
        domain: The domain to search in.
        x0: The initial point to start with.
        x0_method: The method to sample initial points.
        reverse: Whether to reverse the sign (direction) of vector field (VF).
        return_x0: Whether to return the initial points used in the search.
        fval_tol: The tolerance for the function value at the fixed points.
        remove_outliers: Whether to remove the outliers.
        ignore_fsolve_err: Whether to ignore the fsolve error.

    Returns:
        The fixed points found with their Jacobian matrix of the function. The sampled initial points
        will be returned as well if return_x0 == True.
    """
    import numdifftools as nda

    func_ = (lambda x: -func(x)) if reverse else func
    k = domain.shape[1]

    if np.isscalar(x0):
        n = x0

        if k > 2 and x0_method == "grid":
            warn("The dimensionality is too high (%dD). Using lhs instead..." % k)
            x0_method = "lhs"

        if x0_method == "lhs":
            print("Sampling initial points using latin hypercube sampling...")
            x0 = lhsclassic(n, k)
        elif x0_method == "grid":
            print("Sampling initial points on a grid...")
            pass
        else:
            print("Sampling initial points randomly (uniform distribution)...")
            x0 = np.random.rand(n, k)

        x0 = x0 * (domain[1] - domain[0]) + domain[0]

    fp = FixedPoints()
    succeed = 0
    for i in range(len(x0)):
        x, fval_dict, ier, mesg = sp.optimize.fsolve(func_, x0[i], full_output=True, **fsolve_kwargs)

        if ignore_fsolve_err:
            ier = 1
        if fval_dict["fvec"].dot(fval_dict["fvec"]) > fval_tol and ier == 1:
            ier = -1
            mesg = "Function evaluated at the output is larger than the tolerance."
        elif remove_outliers and is_outside_domain(x, domain) and ier == 1:
            ier = -2
            mesg = "The output is outside the domain."

        if ier == 1:
            jacobian_mat = nda.Jacobian(func_)(np.array(x))
            fp.add_fixed_points([x], [jacobian_mat])
            succeed += 1
        else:
            # jacobian_mat = nda.Jacobian(func_)(np.array(x))
            # fp.add_fixed_points([x], [jacobian_mat])
            print("Solution not found: " + mesg)

    print("%d/%d solutions found." % (succeed, len(x0)))

    if return_x0:
        return fp, x0
    else:
        return fp


def gen_gradient(
    dim: int,
    N: int,
    Function: Callable,
    DiffusionMatrix: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the gradient of the (learned) vector field function for the least action path (LAP) symbolically

    Args:
        dim: The number of dimension of the system.
        N: The number of the points on the discretized path of the LAP.
        Function: The function of the (learned) vector field function that is needed to calculate the Jacobian matrix.
        DiffusionMatrix: The function that returns the diffusion matrix which can be variable (e.g. gene) dependent

    Returns:
        ret: The symbolic function that calculates the gradient of the LAP based on the Jacobian of the vector field function.
        V: A matrix consists of the coordinates of the unstable steady state.
    """

    from StringFunction import StringFunction
    from sympy import Identity, Matrix, MatrixSymbol, simplify, symbols

    N = N + 1
    X = MatrixSymbol("x", dim, N)
    X2 = MatrixSymbol("y", dim, N)

    # for i in range(dim):
    #     for j in range(N):
    #         X2[i, j] = symbols('x[' + str(i) + ', ' + str(j) + ']')

    S = 0 * Identity(1)

    dt = symbols("dt")
    for k in np.arange(1, N):  # equation 18
        k
        # S+1/4*dt*((X(:,k)-X(:,k-1))/dt-ODE(X(:,k-1))).'*(DiffusionMatrix(X(:,k-1)))^(-1)*((X(:,k)-X(:,k-1))/dt-ODE(X(:,k-1)));
        t1 = 1 / 4 * dt * ((X[:, k] - X[:, k - 1]) / dt - Matrix(Function(X[:, k - 1]))).T

        t2 = Matrix(np.linalg.inv(DiffusionMatrix(X[:, k - 1])))
        t3 = (X[:, k] - X[:, k - 1]) / dt - Matrix(Function(X[:, k - 1]))
        S = S + t1 * t2 * t3

    J_res = Matrix(S).jacobian(Matrix(X).reshape(Matrix(X).shape[1] * 2, 1))
    ret = simplify(J_res)  #
    # ret=simplify(jacobian(S,reshape(X,[],1)));
    ret = ret.reshape(X.shape[0], X.shape[1])
    # ret=reshape(ret,2,[]);
    V = ret.reshape(X.shape[0], X.shape[1])  # retsubs(X, X2)

    # convert the result into a function by st.StringFunction
    str_V = str(V)
    str_V_processed = str_V.replace("transpose", "").replace("Matrix", "np.array")

    f_str = """
    def graident(dt, x):
        ret = %s

        return ret

            """ % str(
        str_V_processed
    )
    # ret = StringFunction(f_str, independent_variable=x, dt=dt, x=x)

    return ret, V


def DiffusionMatrix(x: np.ndarray) -> np.ndarray:
    """Diffusion matrix can be variable dependent

    Args:
        x: The matrix of sampled points (cells) in the (gene expression) state space. A

    Returns:
        out: The diffusion matrix. By default, it is a diagonal matrix.
    """
    out = np.zeros((x.shape[0], x.shape[0]))
    np.fill_diagonal(out, 1)

    return out
