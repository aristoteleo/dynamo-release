import numpy as np
import scipy as sp
import scipy.optimize

from .Bhattacharya import path_integral, alignment
from .Ao import Ao_pot_map

# import autograd.numpy as autonp
# from autograd import grad, jacobian # calculate gradient and jacobian

from .Wang import Wang_action, Wang_LAP
# the LAP method should be rewritten in TensorFlow/PyTorch using optimization with SGD

from .topography import FixedPoints
from .utils import is_outside_domain
from ..tools.sampling import lhsclassic
from warnings import warn

def search_fixed_points(func, domain, x0, x0_method='lhs', 
    reverse=False, return_x0=False, fval_tol=1e-8, 
    remove_outliers=True, ignore_fsolve_err=False, **fsolve_kwargs):
    import numdifftools as nda

    func_ = (lambda x: -func(x)) if reverse else func
    k = domain.shape[1]

    if np.isscalar(x0):
        n = x0

        if k > 2 and x0_method == 'grid':
            warn('The dimensionality is too high (%dD). Using lhs instead...'%k)
            x0_method = 'lhs'
        
        if x0_method == 'lhs':
            print('Sampling initial points using latin hypercube sampling...')
            x0 = lhsclassic(n, k) 
        elif x0_method == 'grid':
            print('Sampling initial points on a grid...')
            pass
        else:
            print('Sampling initial points randomly (uniform distribution)...')
            x0 = np.random.rand(n, k)

        x0 = x0 * (domain[1] - domain[0]) + domain[0]

    fp = FixedPoints()
    succeed = 0
    for i in range(len(x0)):
        x, fval_dict, ier, mesg = sp.optimize.fsolve(
                func_, x0[i], full_output=True, **fsolve_kwargs)

        if ignore_fsolve_err:
            ier = 1
        if fval_dict["fvec"].dot(fval_dict["fvec"]) > fval_tol and ier == 1:
            ier = -1
            mesg = 'Function evaluated at the output is larger than the tolerance.'
        elif remove_outliers and is_outside_domain(x, domain) and ier == 1:
            ier = -2
            mesg = 'The output is outside the domain.'

        if ier == 1:
            jacobian_mat = nda.Jacobian(func_)(np.array(x))
            fp.add_fixed_points([x], [jacobian_mat])
            succeed += 1
        else:
            #jacobian_mat = nda.Jacobian(func_)(np.array(x))
            #fp.add_fixed_points([x], [jacobian_mat])
            print('Solution not found: ' + mesg)

    print('%d/%d solutions found.'%(succeed, len(x0)))
        
    if return_x0:
        return fp, x0
    else:
        return fp
    

def gen_fixed_points(
    func, auto_func, dim_range, RandNum, EqNum, reverse=False, grid_num=50, x_ini=None
):
    """ Calculate the fixed points of (learned) vector field function . Classify the fixed points into classes of stable and saddle points
    based on the eigenvalue of the Jacobian on the point.

    Arguments
    ---------
        func: 'function'
            The function of the (learned) vector field function that are required to fixed points for
        auto_func: 'np.ndarray' (not used)
            The function that is written with autograd of the same ODE equations that is used to calculate the Jacobian matrix.
            If auto_func is set to be None, Jacobian is calculated through the fjac, r returned from fsolve.
        dim_range: 'list'
            The range of variables in the ODE equations
        RandNum: 'int'
            The number of random initial points to sample
        EqNum: 'int'
            The number of equations (dimension) of the system
        reverse: `bool`
            Whether to reverse the sign (direction) of vector field (VF).
        grid_num: `int` (default: 50)
            The number of grids on each dimension, only used when the EqNum is 2 and x_ini is None.
        x_ini: 'np.ndarray'
            The user provided initial points that is used to find the fixed points

    Returns
    -------
    stable: 'np.ndarray'
        A matrix consists of the coordinates of the stable steady state
    saddle: 'np.ndarray'
        A matrix consists of the coordinates of the unstable steady state

    """
    import numdifftools as nda

    if reverse is True:
        func_ = lambda x: -func(x)
    else:
        func_ = func
    ZeroConst = 1e-8
    FixedPointConst = 1e-20
    MaxSolution = 1000

    if x_ini is None and EqNum < 4:
        _min, _max = [dim_range[0]] * EqNum, [dim_range[1]] * EqNum
        Grid_list = np.meshgrid(
            *[np.linspace(i, j, grid_num) for i, j in zip(_min, _max)]
        )
        x_ini = np.array([i.flatten() for i in Grid_list]).T

    RandNum = (
        RandNum if x_ini is None else x_ini.shape[0]
    )  # RandNum set to the manually input steady state estimates
    FixedPoint = np.zeros((RandNum, EqNum))
    Type = np.zeros((RandNum, 1))

    StablePoint = np.zeros((MaxSolution, EqNum))
    SaddlePoint = np.zeros((MaxSolution, EqNum))
    StableTimes = np.zeros((MaxSolution, 1))
    SaddleTimes = np.zeros((MaxSolution, 1))
    StableNum = 0
    SaddleNum = 0

    if x_ini is None:
        for time in range(RandNum):
            x0 = (
                np.random.uniform(0, 1, EqNum) * (dim_range[1] - dim_range[0])
                + dim_range[0]
            )
            x, fval_dict, _, _ = sp.optimize.fsolve(
                func_, x0, maxfev=450000, xtol=FixedPointConst, full_output=True
            )
            # fjac: the orthogonal matrix, q, produced by the QR factorization of the final approximate Jacobian matrix,
            # stored column wise; r: upper triangular matrix produced by QR factorization of the same matrix
            # if auto_func is None:
            #     fval, q, r = fval_dict['fvec'], fval_dict['fjac'], fval_dict['r']
            #     matrixr=np.zeros((EqNum, EqNum))
            #     matrixr[np.triu_indices(EqNum)]=fval_dict["r"]
            #     jacobian_mat=(fval_dict["fjac"]).dot(matrixr)
            # else:
            fval = fval_dict["fvec"]
            jacobian_mat = nda.Jacobian(func_)(np.array(x))  # autonp.array?

            jacobian_mat[np.isinf(jacobian_mat)] = 0
            if fval.dot(fval) < FixedPointConst:
                FixedPoint[time, :] = x
                ve, _ = sp.linalg.eig(jacobian_mat)
                for j in range(EqNum):
                    if np.real(ve[j]) > 0:
                        Type[time] = -1
                        break
                if not Type[time]:
                    Type[time] = 1
    else:
        for time in range(x_ini.shape[0]):
            x0 = x_ini[time, :]
            x, fval_dict, _, _ = sp.optimize.fsolve(
                func_, x0, maxfev=450000, xtol=FixedPointConst, full_output=True
            )
            # fjac: the orthogonal matrix, q, produced by the QR factorization of the final approximate Jacobian matrix,
            # stored column wise; r: upper triangular matrix produced by QR factorization of the same matrix
            # if auto_func is None:
            #     fval, q, r = fval_dict['fvec'], fval_dict['fjac'], fval_dict['r']
            #     matrixr=np.zeros((EqNum, EqNum))
            #     matrixr[np.triu_indices(EqNum)]=fval_dict["r"]
            #     jacobian_mat=(fval_dict["fjac"]).dot(matrixr)
            # else:
            fval = fval_dict["fvec"]
            jacobian_mat = nda.Jacobian(func_)(np.array(x))  # autonp.array?

            jacobian_mat[np.isinf(jacobian_mat)] = 0
            if fval.dot(fval) < FixedPointConst:
                FixedPoint[time, :] = x
                ve, _ = sp.linalg.eig(jacobian_mat)
                for j in range(EqNum):
                    if np.real(ve[j]) > 0:
                        Type[time] = -1
                        break
                if not Type[time]:
                    Type[time] = 1

    for time in range(RandNum):
        if Type[time] == 0:
            continue
        elif Type[time] == 1:
            for i in range(StableNum + 1):
                temp = StablePoint[i, :] - FixedPoint[time, :]
                if (
                    i is not StableNum + 1 and temp.dot(temp) < ZeroConst
                ):  # avoid duplicated attractors
                    StableTimes[i] = StableTimes[i] + 1
                    break
            if i is StableNum:
                StableTimes[StableNum] = StableTimes[StableNum] + 1
                StablePoint[StableNum, :] = FixedPoint[time, :]
                StableNum = StableNum + 1
        elif Type[time] == -1:
            for i in range(SaddleNum + 1):
                temp = SaddlePoint[i, :] - FixedPoint[time, :]
                if (
                    i is not SaddlePoint and temp.dot(temp) < ZeroConst
                ):  # avoid duplicated saddle point
                    SaddleTimes[i] = SaddleTimes[i] + 1
                    break
            if i is SaddleNum:
                SaddleTimes[SaddleNum] = SaddleTimes[SaddleNum] + 1
                SaddlePoint[SaddleNum, :] = FixedPoint[time, :]
                SaddleNum = SaddleNum + 1

    stable, saddle = StablePoint[:StableNum, :].T, SaddlePoint[:SaddleNum, :].T

    return stable, saddle


def gen_gradient(dim, N, Function, DiffusionMatrix):
    """Calculate the gradient of the (learned) vector field function for the least action path (LAP) symbolically

    Arguments
    ---------
        dim: 'int'
            The number of dimension of the system
        N: 'int'
            The number of the points on the discretized path of the LAP
        Function: 'function'
            The function of the (learned) vector field function that is needed to calculate the Jacobian matrix
        DiffusionMatrix: Python function
            The function that returns the diffusion matrix which can be variable (for example, gene) dependent

    Returns
    -------
    ret: 'np.ndarray'
        The symbolic function that calculates the gradient of the LAP based on the Jacobian of the vector field function
    V: 'np.ndarray'
        A matrix consists of the coordinates of the unstable steady state
    """

    from sympy import MatrixSymbol, Identity, symbols, Matrix, simplify
    from StringFunction import StringFunction

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
        t1 = (
            1
            / 4
            * dt
            * ((X[:, k] - X[:, k - 1]) / dt - Matrix(Function(X[:, k - 1]))).T
        )

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


##################################################
# rewrite gen_gradient with autograd or TF
##################################################


def IntGrad(points, Function, DiffusionMatrix, dt):
    """Calculate the action of the path based on the (reconstructed) vector field function and diffusion matrix (Eq. 18)

    Arguments
    ---------
        points: 'np.ndarray'
            The sampled points in the state space used to calculate the action.
        Function: 'function'
            The (learned) vector field function.
        DiffusionMatrix: 'function'
            The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
        dt: 'function'
            The time interval used in calculating action

    Returns
    -------
    integral: 'np.ndarray'
        The action calculated based on the input path, the vector field function and the diffusion matrix.
    """

    integral = 0
    for k in np.arange(1, points.shape[1]):
        Tmp = (points[:, k] - points[:, k - 1]).reshape((-1, 1)) / dt - Function(
            points[:, k - 1]
        ).reshape((-1, 1))
        integral = (
            integral
            + (Tmp.T)
            .dot(np.linalg.matrix_power(DiffusionMatrix(points[:, k - 1]), -1))
            .dot(Tmp)
            * dt
        )  # part of the Eq. 18 in Sci. Rep. paper
    integral = integral / 4

    return integral[0, 0]


def DiffusionMatrix(x):
    """ Diffusion matrix can be variable dependent

    Arguments
    ---------
        x: 'np.ndarray'
            The matrix of sampled points (cells) in the (gene expression) state space.

    Returns
    -------
    out: 'np.ndarray'
        The diffusion matrix. By default, it is a diagonal matrix.
    """
    out = np.zeros((x.shape[0], x.shape[0]))
    np.fill_diagonal(out, 1)

    return out


# rewrite action in TF with SGD
def action(n_points, tmax, point_start, point_end, boundary, Function, DiffusionMatrix):
    """It calculates the minimized action value given an intial path, ODE, and diffusion matrix. The minimization is
    realized by scipy.optimize.Bounds function in python (withnot using the gradient of the action function).

    Arguments
    ---------
        n_points: 'int'
            The number of points along the least action path.
        tmax: 'int'
            The value at maximum t.
        point_start: 'np.ndarray'
            The matrix for storing the coordinates (gene expression configuration) of the start point (initial cell state).
        point_end: 'np.ndarray'
            The matrix for storing the coordinates (gene expression configuration) of the end point (terminal cell state).
        Function: 'function'
            The (reconstructed) vector field function.
        DiffusionMatrix: 'function'
            The function that returns the diffusion matrix which can variable (for example, gene) dependent.

    Returns
    -------
    fval: 'np.ndarray'
        The action value for the learned least action path.
    output_path: 'np.ndarray'
        The least action path learned
    """

    dim = point_end.shape[0]  # genes x cells
    dt = tmax / n_points
    lambda_f = lambda x: IntGrad(
        np.hstack((point_start, x.reshape((2, -1)), point_end)),
        Function,
        DiffusionMatrix,
        dt,
    )

    # initial path as a line connecting start point and end point point_start*ones(1,n_points+1)+(point_end-point_start)*(0:tmax/n_points:tmax)/tmax;
    initpath = (
        point_start.dot(np.ones((1, n_points + 1)))
        + (point_end - point_start).dot(
            np.linspace(0, tmax, n_points + 1, endpoint=True).reshape(1, -1)
        )
        / tmax
    )

    Bounds = scipy.optimize.Bounds(
        (np.ones((1, (n_points - 1) * 2)) * boundary[0]).flatten(),
        (np.ones((1, (n_points - 1) * 2)) * boundary[1]).flatten(),
        keep_feasible=True,
    )

    # sp.optimize.least_squares(lambda_f, initpath[:, 1:n_points].flatten())
    res = sp.optimize.minimize(
        lambda_f, initpath[:, 1:n_points], tol=1e-12
    )  # , bounds=Bounds , options={"maxiter": 250}
    fval, output_path = (
        res["fun"],
        np.hstack((point_start, res["x"].reshape((2, -1)), point_end)),
    )

    return fval, output_path


def Potential(adata, DiffMat=None, method="Ao", **kwargs):
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

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains embedding and velocity data
        method: `str` (default: `Ao`)
            Method to map the potential landscape.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            `AnnData` object that is updated with the `Pot` dictionary in the `uns` attribute.

    """

    Function = adata.uns["VecFld"]["VecFld"]
    DiffMat = DiffusionMatrix if DiffMat is None else DiffMat
    pot = Pot(Function, DiffMat, **kwargs)
    pot.fit(method=method)

    return adata


class Pot:
    def __init__(
        self,
        Function=None,
        DiffMat=None,
        boundary=None,
        n_points=25,
        fixed_point_only=False,
        find_fixed_points=False,
        refpoint=None,
        stable=None,
        saddle=None,
    ):
        """ It implements the least action method to calculate the potential values of fixed points for a given SDE (stochastic
        differential equation) model. The function requires the vector field function and a diffusion matrix. This code is based
        on the MATLAB code from Ruoshi Yuan and Ying Tang. Potential landscape of high dimensional nonlinear stochastic dynamics with
        large noise. Y Tang, R Yuan, G Wang, X Zhu, P Ao - Scientific reports, 2017

        Arguments
        ---------
            Function: 'function'
                The (reconstructed) vector field function.
            DiffMat: 'function'
                The function that returns the diffusion matrix which can variable (for example, gene) dependent.
            boundary: 'list'
                The range of variables (genes).
            n_points: 'int'
                The number of points along the least action path.
            fixed_point_only: 'bool'
                The logic flag to determine whether only the potential for fixed point or entire space should be mapped.
            find_fixed_points: 'bool'
                The logic flag to determine whether only the gen_fixed_points function should be run to identify fixed points.
            refpoint: 'np.ndarray'
                The reference point to define the potential.
            stable: 'np.ndarray'
                The matrix for storing the coordinates (gene expression configuration) of the stable fixed point (characteristic state of a particular cell type).
            saddle: 'np.ndarray'
                The matrix for storing the coordinates (gene expression configuration) of the unstable fixed point (characteristic state of cells prime to bifurcation).
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
        adata,
        x_lim,
        y_lim,
        basis="umap",
        method="Ao",
        xyGridSpacing=2,
        dt=1e-2,
        tol=1e-2,
        numTimeSteps=1400,
    ):
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

        Arguments
        ---------
            adata: :class:`~anndata.AnnData`
                AnnData object that contains U_grid and V_grid data
            x_lim: `list`
                Lower or upper limit of x-axis.
            y_lim: `list`
                Lower or upper limit of y-axis
            basis: `str` (default: umap)
                The dimension reduction method to use.
            method: 'string' (default: Bhattacharya)
                Method used to map the pseudo-potential landscape. By default, it is Bhattacharya (A deterministic map of
                Waddington’s epigenetic landscape for cell fate specification. Sudin Bhattacharya, Qiang Zhang and Melvin
                E. Andersen). Other methods will be supported include: Tang (), Ping (), Wang (), Zhou ().

        Returns
        -------
        if Bhattacharya is used:
            Xgrid: 'np.ndarray'
                The X grid to visualize "potential surface"
            Ygrid: 'np.ndarray'
                The Y grid to visualize "potential surface"
            Zgrid: 'np.ndarray'
                The interpolate potential corresponding to the X,Y grids.

        if Tang method is used:
        retmat: 'np.ndarray'
            The action value for the learned least action path.
        LAP: 'np.ndarray'
            The least action path learned
        """

        if method == "Ao":
            X = adata.obsm["X_" + basis]
            X, U, P, vecMat, S, A = Ao_pot_map(
                self.VecFld["Function"], X, D=self.VecFld["DiffusionMatrix"]
            )

            adata.uns["grid_Pot_" + basis] = {
                "Xgrid": X,
                "Ygrid": U,
                "Zgrid": P,
                "S": S,
                "A": A,
            }
        elif method == "Bhattacharya":
            (
                numPaths,
                numTimeSteps,
                pot_path,
                path_tag,
                attractors_pot,
                x_path,
                y_path,
            ) = path_integral(
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
            (
                boundary,
                n_points,
                fixed_point_only,
                find_fixed_points,
                refpoint,
                stable,
                saddle,
            ) = (
                self.parameters["boundary"],
                self.parameters["n_points"],
                self.parameters["fixed_point_only"],
                self.parameters["find_fixed_points"],
                self.parameters["refpoint"],
                self.parameters["stable"],
                self.parameters["saddle"],
            )

            print("stable is ", stable)

            if (stable is None and saddle is None) or find_fixed_points is True:
                print("stable is ", stable)
                stable, saddle = gen_fixed_points(
                    Function, auto_func=False, dim_range=range, RandNum=100, EqNum=2
                )  # gen_fixed_points(vector_field_function, auto_func = None, dim_range = [-25, 25], RandNum = 5000, EqNum = 2, x_ini = None)

            print("stable 2 is ", stable)
            points = np.hstack((stable, saddle))
            refpoint = stable[:, 0][:, None] if refpoint is None else refpoint

            TotalTime, TotalPoints = 2, n_points

            if fixed_point_only:
                StateNum = points.shape[1]
                retmat = np.Inf * np.ones((StateNum, 1))
                LAP = [None] * StateNum
                I = range(StateNum)
            else:
                dx = (np.diff(boundary)) / (TotalPoints - 1)
                dy = dx
                [X, Y] = np.meshgrid(
                    np.linspace(boundary[0], boundary[1], TotalPoints),
                    np.linspace(boundary[0], boundary[1], TotalPoints),
                )
                retmat = np.Inf * np.ones((TotalPoints, TotalPoints))
                LAP = [None] * TotalPoints * TotalPoints

                points = np.vstack((X.flatten(), Y.flatten()))
                I = range(points.shape[1])

            for (
                ind
            ) in (
                I
            ):  # action(n_points,tmax,point_start,point_end, boundary, Function, DiffusionMatrix):
                print("current ind is ", ind)
                lav, lap = action(
                    TotalPoints,
                    TotalTime,
                    points[:, ind][:, None],
                    refpoint,
                    boundary,
                    Function,
                    DiffusionMatrix,
                )

                i, j = ind % TotalPoints, int(ind / TotalPoints)
                print(
                    "TotalPoints is ",
                    TotalPoints,
                    "ind is ",
                    ind,
                    "i, j are",
                    i,
                    " ",
                    j,
                )
                if lav < retmat[i, j]:
                    retmat[i, j] = lav
                    LAP[ind] = lap
                print(retmat)

            # adata.uns['grid_Pot_' + basis] = {'Xgrid': Xgrid, "Ygrid": Ygrid, 'Zgrid': Zgrid}

            return adata
            # return retmat, LAP
