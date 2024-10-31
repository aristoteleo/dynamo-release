from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy as sp
import scipy.optimize

# import autograd.numpy as autonp
# from autograd import grad, jacobian # calculate gradient and jacobian


# the LAP method should be rewritten in TensorFlow/PyTorch using optimization with SGD
def Tang_action(
    Function: Callable,
    DiffusionMatrix: Callable,
    boundary: List[float],
    n_points: int = 25,
    fixed_point_only: bool = False,
    find_fixed_points: bool = False,
    refpoint: Optional[np.ndarray] = None,
    stable: Optional[np.ndarray] = None,
    saddle: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mapping potential landscape with the algorithm developed by Tang method.

    Details of the method can be found at Tang, Ying, et al. "Potential landscape of high dimensional nonlinear
    stochastic dynamics with large noise." Scientific reports 7.1 (2017): 15762. Main steps include:
        1. Find the fixed points.
        2. Classify all the fixed points into two groups by calculating the eigenvalues of the linearized Jacobian
            matrix in their neighborhood: stable fixed points (no eigenvalue with positive real part) and unstable
            points (at least one eigenvalue with positive real part).
        3. Choose a saddle point as reference. Start from the points in a small neighborhood of the saddle point, a
            nd simulate the ODE to find all the stable fixed points reached. Calculate potential difference between
            the saddle point and the stable fixed points by the least action method.
        4. Repeat step 3 for all saddle points. Assign relative potential difference between the saddle points if they
            reach a common stable fixed point.
        5. For any other points in state space, simulate the ODE to find the fixed point it reaches. Obtain their
            potential difference by the least action method. The total computational cost depends on the potential value
            of how many points are calculated.
        6. With the calculated potential values, extract the relative probabilities between the states.

    Args:
        Function: The (learned) vector field function.
        DiffusionMatrix: The function that returns the diffusion matrix which can variable (for example, gene) dependent.
        boundary: The boundary of the state space.
        n_points: The number of points along the least action path.
        fixed_point_only: Whether to calculate the potential value for the fixed points only.
        find_fixed_points: Whether to find the fixed points.
        refpoint: The reference point (saddle point) used to calculate the potential value of other points.
        stable: The matrix consists of the coordinates (gene expression configuration) of the stable steady state.
        saddle: The matrix consists of the coordinates (gene expression configuration) of the unstable steady state.

    Returns:
        A tuple includes:
            X: A matrix storing the x-coordinates on the two-dimensional grid.
            Y: A matrix storing the y-coordinates on the two-dimensional grid.
            retmat: A matrix storing the potential value at each position.
            LAP: A list of the least action path (LAP) for each position.
    """

    print("stable is ", stable)

    if (stable is None and saddle is None) or find_fixed_points is True:
        print("stable is ", stable)
        stable, saddle = gen_fixed_points(
            Function,
            auto_func=False,
            dim_range=[-25, 25],
            RandNum=100,
            EqNum=2,
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
        X, Y = None, None
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

    for ind in I:  # action(n_points,tmax,point_start,point_end, boundary, Function, DiffusionMatrix):
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

    return X, Y, retmat, LAP


def gen_fixed_points(
    func: Callable,
    auto_func: Optional[np.ndarray],
    dim_range: List,
    RandNum: int,
    EqNum: int,
    reverse: bool = False,
    grid_num: int = 50,
    x_ini: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the fixed points of (learned) vector field function . Classify the fixed points into classes of stable and saddle points
    based on the eigenvalue of the Jacobian on the point.

    Args:
        func: The function of the (learned) vector field function that are required to fixed points for
        auto_func: The function that is written with autograd of the same ODE equations
            that is used to calculate the Jacobian matrix. If auto_func is set to be None,
            Jacobian is calculated through the fjac, r returned from fsolve.
        dim_range: The range of variables in the ODE equations.
        RandNum: The number of random initial points to sample.
        EqNum: The number of equations (dimension) of the system.
        reverse: Whether to reverse the sign (direction) of vector field (VF).
        grid_num: The number of grids on each dimension, only used when the EqNum is 2 and x_ini is None.
        x_ini: The user provided initial points that is used to find the fixed points

    Returns:
        stable: A matrix consists of the coordinates of the stable steady state
        saddle: A matrix consists of the coordinates of the unstable steady state

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
        Grid_list = np.meshgrid(*[np.linspace(i, j, grid_num) for i, j in zip(_min, _max)])
        x_ini = np.array([i.flatten() for i in Grid_list]).T

    RandNum = RandNum if x_ini is None else x_ini.shape[0]  # RandNum set to the manually input steady state estimates
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
            x0 = np.random.uniform(0, 1, EqNum) * (dim_range[1] - dim_range[0]) + dim_range[0]
            x, fval_dict, _, _ = sp.optimize.fsolve(func_, x0, maxfev=450000, xtol=FixedPointConst, full_output=True)
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
            x, fval_dict, _, _ = sp.optimize.fsolve(func_, x0, maxfev=450000, xtol=FixedPointConst, full_output=True)
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
                if i is not StableNum + 1 and temp.dot(temp) < ZeroConst:  # avoid duplicated attractors
                    StableTimes[i] = StableTimes[i] + 1
                    break
            if i is StableNum:
                StableTimes[StableNum] = StableTimes[StableNum] + 1
                StablePoint[StableNum, :] = FixedPoint[time, :]
                StableNum = StableNum + 1
        elif Type[time] == -1:
            for i in range(SaddleNum + 1):
                temp = SaddlePoint[i, :] - FixedPoint[time, :]
                if i is not SaddlePoint and temp.dot(temp) < ZeroConst:  # avoid duplicated saddle point
                    SaddleTimes[i] = SaddleTimes[i] + 1
                    break
            if i is SaddleNum:
                SaddleTimes[SaddleNum] = SaddleTimes[SaddleNum] + 1
                SaddlePoint[SaddleNum, :] = FixedPoint[time, :]
                SaddleNum = SaddleNum + 1

    stable, saddle = StablePoint[:StableNum, :].T, SaddlePoint[:SaddleNum, :].T

    return stable, saddle


def action(
    n_points: int,
    tmax: int,
    point_start: np.ndarray,
    point_end: np.ndarray,
    boundary: np.ndarray,
    Function: Callable,
    DiffusionMatrix: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """It calculates the minimized action value given an initial path, ODE, and diffusion matrix. The minimization is
    realized by scipy.optimize.Bounds function in python (without using the gradient of the action function).

    Args:
        n_points: The number of points along the least action path.
        tmax: The value at maximum t.
        point_start: The matrix for storing the coordinates (gene expression configuration) of the start point (initial cell state).
        point_end: The matrix for storing the coordinates (gene expression configuration) of the end point (terminal cell state).
        boundary: Not used.
        Function: The (reconstructed) vector field function.
        DiffusionMatrix: The function that returns the diffusion matrix which can variable (for example, gene) dependent.

    Returns:
        fval: The action value for the learned least action path.
        output_path: The least action path learned.
    """

    dim = point_end.shape[0]  # genes x cells
    dt = tmax / n_points
    lambda_f = lambda x: IntGrad(
        np.hstack((point_start, x.reshape((dim, -1)), point_end)),
        Function,
        DiffusionMatrix,
        dt,
    )

    # initial path as a line connecting start point and end point point_start*ones(1,n_points+1)+(point_end-point_start)*(0:tmax/n_points:tmax)/tmax;
    initpath = (
        point_start.dot(np.ones((1, n_points + 1)))
        + (point_end - point_start).dot(np.linspace(0, tmax, n_points + 1, endpoint=True).reshape(1, -1)) / tmax
    )

    Bounds = scipy.optimize.Bounds(
        (np.ones((1, (n_points - 1) * 2)) * boundary[0]).flatten(),
        (np.ones((1, (n_points - 1) * 2)) * boundary[1]).flatten(),
        keep_feasible=True,
    )

    # sp.optimize.least_squares(lambda_f, initpath[:, 1:n_points].flatten())
    res = sp.optimize.minimize(
        lambda_f, initpath[:, 1:n_points].flatten(), tol=1e-12
    )  # , bounds=Bounds , options={"maxiter": 250}
    fval, output_path = (
        res["fun"],
        np.hstack((point_start, res["x"].reshape((2, -1)), point_end)),
    )

    return fval, output_path


# rewrite gen_gradient with autograd or TF
def IntGrad(
    points: np.ndarray,
    Function: Callable,
    DiffusionMatrix: Callable,
    dt: float,
) -> np.ndarray:
    """Calculate the action of the path based on the (reconstructed) vector field function and diffusion matrix (Eq. 18)

    Arg:
        points: The sampled points in the state space used to calculate the action.
        Function: The (learned) vector field function.
        DiffusionMatrix: The function that returns diffusion matrix which can be dependent on the variables (for example, genes).
        dt: The time interval used in calculating action.

    Returns:
        integral: The action calculated based on the input path, the vector field function and the diffusion matrix.
    """

    integral = 0
    for k in np.arange(1, points.shape[1]):
        Tmp = (points[:, k] - points[:, k - 1]).reshape((-1, 1)) / dt - Function(points[:, k - 1]).reshape((-1, 1))
        integral = (
            integral + (Tmp.T).dot(np.linalg.matrix_power(DiffusionMatrix(points[:, k - 1]), -1)).dot(Tmp) * dt
        )  # part of the Eq. 18 in Sci. Rep. paper
    integral = integral / 4

    return integral[0, 0]
