import numpy as np
from scipy.linalg import lstsq
import numpy.matlib


def norm(X, V, T):
    """Normalizes the X, Y (X + V) matrix to have zero means and unit covariance.
        We use the mean of X, Y's center and scale parameters to normalize T.

        Arguments
        ---------
        X: 'np.ndarray'
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        V: 'np.ndarray'
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic velocity estimated calculated by dynamo or velocyto, scvelo.
        T: 'np.ndarray'
            Current state on a grid which is often used to visualize the vector field. This corresponds to, for example, the spliced transcriptomic state.

        Returns
        -------
            Add norm_dict to the class which includes the mean and scale values for X, Y used in normalizing the data.
        """
    Y = X + V
    n, m = X.shape[0], V.shape[0]

    xm = np.mean(X, 0)
    ym = np.mean(Y, 0)

    x, y, t = (
        X - xm[None, :],
        Y - ym[None, :],
        T - (1 / 2 * (xm[None, :] + ym[None, :])),
    )

    xscale, yscale = (
        np.sqrt(np.sum(np.sum(x ** 2, 1)) / n),
        np.sqrt(np.sum(np.sum(y ** 2, 1)) / m),
    )

    X, Y, T = x / xscale, y / yscale, t / (1 / 2 * (xscale + yscale))

    X, V, T = X, Y - X, T
    norm_dict = {"xm": xm, "ym": ym, "xscale": xscale, "yscale": yscale}

    return X, V, T, norm_dict


def con_K(x, y, beta):
    """con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Arguments
    ---------
        x: 'np.ndarray'
            Original training data points.
        y: 'np.ndarray'
            Control points used to build kernel basis functions.
        beta: 'float' (default: 0.1)
            Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),

    Returns
    -------
    K: 'np.ndarray'
    the kernel to represent the vector field function.
    """

    n = x.shape[0]
    m = y.shape[0]

    # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
    # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
    K = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(
        np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
    K = np.squeeze(np.sum(K ** 2, 1))
    K = -beta * K
    K = np.exp(K)  #

    return K


def con_K_div_cur_free(x, y, sigma=0.8, gamma=0.5):
    """Learn a convex combination of the divergence-free kernel T_df and curl-free kernel T_cf with a bandwidth sigma and a combination coefficient gamma.

    Arguments
    ---------
        x: 'np.ndarray'
            Original training data points.
        y: 'np.ndarray'
            Control points used to build kernel basis functions
        sigma: 'int'
            Bandwidth parameter.
        gamma: 'int'
            Combination coefficient for the divergence-free or the curl-free kernels.

    Returns
    -------
        A tuple of G (the combined kernel function), divergence-free kernel and curl-free kernel.

    See also:: :func:`sparseVFC`.
    """
    m, d = x.shape
    n, d = y.shape
    sigma2 = sigma ** 2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(
        np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0]
    )
    G_tmp = np.squeeze(np.sum(G_tmp ** 2, 1))
    G_tmp3 = -G_tmp / sigma2
    G_tmp = -G_tmp / (2 * sigma2)
    G_tmp = np.exp(G_tmp) / sigma2
    G_tmp = np.kron(G_tmp, np.ones(d))

    x_tmp = np.matlib.tile(x, [n, 1])
    y_tmp = np.matlib.tile(y, [1, m]).T
    y_tmp = y_tmp.reshape((d, m * n)).T
    xminusy = x_tmp - y_tmp
    G_tmp2 = np.zeros(d * m, d * n)

    for i in range(d):
        for j in range(d):
            tmp1 = xminusy[:, i].reshape((m, n))
            tmp2 = xminusy[:, j].reshape((m, n))
            tmp3 = tmp1 * tmp2
            tmp4 = np.zeros(d)
            tmp4[i, j] = 1
            tmp4[j, i] = 1
            G_tmp2 = G_tmp2 + np.kron(tmp3, tmp4)

    G_tmp2 = G_tmp2 / sigma2
    G_tmp3 = np.kron((G_tmp3 + d - 1), np.eye(d))
    G_tmp4 = np.kron(np.ones(m, n), np.eye(d)) - G_tmp2
    G = (1 - gamma) * G_tmp * (G_tmp2 + G_tmp3) + gamma * G_tmp * G_tmp4

    return G, (1 - gamma) * G_tmp * (G_tmp2 + G_tmp3), gamma * G_tmp * G_tmp4


def vector_field_function(x, VecFld, dim=None):
    """Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
    """
    # x=np.array(x).reshape((1, -1))
    x = np.array(x)
    if x.ndim == 1:
        x = x[None, :]
    K = con_K(x, VecFld["X_ctrl"], VecFld["beta"])

    if dim is None:
        K = K.dot(VecFld["C"])
    else:
        K = K.dot(VecFld["C"][:, dim])
    return K


def SparseVFC(
    X,
    Y,
    Grid,
    M=100,
    a=5,
    beta=0.1,
    ecr=1e-5,
    gamma=0.9,
    lambda_=3,
    minP=1e-5,
    MaxIter=500,
    theta=0.75,
    div_cur_free_kernels=False,
):
    """Apply sparseVFC (vector field consensus) algorithm to learn a functional form of the vector field on the entire space robustly and efficiently.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

    Arguments
    ---------
        X: 'np.ndarray'
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        Y: 'np.ndarray'
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic velocity estimated calculated by velocyto, scvelo or dynamo.
        Grid: 'np.ndarray'
            Current state on a grid which is often used to visualize the vector field. This corresponds to, for example, the spliced transcriptomic state.
        M: 'int' (default: 100)
            The number of basis functions to approximate the vector field.
        a: 'float' (default: 10)
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.
        beta: 'float' (default: 0.1)
            Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),
        ecr: 'float' (default: 1e-5)
            The minimum limitation of energy change rate in the iteration process.
        gamma: 'float' (default: 0.9)
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        lambda_: 'float' (default: 0.3)
            Represents the trade-off between the goodness of data fit and regularization.
        minP: 'float' (default: 1e-5)
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as minP.
        MaxIter: 'int' (default: 500)
            Maximum iterition times.
        theta: 'float' (default: 0.75)
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta, then it is regarded as an inlier.

    Returns
    -------
    VecFld: 'dict'
    A dictionary which contains X, Y, beta, V, C, P, VFCIndex. Where V = f(X), P is the posterior probability and
    VFCIndex is the indexes of inliers which found by VFC. Note that V = con_K(Grid, ctrl_pts, beta).dot(C) gives the prediction of velocity on Grid (can be any point in the gene expressionstate space).
    """
    Y[~np.isfinite(Y)] = 0  # set nan velocity to 0.
    N, D = Y.shape
    grid_U = None

    # Construct kernel matrix K
    tmp_X = np.unique(X, axis=0)  # return unique rows
    idx = np.random.RandomState(seed=0).permutation(
        tmp_X.shape[0]
    )  # rand select some initial points
    idx = idx[range(min(M, tmp_X.shape[0]))]
    ctrl_pts = tmp_X[idx, :]
    # ctrl_pts = X[range(500), :]

    K = (
        con_K(ctrl_pts, ctrl_pts, beta)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(ctrl_pts, ctrl_pts)[0]
    )
    U = (
        con_K(X, ctrl_pts, beta)
        if div_cur_free_kernels is False
        else con_K_div_cur_free(X, ctrl_pts)[0]
    )
    if Grid is not None:
        grid_U = (
            con_K(Grid, ctrl_pts, beta)
            if div_cur_free_kernels is False
            else con_K_div_cur_free(Grid, ctrl_pts)[0]
        )
    M = ctrl_pts.shape[0]

    # Initialization
    V = np.zeros((N, D))
    C = np.zeros((M, D))
    i, tecr, E = 1, 1, 1
    sigma2 = sum(sum((Y - V) ** 2)) / (N * D)  ## test this
    # sigma2 = 1e-7 if sigma2 > 1e-8 else sigma2

    while i < MaxIter and tecr > ecr and sigma2 > 1e-8:
        # E_step
        E_old = E
        P, E = get_P(Y, V, sigma2, gamma, a)

        E = E + lambda_ / 2 * np.trace(C.T.dot(K).dot(C))
        tecr = abs((E - E_old) / E)

        # print('iterate: {}, gamma: {}, the energy change rate: {}, sigma2={}\n'.format(*[iter, gamma, tecr, sigma2]))

        # M-step. Solve linear system for C.
        P = np.maximum(P, minP)
        C = lstsq(
            ((U.T * numpy.matlib.repmat(P.T, M, 1)).dot(U) + lambda_ * sigma2 * K),
            (U.T * numpy.matlib.repmat(P.T, M, 1)).dot(Y),
        )[0]

        # Update V and sigma**2
        V = U.dot(C)
        Sp = sum(P)
        sigma2 = sum(P.T * np.sum((Y - V) ** 2, 1)) / np.dot(Sp, D)

        # Update gamma
        numcorr = len(np.where(P > theta)[0])
        gamma = numcorr / X.shape[0]

        if gamma > 0.95:
            gamma = 0.95
        elif gamma < 0.05:
            gamma = 0.05

        i += 1

    grid_V = None
    if Grid is not None:
        grid_V = np.dot(grid_U, C)

    VecFld = {
        "X": X,
        "X_ctrl": ctrl_pts,
        "Y": Y,
        "beta": beta,
        "V": V,
        "C": C,
        "P": P,
        "VFCIndex": np.where(P > theta)[0],
        "sigma2": sigma2,
        "grid": Grid,
        "grid_V": grid_V,
        "iteration": i - 1,
    }

    return VecFld


def get_P(Y, V, sigma2, gamma, a):
    """GET_P estimates the posterior probability and part of the energy.

    Arguments
    ---------
        Y: 'np.ndarray'
            Original data.
        V: 'np.ndarray'
            Original data.
        sigma2: 'float'
            sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a.

    Returns
    -------
    P: 'np.ndarray'
        Posterior probability, related to equation 27.
    E: `np.ndarray'
        Energy, related to equation 26.

    """
    D = Y.shape[1]
    temp1 = np.exp(-np.sum((Y - V) ** 2, 1) / (2 * sigma2))
    temp2 = (2 * np.pi * sigma2) ** (D / 2) * (1 - gamma) / (gamma * a)
    temp1[temp1 == 0] = np.min(temp1[temp1 != 0])
    P = temp1 / (temp1 + temp2)
    E = (
        P.T.dot(np.sum((Y - V) ** 2, 1)) / (2 * sigma2)
        + np.sum(P) * np.log(sigma2) * D / 2
    )

    return P, E


class vectorfield:
    def __init__(
        self,
        X=None,
        V=None,
        Grid=None,
        M=100,
        a=5,
        beta=0.1,
        ecr=1e-5,
        gamma=0.9,
        lambda_=3,
        minP=1e-5,
        MaxIter=500,
        theta=0.75,
        div_cur_free_kernels=False,
    ):
        """Initialize the VectorField class.

        Parameters
        ----------
        X: 'np.ndarray' (dimension: n_obs x n_features)
                Original data.
        V: 'np.ndarray' (dimension: n_obs x n_features)
                Velocities of cells in the same order and dimension of X.
        Grid: 'np.ndarray'
                The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
        M: 'int' (default: 100)
                The number of basis functions to approximate the vector field.
        a: `float` (default 5)
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's
            variation space is a.
        beta: `float` (default: 0.1)
             Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
        ecr: `float` (default: 1e-5)
            The minimum limitation of energy change rate in the iteration process.
        gamma: `float` (default:  0.9)
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
            Default value is 0.9.
        lambda_: `float` (default: 3)
            Represents the trade-off between the goodness of data fit and regularization.
        minP: `float` (default: 1e-5)
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as minP.
        MaxIter: `int` (default: 500)
            Maximum iterition times.
        theta: `float` (default 0.75)
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta, then
            it is regarded as an inlier.
        div_cur_free_kernels: `bool` (default: False)
            A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the vector
            field.
        """

        self.data = {"X": X, "V": V, "Grid": Grid}

        self.parameters = {
            "M": M,
            "a": a,
            "beta": beta,
            "ecr": ecr,
            "gamma": gamma,
            "lambda_": lambda_,
            "minP": minP,
            "MaxIter": MaxIter,
            "theta": theta,
            "div_cur_free_kernels": div_cur_free_kernels,
        }
        self.norm_dict = {}

    def fit(self, normalize=False, method="SparseVFC"):
        """Learn an function of vector field from sparse single cell samples in the entire space robustly.
        Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

        Arguments
        ---------
            normalize: 'bool' (default: False)
                Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is often
                required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension). But it is
                normally not required for low dimensional embeddings by PCA or other non-linear dimension reduction methods.
            method: 'string'
                Method that is used to reconstruct the vector field functionally. Currently only SparseVFC supported but other
                improved approaches are under development.

        Returns
        -------
            VecFld: 'dict'
                A dictionary which contains X, Y, beta, V, C, P, VFCIndex. Where V = f(X), P is the posterior probability and
                VFCIndex is the indexes of inliers which found by VFC.
        """

        if normalize:
            X, V, T, norm_dict = norm(self.data["X"], self.data["V"], self.data["Grid"])
            self.data["X"], self.data["V"], self.data["Grid"], self.norm_dict = (
                X,
                V,
                T,
                norm_dict,
            )

        if method == "SparseVFC":
            VecFld = SparseVFC(
                self.data["X"],
                self.data["V"],
                self.data["Grid"],
                M=self.parameters["M"],
                a=self.parameters["a"],
                beta=self.parameters["beta"],
                ecr=self.parameters["ecr"],
                gamma=self.parameters["gamma"],
                lambda_=self.parameters["lambda_"],
                minP=self.parameters["minP"],
                MaxIter=self.parameters["MaxIter"],
                theta=self.parameters["theta"],
            )

        return VecFld

    def evaluate(self, CorrectIndex, VFCIndex, siz):
        """Evaluate the precision, recall, corrRate of the sparseVFC algorithm.

        Arguments
        ---------
            CorrectIndex: 'List'
                Ground truth indexes of the correct vector field samples.
            VFCIndex: 'List'
                Indexes of the correct vector field samples learned by VFC.
            siz: 'int'
                Number of initial matches.

        Returns
        -------
        A tuple of precision, recall, corrRate:
        Precision, recall, corrRate: Precision and recall of VFC, percentage of initial correct matches.

        See also:: :func:`sparseVFC`.
        """

        if len(VFCIndex) == 0:
            VFCIndex = range(siz)

        VFCCorrect = np.intersect1d(VFCIndex, CorrectIndex)
        NumCorrectIndex = len(CorrectIndex)
        NumVFCIndex = len(VFCIndex)
        NumVFCCorrect = len(VFCCorrect)

        corrRate = NumCorrectIndex / siz
        precision = NumVFCCorrect / NumVFCIndex
        recall = NumVFCCorrect / NumCorrectIndex

        print(
            "correct correspondence rate in the original data: %d/%d = %f"
            % (NumCorrectIndex, siz, corrRate)
        )
        print("precision rate: %d/%d = %f" % (NumVFCCorrect, NumVFCIndex, precision))
        print("recall rate: %d/%d = %f" % (NumVFCCorrect, NumCorrectIndex, recall))

        return corrRate, precision, recall

