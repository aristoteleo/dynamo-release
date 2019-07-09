import numpy as np
import scipy
import numpy.matlib


def SparseVFC(X, Y, Grid, M = 100, a = 5, beta = 0.1, ecr = 1e-5, gamma = 0.9, lambda_ = 3, minP = 1e-5, MaxIter = 500, theta = 0.75, div_cur_free_kernels = False):
    '''Apply sparseVFC (vector field consensus) algorithm to learn an analytical function of vector field on the entire space robustly.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

    Arguments
    ---------
        X: 'np.ndarray'
            Current state. This corresponds to, for example, the spliced transcriptomic state.
        Y: 'np.ndarray'
            Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic velocity estimated calculated by velocyto, scvelo or dynamo.
        Grid: 'np.ndarray'
            Current state on a grid which is often used to visualize the vector field. This corresponds to, for example, the spliced transcriptomic state.
        M: 'np.ndarray'
            The number of basis functions to approximate the vector field. By default, it is 100.
        a: 'float'
            Paramerter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of outlier's variation space is a. Default Value is 10.
        beta: 'float'
            Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2), Default value is 0.1.
        ecr: 'float'
            The minimum limitation of energy change rate in the iteration process. Default value is 1e-5.
        gamma: 'float'
            Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important. Default value is 0.9.
        lambda_: 'float'
            Represents the trade-off between the goodness of data fit and regularization. Default value is 3.
        minP: 'float'
            The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as minP. Default value is 1e-5.
        MaxIter: 'int'
            Maximum iterition times. Defualt value is 500.
        theta: 'float'
            Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta, then it is regarded as an inlier. Default value is 0.75.

    Returns
    -------
    VecFld: 'dict'
    A dictionary which contains X, Y, beta, V, C, P, VFCIndex. Where V = f(X), P is the posterior probability and
    VFCIndex is the indexes of inliers which found by VFC. Note that V = con_K(Grid, ctrl_pts, beta).dot(C) gives the prediction of velocity on Grid (can be any point in the gene expressionstate space).
    '''

    N, D = Y.shape

    # Construct kernel matrix K
    M = 500 if M is None else M
    tmp_X = np.unique(X, axis = 0) # return unique rows
    idx = np.random.RandomState(seed=0).permutation(tmp_X.shape[0]) # rand select some intial points
    idx = idx[range(min(M, tmp_X.shape[0]))]
    ctrl_pts = tmp_X[idx, :]
    # ctrl_pts = X[range(500), :]

    K = con_K(ctrl_pts, ctrl_pts, beta) if div_cur_free_kernels is False else con_K_div_cur_free(ctrl_pts, ctrl_pts)[0]
    U = con_K(X, ctrl_pts, beta) if div_cur_free_kernels is False else con_K_div_cur_free(X, ctrl_pts)[0]
    grid_U = con_K(Grid, ctrl_pts, beta) if div_cur_free_kernels is False else con_K_div_cur_free(Grid, ctrl_pts)[0]
    M = ctrl_pts.shape[0]

    # Initialization
    V = np.zeros((N, D))
    C = np.zeros((M, D))
    iter, tecr, E = 1, 1, 1
    sigma2 = sum(sum((Y - V)**2)) / (N * D) ## test this

    while iter < MaxIter and tecr > ecr and sigma2 > 1e-8:
        # E_step
        E_old = E
        P, E = get_P(Y, V, sigma2, gamma, a)

        E = E + lambda_ / 2 * scipy.trace(C.T.dot(K).dot(C))
        tecr = abs((E - E_old) / E)

        # print('iterate: {}, gamma: {}, the energy change rate: {}, sigma2={}\n'.format(*[iter, gamma, tecr, sigma2]))

        # M-step. Solve linear system for C.
        P = scipy.maximum(P, minP)
        C = scipy.linalg.lstsq(((U.T * numpy.matlib.repmat(P.T, M, 1)).dot(U) + lambda_ * sigma2 * K), \
                               (U.T * numpy.matlib.repmat(P.T, M, 1)).dot(Y))[0]

        # Update V and sigma**2
        V = U.dot(C)
        Sp = sum(P)
        sigma2 = sum(P.T * np.sum((Y - V)**2, 1)) / np.dot(Sp, D)

        # Update gamma
        numcorr = len(np.where(P > theta)[0])
        gamma = numcorr / X.shape[0]

        if gamma > 0.95:
            gamma = 0.95
        elif gamma < 0.05:
            gamma = 0.05

        iter += 1

    grid_V = np.dot(grid_U, C)

    VecFld = {"X": ctrl_pts, "Y": Y, "beta": beta, "V": V, "C": C , "P": P, "VFCIndex": np.where(P > theta)[0], "sigma2": sigma2, "grid": Grid, "grid_V": grid_V}

    return VecFld


def con_K(x, y, beta):
    '''Con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Arguments
    ---------
        x: 'np.ndarray'
            Original training data points.
        y: 'np.ndarray'
            Control points used to build kernel basis functions.
        beta: 'np.ndarray'
            The function that returns diffusion matrix which can be dependent on the variables (for example, genes)

    Returns
    -------
    K: 'np.ndarray'
    the kernel to represent the vector field function.
    '''

    n, d = x.shape
    m, d = y.shape

    # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
    # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
    K = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
    K = np.squeeze(np.sum(K**2, 1))
    K = - beta * K
    K = np.exp(K) #

    return K


def get_P(Y, V, sigma2, gamma, a):
    '''GET_P estimates the posterior probability and part of the energy.

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

    '''
    D = Y.shape[1]
    temp1 = np.exp(-np.sum((Y - V)**2, 1) / (2 * sigma2))
    temp2 = (2 * np.pi * sigma2)**(D/2) * (1 - gamma) / (gamma * a)
    P = temp1 / (temp1 + temp2)
    E = P.T.dot(np.sum((Y - V)**2, 1)) / (2 * sigma2) + np.sum(P) * np.log(sigma2) * D / 2

    return P, E

def VectorField(X, Y, Grid, M = None, method = 'SparseVFC'):
    '''Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

    Arguments
    ---------
        X: 'np.ndarray'
            Original data.
        Y: 'np.ndarray'
            Original data.
        Grid: 'np.ndarray'
            The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
        M: 'function'
            The number of basis functions to approximate the vector field. By default, it is 100.
        method: 'str'
            Method that is used to reconstruct the vector field analytically. Currently only SparseVFC supported but other
            improved approaches are under development.

    Returns
    -------
    VecFld: 'dict'
    A dictionary which contains X, Y, beta, V, C, P, VFCIndex. Where V = f(X), P is the posterior probability and
    VFCIndex is the indexes of inliers which found by VFC.
    '''

    if(method == 'SparseVFC'):
        VecFld = SparseVFC(X, Y, Grid, M = M, a = 5, beta = 0.1, ecr = 1e-5, gamma = 0.9, lambda_ = 3, minP = 1e-5, MaxIter = 500, theta = 0.75)

    return VecFld


def evaluate(CorrectIndex, VFCIndex, siz):
    '''Evaluate the precision, recall, corrRate of the sparseVFC algorithm.

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
    '''

    if len(VFCIndex) == 0:
        VFCIndex = range(siz)

    VFCCorrect = np.intersect1d(VFCIndex, CorrectIndex)
    NumCorrectIndex = len(CorrectIndex)
    NumVFCIndex = len(VFCIndex)
    NumVFCCorrect = len(VFCCorrect)

    corrRate = NumCorrectIndex/siz
    precision = NumVFCCorrect/NumVFCIndex
    recall = NumVFCCorrect/NumCorrectIndex

    print('correct correspondence rate in the original data: %d/%d = %f' % (NumCorrectIndex, siz, corrRate))
    print('precision rate: %d/%d = %f'% (NumVFCCorrect, NumVFCIndex, precision))
    print('recall rate: %d/%d = %f' % (NumVFCCorrect, NumCorrectIndex, recall))

    return corrRate, precision, recall


def con_K_div_cur_free(x, y, sigma = 0.8, gamma = 0.5):
    '''Learn a convex combination of the divergence-free kernel T_df and curl-free kernel T_cf with a bandwidth sigma and a combination coefficient gamma.

    Arguments
    ---------
        x: 'np.ndarray'
            Original training data points.
        y: 'np.ndarray'
            Control points used to build kernel basis functions
        sigma: 'int'
            Bandwidth parameter.
        sigma: 'int'
            Combination coefficient for the divergence-free or the curl-free kernels.

    Returns
    -------
    K: 'np.ndarray'
    the kernel to represent the vector field function.
    Returns
    -------
    A tuple of G (the combined kernel function), divergence-free kernel and curl-free kernel.

    See also:: :func:`sparseVFC`.
    '''
    m, d = x.shape; n, d = y.shape
    sigma2 = sigma**2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0])
    G_tmp = np.squeeze(np.sum(K**2, 1))
    G_tmp3 = - G_tmp / sigma2
    G_tmp = -G_tmp/(2*sigma2)
    G_tmp = np.exp(G_tmp)/sigma2
    G_tmp = np.kron(G_tmp, np.ones(d))

    x_tmp = np.matlib.tile(x,[n, 1])
    y_tmp = np.matlib.tile(y,[1, m]).T
    y_tmp = y_tmp.reshape((d,m*n)).T
    xminusy = (x_tmp-y_tmp)
    G_tmp2 = np.zeros(d*m, d*n)

    for i in range(d):
        for j in range(d):
            tmp1 = xminusy[:, i].reshape((m, n))
            tmp2 = xminusy[:, j].reshape((m, n))
            tmp3 = tmp1 * tmp2
            tmp4 = np.zeros(d)
            tmp4[i, j] = 1; tmp4[j, i] = 1
            G_tmp2 = G_tmp2 + np.kron(tmp3, tmp4)

    G_tmp2 = G_tmp2/sigma2
    G_tmp3 = np.kron((G_tmp3+d-1), np.eye(d))
    G_tmp4 = np.kron(np.ones(m,n),np.eye(d))-G_tmp2
    G = (1-gamma)*G_tmp*(G_tmp2+G_tmp3)+gamma*G_tmp*G_tmp4

    return G, (1-gamma)*G_tmp*(G_tmp2+G_tmp3), gamma*G_tmp*G_tmp4

def vector_field_function(x, VecFld, autograd = False):
    '''Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
    '''

    K= con_K(x, VecFld['X'], VecFld['beta']) if autograd is False else auto_con_K(x, VecFld['X'], VecFld['beta'])

    K = K.dot(VecFld['C']).T

    return K

def vector_field_function_auto(x, VecFld, autograd = False):
    '''Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.
    Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
    '''

    K= con_K(x, VecFld['X'], VecFld['beta']) if autograd is False else auto_con_K(x, VecFld['X'], VecFld['beta'])

    K = K.dot(VecFld['C']).T

    return K

def auto_con_K(x, y, beta):
    '''Con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Arguments
    ---------
        x: 'np.ndarray'
            Original training data points.
        y: 'np.ndarray'
            control points used to build kernel basis functions
        beta: 'np.ndarray'
            The function that returns diffusion matrix which can be dependent on the variables (for example, genes)

    Returns
    -------
    K: 'np.ndarray'
    the kernel to represent the vector field function.
    '''

    n, d = x.shape
    m, d = y.shape

    # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
    # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
    K = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
    K = np.squeeze(np.sum(K**2, 1))
    K = - beta * K
    K = np.exp(K) #

    return K
