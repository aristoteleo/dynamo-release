from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from ..dynamo_logger import main_warning


def hill_inh_func(
    x: np.ndarray,
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    g: Union[float, int, np.ndarray],
) -> np.ndarray:
    """The inhibitory Hill equation function.

    Args:
        x: The input transcription factor.
        A: The maximum transcription rate.
        K: Is the ligand concentration producing half occupation.
        n: The Hill coefficient.
        g: The degradation constant as the offset.

    Returns:
        The output of the Hill equation representing the inhibitory effect of the transcription factor.
    """
    Kd = K**n
    return A * Kd / (Kd + x**n) - g * x


def hill_inh_grad(
    x: np.ndarray,
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    g: Union[float, int, np.ndarray],
) -> np.ndarray:
    """The gradient function of the inhibitory Hill equation.

    Args:
        x: The input transcription factor.
        A: The maximum transcription rate.
        K: Is the ligand concentration producing half occupation.
        n: The Hill coefficient.
        g: The degradation constant as the offset.

    Returns:
        The gradient of the Hill equation.
    """
    Kd = K**n
    return -A * n * Kd * x ** (n - 1) / (Kd + x**n) ** 2 - g


def hill_act_func(
    x: np.ndarray,
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    g: Union[float, int, np.ndarray],
) -> np.ndarray:
    """The activation Hill equation function.

    Args:
        x: The input transcription factor.
        A: The maximum transcription rate.
        K: Is the ligand concentration producing half occupation.
        n: The Hill coefficient.
        g: The degradation constant as the offset.

    Returns:
        The output of the Hill equation representing the activation effect of the transcription factor.
    """
    Kd = K**n
    return A * x**n / (Kd + x**n) - g * x


def hill_act_grad(
    x: np.ndarray,
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    g: Union[float, int, np.ndarray],
) -> np.ndarray:
    """The gradient function of the activation Hill equation.

    Args:
        x: The input transcription factor.
        A: The maximum transcription rate.
        K: Is the ligand concentration producing half occupation.
        n: The Hill coefficient.
        g: The degradation constant as the offset.

    Returns:
        The gradient of the Hill equation.
    """
    Kd = K**n
    return A * n * Kd * x ** (n - 1) / (Kd + x**n) ** 2 - g


def calc_mean_squared_deviation(
    func: Callable,
    x_data: np.ndarray,
    y_mean: np.ndarray,
    y_sigm: np.ndarray,
    weighted=True,
) -> float:
    """Calculate the mean squared deviation of the fit.

    Args:
        func: The function to evaluate.
        x_data: An array of the x data.
        y_mean: An array of the mean of y data.
        y_sigm: An array of the standard deviation of the y data.
        weighted: Whether to use weighted mean squared deviation.

    Returns:
        The mean squared deviation.
    """
    err = func(x_data) - y_mean
    if weighted:
        sig = np.array(y_sigm, copy=True)
        if np.any(sig == 0):
            main_warning("Some standard deviations are 0; Set to 1 instead.")
            sig[sig == 0] = 1
        err /= sig
    return np.sqrt(err.dot(err))


def fit_hill_grad(
    x_data: np.ndarray,
    y_mean: np.ndarray,
    type: str,
    y_sigm: Optional[np.ndarray] = None,
    fix_g: Optional[Union[float, int, np.ndarray]] = None,
    n_num: int = 5,
    x_tol: float = 1e-5,
    x_shift: int = 0,
) -> Tuple:
    """Fit the Hill equation to the data.

    This can reveal whether a gene is regulated (activated or suppressed) by certain transcription factor through
    fitting the derivatives of inhibitory or activation Hill equations to the corresponding Jacobian elements.

    Args:
        x_data: An array of the x data.
        y_mean: An array of the mean of y data.
        type: The type of the Hill equation, either `act` or `inh`.
        y_sigm: An array of the standard deviation of the y data.
        fix_g: The degradation constant as the offset.
        n_num: The number of Hill coefficients to try.
        x_tol: The tolerance of the x data.
        x_shift: The shift of the x data.

    Returns:
        The fitted parameters and the mean squared deviation.
    """
    assert type in ["act", "inh"], "`type` must be either `act` or `inh`."

    A0 = -np.min(y_mean) if type == "inh" else np.max(y_mean)
    K0 = (x_data[0] + x_data[-1]) / 2
    g0 = 0 if fix_g is None else fix_g
    logN0 = np.linspace(-2, 2, n_num)

    if x_shift is not None:
        x_data_ = np.array(x_data, copy=True)
        x_data_ += x_shift

    if y_sigm is None:
        sig = None
    else:
        sig = y_sigm[x_data_ > x_tol]

    msd_min = np.inf
    p_opt_min = None
    func = hill_inh_grad if type == "inh" else hill_act_grad

    for logn0 in logN0:
        if fix_g is None:
            fit_func = lambda x, A, K, logn, g: func(x, A, K, np.exp(logn), g)
            p0 = [A0, K0, logn0, g0]
            bounds = [(0, 0, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf)]
        else:
            fit_func = lambda x, A, K, logn: func(x, A, K, np.exp(logn), fix_g)
            p0 = [A0, K0, logn0]
            bounds = [(0, 0, -np.inf), (np.inf, np.inf, np.inf)]

        try:
            p_opt, _ = curve_fit(
                fit_func, x_data_[x_data_ > x_tol], y_mean[x_data_ > x_tol], p0=p0, sigma=sig, bounds=bounds
            )
            if fix_g is None:
                A, K, n, g = p_opt[0], p_opt[1], np.exp(p_opt[2]), p_opt[3]
            else:
                A, K, n, g = p_opt[0], p_opt[1], np.exp(p_opt[2]), fix_g

            msd = calc_mean_squared_deviation(lambda x: func(x, A, K, n, g), x_data_, y_mean, y_sigm)

            if msd < msd_min:
                msd_min = msd
                p_opt_min = [A, K, n, g]

        except:
            pass

    if p_opt_min is None:
        return None, np.inf
    else:
        A, K, n, g = p_opt_min[0], p_opt_min[1], p_opt_min[2], p_opt_min[3]
        msd_min = calc_mean_squared_deviation(lambda x: func(x, A, K, n, g), x_data, y_mean, y_sigm)
        return {"A": A, "K": K, "n": n, "g": g}, msd_min


def fit_hill_inh_grad(
    x_data: np.ndarray,
    y_mean: np.ndarray,
    y_sigm: Optional[np.ndarray] = None,
    n_num: int = 5,
    x_tol: float = 1e-5,
    x_shift: float = 1e-4,
) -> Tuple[Dict, float]:
    """Fit the inhibitory Hill equation to the data.

    Args:
        x_data: An array of the x data.
        y_mean: An array of the mean of y data.
        y_sigm: An array of the standard deviation of the y data.
        n_num: The number of Hill coefficients to try.
        x_tol: The tolerance of the x data.
        x_shift: The shift of the x data.

    Returns:
        The fitted parameters and the mean squared deviation.
    """
    A0 = -np.min(y_mean)
    K0 = (x_data[0] + x_data[-1]) / 2
    g0 = 0
    logN0 = np.linspace(-2, 2, n_num)

    if x_shift is not None:
        x_data = np.array(x_data, copy=True)
        x_data += x_shift

    if y_sigm is None:
        sig = None
    else:
        sig = y_sigm[x_data > x_tol]

    msd_min = np.inf
    p_opt_min = None
    for logn0 in logN0:
        try:
            p_opt, _ = curve_fit(
                lambda x, A, K, logn, g: hill_inh_grad(x, A, K, np.exp(logn), g),
                x_data[x_data > x_tol],
                y_mean[x_data > x_tol],
                p0=[A0, K0, logn0, g0],
                sigma=sig,
                bounds=[(0, 0, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf)],
            )
            A, K, n, g = p_opt[0], p_opt[1], np.exp(p_opt[2]), p_opt[3]
            msd = calc_mean_squared_deviation(lambda x: hill_inh_grad(x, A, K, n, g), x_data, y_mean, y_sigm)

            if msd < msd_min:
                msd_min = msd
                p_opt_min = p_opt

        except:
            pass

    return {"A": p_opt_min[0], "K": p_opt_min[1], "n": np.exp(p_opt_min[2]), "g": p_opt_min[3]}, msd_min


def fit_hill_act_grad(
    x_data: np.ndarray,
    y_mean: np.ndarray,
    y_sigm: Optional[np.ndarray] = None,
    n_num: int = 5,
    x_tol: float = 1e-5,
    x_shift: float = 1e-4,
) -> Tuple[Dict, float]:
    """Fit the activation Hill equation to the data.

    Args:
        x_data: An array of the x data.
        y_mean: An array of the mean of y data.
        y_sigm: An array of the standard deviation of the y data.
        n_num: The number of Hill coefficients to try.
        x_tol: The tolerance of the x data.
        x_shift: The shift of the x data.

    Returns:
        The fitted parameters and the mean squared deviation.
    """
    A0 = np.max(y_mean)
    K0 = (x_data[0] + x_data[-1]) / 2
    g0 = 0
    logN0 = np.linspace(-2, 2, n_num)

    if x_shift is not None:
        x_data = np.array(x_data, copy=True)
        x_data += x_shift

    if y_sigm is None:
        sig = None
    else:
        sig = y_sigm[x_data > x_tol]

    msd_min = np.inf
    p_opt_min = None
    for logn0 in logN0:
        try:
            p_opt, _ = curve_fit(
                lambda x, A, K, logn, g: hill_act_grad(x, A, K, np.exp(logn), g),
                x_data[x_data > x_tol],
                y_mean[x_data > x_tol],
                p0=[A0, K0, logn0, g0],
                sigma=sig,
                bounds=[(0, 0, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf)],
            )
            A, K, n, g = p_opt[0], p_opt[1], np.exp(p_opt[2]), p_opt[3]
            msd = calc_mean_squared_deviation(lambda x: hill_act_grad(x, A, K, n, g), x_data, y_mean, y_sigm)

            if msd < msd_min:
                msd_min = msd
                p_opt_min = p_opt

        except:
            pass

    return {"A": p_opt_min[0], "K": p_opt_min[1], "n": np.exp(p_opt_min[2]), "g": p_opt_min[3]}, msd_min
