from typing import Callable, Union

import numpy as np


def directMethod(prop_fcn: Callable, update_fcn: Callable, tspan, C0, record_skip_steps=0, record_max_length=1e5):
    """Gillespie direct method.

    Parameters
    ----------
        prop_fcn: Callable
            a function that calculates the propensity for each reaction.
            input: an array of copy numbers of all species;
            output: an array of propensities of all reactions.
        update_fcn: Callable
            a function that determines how the copy number of each species increases or decreases after each reaction.
            input: (1) an array of current copy numbers of all species; (2) the index of the occurred reaction.
            output: an array of updated of copy numbers of all species.
        tspan: list
            a list of starting and end simulation time, e.g. [0, 100].
        C0: :class:`~numpy.ndarray`
            transcription rate with active promoter
        record_skip_steps: int
            transcription rate with inactive promoter
        record_max_length: int
            sigma, degradation rate

    Returns
    -------
        retT: :class:`~numpy.ndarray`
            a 1d numpy array of time points.
        retC: :class:`~numpy.ndarray`
            a 2d numpy array (n_species x n_time_points) of copy numbers for each species at each time point.
    """
    retC = np.zeros((len(C0), int(record_max_length)), np.float64)
    retT = np.zeros(int(record_max_length), np.float64)
    c = C0.flatten()
    t = tspan[0]
    retC[:, 0] = c
    retT[0] = t
    count = 0
    count_rec = 0
    while (t <= tspan[-1]) & (count < record_max_length - 1):
        count += 1

        a = prop_fcn(c)
        a0 = sum(a)

        r = np.random.rand(2)
        tau = -np.log(r[0]) / a0
        mu = np.cumsum(a).searchsorted(r[1] * a0)

        t += tau
        c = update_fcn(c, mu)

        if record_skip_steps == 0 or count_rec % record_skip_steps == 0:
            retT[count_rec] = t
            retC[:, count_rec] = c
            count_rec += 1

    retT = retT[:count_rec]
    retC = retC[:, :count_rec]

    return retT, retC


def prop_slam(C, a, b, la, aa, ai, si, be, ga):
    # species
    s = C[0]
    ul = C[1]
    uu = C[2]
    sl = C[3]
    su = C[4]

    # propensities
    prop = np.zeros(11)
    if s > 0:  # promoter is active
        prop[0] = a  # A -> I
        prop[2] = la * aa  # A --> ul
        prop[3] = (1 - la) * aa  # A --> uu
    else:  # promoter is inactive
        prop[1] = b  # I -> A
        prop[4] = la * ai  # I --> ul
        prop[5] = (1 - la) * ai  # I --> uu

    prop[6] = (1 - si) * be * ul  # ul -> sl
    prop[7] = si * be * ul  # ul -> su
    prop[8] = be * uu  # uu -> su
    prop[9] = ga * sl  # sl -> 0
    prop[10] = ga * su  # su -> 0

    return prop


def simulate_Gillespie(a, b, la, aa, ai, si, be, ga, C0, t_span, n_traj, report=False):
    # species
    s = 0
    u_l = 1
    u_u = 2
    s_l = 3
    s_u = 4
    n_species = 5

    # stoichiometry matrix
    stoich = np.zeros((11, n_species))
    stoich[0, s] = -1  # A -> I
    stoich[1, s] = 1  # I -> A
    stoich[2, u_l] = 1  # A --> u_l
    stoich[3, u_u] = 1  # A --> u_u
    stoich[4, u_l] = 1  # I --> u_l
    stoich[5, u_u] = 1  # I --> u_u
    stoich[6, u_l] = -1  # u_l --> s_l
    stoich[6, s_l] = 1
    stoich[7, u_l] = -1  # u_l --> s_u
    stoich[7, s_u] = 1
    stoich[8, u_u] = -1  # u_u --> s_u
    stoich[8, s_u] = 1
    stoich[9, s_l] = -1  # s_l --> 0
    stoich[10, s_u] = -1  # s_u --> 0
    update_func = lambda C, mu: C + stoich[mu, :]

    trajs_T = [[]] * n_traj
    trajs_C = [[]] * n_traj

    for i in range(n_traj):
        T, C = directMethod(
            lambda C: prop_slam(C, a, b, la, aa, ai, si, be, ga),
            update_func,
            t_span,
            C0[i],
        )
        trajs_T[i] = T
        trajs_C[i] = C
        if report:
            print("Iteration %d/%d finished." % (i + 1, n_traj), end="\r")
    return trajs_T, trajs_C


def prop_2bifurgenes(C, a, b, S, K, m, n, gamma):
    # species
    x = C[0]
    y = C[1]

    # parameters
    a1, a2 = a[0], a[1]
    b1, b2 = b[0], b[1]
    S1, S2 = S[0], S[1]
    K1, K2 = K[0], K[1]
    m1, m2 = m[0], m[1]
    n1, n2 = n[0], n[1]
    ga1, ga2 = gamma[0], gamma[1]

    # propensities
    prop = np.zeros(4)
    prop[0] = a1 * x ** m1 / (S1 ** m1 + x ** m1) + b1 * K1 ** n1 / (K1 ** n1 + y ** n1)  # 0 -> x
    prop[1] = ga1 * x  # x -> 0
    prop[2] = a2 * y ** m2 / (S2 ** m2 + y ** m2) + b2 * K2 ** n2 / (K2 ** n2 + x ** n2)  # 0 -> y
    prop[3] = ga2 * y  # y -> 0

    return prop


def stoich_2bifurgenes():
    # species
    x = 0
    y = 1

    # stoichiometry matrix
    stoich = np.zeros((4, 2))
    stoich[0, x] = 1  # 0 -> x
    stoich[1, x] = -1  # x -> 0
    stoich[2, y] = 1  # 0 -> y
    stoich[3, y] = -1  # y -> 0

    return stoich


def prop_2bifurgenes_splicing(C, a, b, S, K, m, n, beta, gamma):
    # species
    u1 = C[0]
    s1 = C[1]
    u2 = C[2]
    s2 = C[3]

    # parameters
    a1, a2 = a[0], a[1]
    b1, b2 = b[0], b[1]
    S1, S2 = S[0], S[1]
    K1, K2 = K[0], K[1]
    m1, m2 = m[0], m[1]
    n1, n2 = n[0], n[1]
    be1, be2 = beta[0], beta[1]
    ga1, ga2 = gamma[0], gamma[1]

    # propensities
    prop = np.zeros(6)
    prop[0] = a1 * s1 ** m1 / (S1 ** m1 + s1 ** m1) + b1 * K1 ** n1 / (K1 ** n1 + s2 ** n1)  # 0 -> u1
    prop[1] = be1 * u1  # u1 -> s1
    prop[2] = ga1 * s1  # s1 -> 0
    prop[3] = a2 * s2 ** m2 / (S2 ** m2 + s2 ** m2) + b2 * K2 ** n2 / (K2 ** n2 + s1 ** n2)  # 0 -> u2
    prop[4] = be2 * u2  # u2 -> s2
    prop[5] = ga2 * s2  # s2 -> 0

    return prop


def stoich_2bifurgenes_splicing():
    # species
    u1 = 0
    s1 = 1
    u2 = 2
    s2 = 3

    # stoichiometry matrix
    stoich = np.zeros((6, 4))
    stoich[0, u1] = 1  # 0 -> u1
    stoich[1, u1] = -1  # u1 -> s1
    stoich[1, s1] = 1
    stoich[2, s1] = -1  # s1 -> 0
    stoich[3, u2] = 1  # 0 -> u2
    stoich[4, u2] = -1  # u2 -> s2
    stoich[4, s2] = 1
    stoich[5, s2] = -1  # s2 -> 0

    return stoich


def simulate_2bifurgenes(C0, t_span, n_traj, param_dict, report=False, **gillespie_kwargs):
    param_dict = param_dict.copy()
    beta = param_dict.pop("beta", None)
    if beta is None:
        stoich = stoich_2bifurgenes()
    else:
        stoich = stoich_2bifurgenes_splicing()

    update_func = lambda C, mu: C + stoich[mu, :]

    trajs_T = [[]] * n_traj
    trajs_C = [[]] * n_traj

    if beta is None:
        prop_func = lambda C: prop_2bifurgenes(C, **param_dict)
    else:
        prop_func = lambda C: prop_2bifurgenes_splicing(C, beta=beta, **param_dict)

    for i in range(n_traj):
        T, C = directMethod(prop_func, update_func, t_span, C0, **gillespie_kwargs)
        trajs_T[i] = T
        trajs_C[i] = C
        if report:
            print("Iteration %d/%d finished." % (i + 1, n_traj), end="\r")
    return trajs_T, trajs_C


def temporal_average(t, trajs_T, trajs_C, species, f=lambda x: x):
    n = len(trajs_T)
    y = np.zeros((n, len(t)))
    for i in range(n):
        T = trajs_T[i]
        X = trajs_C[i][species, :]
        vq = np.interp(t, T, X)
        y[i] = f(vq)
    return np.nanmean(y, 0)


def temporal_cov(t, trajs_T, trajs_C, species1, species2):
    n = len(trajs_T)
    y = np.zeros((n, len(t)))
    for i in range(n):
        T = trajs_T[i]
        X1 = trajs_C[i][species1, :]
        X2 = trajs_C[i][species2, :]
        vq1 = np.interp(t, T, X1)
        vq2 = np.interp(t, T, X2)
        y[i] = vq1 * vq2
    return np.nanmean(y, 0)


def temporal_interp(t, trajs_T, trajs_C, round=False):
    n = len(trajs_T)
    ret = []
    for i in range(n):
        T = trajs_T[i]
        X = trajs_C[i]
        y = np.zeros((len(X), len(t)))
        for j in range(len(X)):
            y[j] = np.interp(t, T, X[j])
            if round:
                y[j] = np.round(y[j])
        ret.append(y)
    return np.array(ret)


def convert_nosplice(trajs_T, trajs_C):
    trajs_C_nosplice = []
    for i in range(len(trajs_T)):
        traj_temp = np.zeros((2, len(trajs_T[i])))
        traj_temp[0] = trajs_C[i][1] + trajs_C[i][3]  # labeled = uu + su
        traj_temp[1] = trajs_C[i][2] + trajs_C[i][4]  # unlabeled = ul + sl
        trajs_C_nosplice.append(traj_temp)
    return trajs_C_nosplice


def simulate_multigene(a, b, la, aa, ai, si, be, ga, C0, t_span, n_traj, t_eval, report=False):
    n_genes = len(a)
    ret = []
    for i in range(n_genes):
        trajs_T, trajs_C = simulate_Gillespie(
            a[i],
            b[i],
            la[i],
            aa[i],
            ai[i],
            si[i],
            be[i],
            ga[i],
            C0[i],
            t_span,
            n_traj,
            report,
        )
        trajs_C_interp = temporal_interp(t_eval, trajs_T, trajs_C)
        ret.append(trajs_C_interp)
    return np.array(ret)


class trajectories:
    def __init__(self, trajs_T=[], trajs_C=[]):
        # species
        self.trajs_T = trajs_T
        self.trajs_C = trajs_C

    def append(self, traj_T, traj_C):
        self.trajs_T.append(traj_T)
        self.trajs_C.append(traj_C)

    def interpolate(self, T, round=False):
        return temporal_interp(T, self.trajs_T, self.trajs_C, round)
