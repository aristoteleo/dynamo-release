import numpy as np
from .gillespie_utils import directMethod, temporal_interp


# Differentiation model
class sim_diff:
    def __init__(
        self,
        a1,
        b1,
        c1,
        a2,
        b2,
        c2,
        a1_l,
        b1_l,
        c1_l,
        a2_l,
        b2_l,
        c2_l,
        K,
        n,
        be1,
        ga1,
        et1,
        de1,
        be2,
        ga2,
        et2,
        de2,
    ):
        self.parameters = {
            "a1": a1,
            "b1": b1,
            "c1": c1,
            "a2": a2,
            "b2": b2,
            "c2": c2,
            "a1_l": a1_l,
            "b1_l": b1_l,
            "c1_l": c1_l,
            "a2_l": a2_l,
            "b2_l": b2_l,
            "c2_l": c2_l,
            "K": K,
            "n": n,
            "be1": be1,
            "ga1": ga1,
            "et1": et1,
            "de1": de1,
            "be2": be2,
            "ga2": ga2,
            "et2": et2,
            "de2": de2,
        }

    def f_prop(self, C):
        # unlabeled mRNA
        u1 = C[0]
        s1 = C[1]
        u2 = C[2]
        s2 = C[3]

        # labeled mRNA
        w1 = C[4]
        l1 = C[5]
        w2 = C[6]
        l2 = C[7]

        # protein
        p1 = C[8]
        p2 = C[9]

        # parameters
        a1 = self.parameters["a1"]
        a2 = self.parameters["a2"]
        b1 = self.parameters["b1"]
        b2 = self.parameters["b2"]
        c1 = self.parameters["c1"]
        c2 = self.parameters["c2"]
        a1_l = self.parameters["a1_l"]
        a2_l = self.parameters["a2_l"]
        b1_l = self.parameters["b1_l"]
        b2_l = self.parameters["b2_l"]
        c1_l = self.parameters["c1_l"]
        c2_l = self.parameters["c2_l"]
        K = self.parameters["K"]
        n = self.parameters["n"]

        # propensities
        prop = np.zeros(18)
        # transcription
        prop[0] = (
            a1 * p1 ** n / (K ** n + p1 ** n) + b1 * K ** n / (K ** n + p2 ** n) + c1
        )  # 0 -> u1
        prop[1] = (
            a2 * p2 ** n / (K ** n + p2 ** n) + b2 * K ** n / (K ** n + p1 ** n) + c2
        )  # 0 -> u2
        prop[2] = (
            a1_l * p1 ** n / (K ** n + p1 ** n)
            + b1_l * K ** n / (K ** n + p2 ** n)
            + c1_l
        )  # 0 -> w1
        prop[3] = (
            a2_l * p2 ** n / (K ** n + p2 ** n)
            + b2_l * K ** n / (K ** n + p1 ** n)
            + c2_l
        )  # 0 -> w2
        # splicing
        prop[4] = self.parameters["be1"] * u1  # u1 -> s1
        prop[5] = self.parameters["be2"] * u2  # u2 -> s2
        prop[6] = self.parameters["be1"] * w1  # w1 -> l1
        prop[7] = self.parameters["be2"] * w2  # w2 -> l2
        # mRNA degradation
        prop[8] = self.parameters["ga1"] * s1  # s1 -> 0
        prop[9] = self.parameters["ga2"] * s2  # s2 -> 0
        prop[10] = self.parameters["ga1"] * l1  # l1 -> 0
        prop[11] = self.parameters["ga2"] * l2  # l2 -> 0
        # translation
        prop[12] = self.parameters["et1"] * s1  # s1 --> p1
        prop[13] = self.parameters["et2"] * s2  # s2 --> p2
        prop[14] = self.parameters["et1"] * l1  # l1 --> p1
        prop[15] = self.parameters["et2"] * l2  # l2 --> p2
        # protein degradation
        prop[16] = self.parameters["de1"] * p1  # p1 -> 0
        prop[17] = self.parameters["de2"] * p2  # p2 -> 0

        return prop

    def f_stoich(self):
        # species
        u1 = 0
        s1 = 1
        u2 = 2
        s2 = 3
        w1 = 4
        l1 = 5
        w2 = 6
        l2 = 7
        p1 = 8
        p2 = 9

        # stoichiometry matrix
        # transcription
        stoich = np.zeros((18, 10))
        stoich[0, u1] = 1  # 0 -> u1
        stoich[1, u2] = 1  # 0 -> u2
        stoich[2, w1] = 1  # 0 -> w1
        stoich[3, w2] = 1  # 0 -> w2
        # splicing
        stoich[4, u1] = -1  # u1 -> s1
        stoich[4, s1] = 1
        stoich[5, u2] = -1  # u2 -> s2
        stoich[5, s2] = 1
        stoich[6, w1] = -1  # w1 -> l1
        stoich[6, l1] = 1
        stoich[7, w2] = -1  # w2 -> l2
        stoich[7, l2] = 1
        # mRNA degradation
        stoich[8, s1] = -1  # s1 -> 0
        stoich[9, s2] = -1  # s2 -> 0
        stoich[10, l1] = -1  # l1 -> 0
        stoich[11, l2] = -1  # l2 -> 0
        # translation
        stoich[12, p1] = 1  # s1 --> p1
        stoich[13, p2] = 1  # s2 --> p2
        stoich[14, p1] = 1  # l1 --> p1
        stoich[15, p2] = 1  # l2 --> p2
        # protein degradation
        stoich[16, p1] = -1  # p1 -> 0
        stoich[17, p2] = -1  # p2 -> 0

        return stoich


# Oscillator
class sim_osc:
    def __init__(
        self,
        a1,
        b1,
        a2,
        b2,
        a1_l,
        b1_l,
        a2_l,
        b2_l,
        K,
        n,
        be1,
        ga1,
        et1,
        de1,
        be2,
        ga2,
        et2,
        de2,
    ):
        self.parameters = {
            "a1": a1,
            "b1": b1,
            "a2": a2,
            "b2": b2,
            "a1_l": a1_l,
            "b1_l": b1_l,
            "a2_l": a2_l,
            "b2_l": b2_l,
            "K": K,
            "n": n,
            "be1": be1,
            "ga1": ga1,
            "et1": et1,
            "de1": de1,
            "be2": be2,
            "ga2": ga2,
            "et2": et2,
            "de2": de2,
        }

    def f_prop(self, C):
        # unlabeled mRNA
        u1 = C[0]
        s1 = C[1]
        u2 = C[2]
        s2 = C[3]

        # labeled mRNA
        w1 = C[4]
        l1 = C[5]
        w2 = C[6]
        l2 = C[7]

        # protein
        p1 = C[8]
        p2 = C[9]

        # parameters
        K = self.parameters["K"]
        n = self.parameters["n"]

        # propensities
        prop = np.zeros(18)
        # transcription
        uw1 = u1 + w1
        uw2 = u2 + w2
        prop[0] = self.parameters["a1"] * uw1 ** n / (
            K ** n + uw1 ** n
        ) + self.parameters["b1"] * K ** n / (
            K ** n + uw2 ** n
        )  # 0 -> u1
        prop[1] = self.parameters["a2"] * uw2 ** n / (
            K ** n + uw2 ** n
        ) + self.parameters["b2"] * uw1 ** n / (
            K ** n + uw1 ** n
        )  # 0 -(u1 u2)-> u2
        prop[2] = self.parameters["a1_l"] * uw1 ** n / (
            K ** n + uw1 ** n
        ) + self.parameters["b1_l"] * K ** n / (
            K ** n + uw2 ** n
        )  # 0 -> u1
        prop[3] = self.parameters["a2_l"] * uw2 ** n / (
            K ** n + uw2 ** n
        ) + self.parameters["b2_l"] * uw1 ** n / (
            K ** n + uw1 ** n
        )  # 0 -(u1 u2)-> u2
        # splicing
        prop[4] = self.parameters["be1"] * u1  # u1 -> s1
        prop[5] = self.parameters["be2"] * u2  # u2 -> s2
        prop[6] = self.parameters["be1"] * w1  # w1 -> l1
        prop[7] = self.parameters["be2"] * w2  # w2 -> l2
        # mRNA degradation
        prop[8] = self.parameters["ga1"] * s1  # s1 -> 0
        prop[9] = self.parameters["ga2"] * s2  # s2 -> 0
        prop[10] = self.parameters["ga1"] * l1  # l1 -> 0
        prop[11] = self.parameters["ga2"] * l2  # l2 -> 0
        # translation
        prop[12] = self.parameters["et1"] * s1  # s1 --> p1
        prop[13] = self.parameters["et2"] * s2  # s2 --> p2
        prop[14] = self.parameters["et1"] * l1  # l1 --> p1
        prop[15] = self.parameters["et2"] * l2  # l2 --> p2
        # protein degradation
        prop[16] = self.parameters["de1"] * p1  # p1 -> 0
        prop[17] = self.parameters["de2"] * p2  # p2 -> 0

        return prop

    def f_stoich(self):
        # species
        u1 = 0
        s1 = 1
        u2 = 2
        s2 = 3
        w1 = 4
        l1 = 5
        w2 = 6
        l2 = 7
        p1 = 8
        p2 = 9

        # stoichiometry matrix
        # transcription
        stoich = np.zeros((18, 10))
        stoich[0, u1] = 1  # 0 -> u1
        stoich[1, u2] = 1  # 0 -> u2
        stoich[2, w1] = 1  # 0 -> w1
        stoich[3, w2] = 1  # 0 -> w2
        # splicing
        stoich[4, u1] = -1  # u1 -> s1
        stoich[4, s1] = 1
        stoich[5, u2] = -1  # u2 -> s2
        stoich[5, s2] = 1
        stoich[6, w1] = -1  # w1 -> l1
        stoich[6, l1] = 1
        stoich[7, w2] = -1  # w2 -> l2
        stoich[7, l2] = 1
        # mRNA degradation
        stoich[8, s1] = -1  # s1 -> 0
        stoich[9, s2] = -1  # s2 -> 0
        stoich[10, l1] = -1  # l1 -> 0
        stoich[11, l2] = -1  # l2 -> 0
        # translation
        stoich[12, p1] = 1  # s1 --> p1
        stoich[13, p2] = 1  # s2 --> p2
        stoich[14, p1] = 1  # l1 --> p1
        stoich[15, p2] = 1  # l2 --> p2
        # protein degradation
        stoich[16, p1] = -1  # p1 -> 0
        stoich[17, p2] = -1  # p2 -> 0

        return stoich


def simulate(model, C0, t_span, n_traj, report=False):
    stoich = model.f_stoich()
    update_func = lambda C, mu: C + stoich[mu, :]

    trajs_T = [[]] * n_traj
    trajs_C = [[]] * n_traj

    for i in range(n_traj):
        T, C = directMethod(model.f_prop, update_func, t_span, C0[i])
        trajs_T[i] = T
        trajs_C[i] = C
        if report:
            print("Iteration %d/%d finished." % (i + 1, n_traj), end="\r")
    return trajs_T, trajs_C


# synthesize labeling data (kinetics) at different time points (multi-time-series)
def syn_kin_data(model_lab, n_trajs, t_idx, trajs_CP, Tl, n_cell):
    C0 = [trajs_CP[j][:, t_idx] for j in range(n_trajs)]
    # label for 1 unit of time
    trajs_T, trajs_C = simulate(
        model_lab, C0=C0, t_span=[0, 1], n_traj=n_cell, report=True
    )
    # interpolate labeling data
    trajs_C = temporal_interp(Tl, trajs_T, trajs_C, round=True)
    gene_num = 2
    uu_kin, su_kin, ul_kin, sl_kin, pr_kin = (
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
    )
    for i, t in enumerate(Tl):
        u = [trajs_C[j][(0, 2), i] for j in range(n_trajs)]
        s = [trajs_C[j][(1, 3), i] for j in range(n_trajs)]
        w = [trajs_C[j][(4, 6), i] for j in range(n_trajs)]
        l = [trajs_C[j][(5, 7), i] for j in range(n_trajs)]
        p = [trajs_C[j][(8, 9), -1] for j in range(n_trajs)]

        (
            uu_kin[(i * n_trajs) : ((i + 1) * n_trajs), :],
            su_kin[(i * n_trajs) : ((i + 1) * n_trajs), :],
            ul_kin[(i * n_trajs) : ((i + 1) * n_trajs), :],
            sl_kin[(i * n_trajs) : ((i + 1) * n_trajs), :],
            pr_kin[(i * n_trajs) : ((i + 1) * n_trajs), :],
        ) = (np.array(u), np.array(s), np.array(w), np.array(l), np.array(p))

    return uu_kin, su_kin, ul_kin, sl_kin, pr_kin


# synthesize labeling data (degradation) at the begining and the end
def syn_deg_data(model_lab, model_unlab, n_trajs, t_idx, trajs_CP, Tl, n_cell):
    C0 = [trajs_CP[j][:, t_idx] for j in range(n_trajs)]
    # label for 10 unit of time
    trajs_T, trajs_C = simulate(
        model_lab, C0=C0, t_span=[0, 10], n_traj=n_cell, report=True
    )
    # stop labeling, and detect at t = 0, 1, 2, 4, 8
    C0 = [trajs_C[j][:, -1] for j in range(n_trajs)]
    trajs_T, trajs_C = simulate(
        model_unlab, C0=C0, t_span=[0, 10], n_traj=n_cell, report=True
    )
    # interpolate labeling data
    trajs_C = temporal_interp(Tl, trajs_T, trajs_C, round=True)
    gene_num = 2
    uu_deg, su_deg, ul_deg, sl_deg, pr_deg = (
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
        np.zeros((len(Tl) * n_trajs, gene_num)),
    )
    for i, t in enumerate(Tl):
        u = [trajs_C[j][(0, 2), i] for j in range(n_trajs)]
        s = [trajs_C[j][(1, 3), i] for j in range(n_trajs)]
        w = [trajs_C[j][(4, 6), i] for j in range(n_trajs)]
        l = [trajs_C[j][(5, 7), i] for j in range(n_trajs)]
        p = [trajs_C[j][(8, 9), -1] for j in range(n_trajs)]

        (
            uu_deg[(i * n_trajs) : ((i + 1) * n_trajs), :],
            su_deg[(i * n_trajs) : ((i + 1) * n_trajs), :],
            ul_deg[(i * n_trajs) : ((i + 1) * n_trajs), :],
            sl_deg[(i * n_trajs) : ((i + 1) * n_trajs), :],
            pr_deg[(i * n_trajs) : ((i + 1) * n_trajs), :],
        ) = (np.array(u), np.array(s), np.array(w), np.array(l), np.array(p))

    return uu_deg, su_deg, ul_deg, sl_deg, pr_deg


def osc_diff_dup(n_species, trajs_C, model_lab, model_unlab, n_cell):

    n_trajs = len(trajs_C)

    # get the steady state expression (cell state at the final time point) for the cells
    C0 = []
    for i in range(n_trajs):
        c0 = np.zeros(n_species)
        for j in range(n_species):
            c0[j] = trajs_C[i][j, -1]
        C0.append(c0)

    # synthesize data after treatment
    trajs_T, trajs_C = simulate(
        model_unlab, C0=C0, t_span=[0, 410], n_traj=n_cell, report=True
    )

    # interpolate checkpoint data
    T_CP = np.array([0, 5, 10, 40, 100, 200, 300, 400])
    trajs_CP = temporal_interp(T_CP, trajs_T, trajs_C, round=True)

    # synthesize labeling data (one-shot) at each checkpoint
    t_lab, gene_num = 1.0, 2
    Tl = np.array([0, 0.1, 0.2, 0.4, 0.8])
    kin_5, kin_40, kin_200, kin_300 = (
        syn_kin_data(model_lab, n_trajs, 1, trajs_CP, Tl, n_cell),
        syn_kin_data(model_lab, n_trajs, 3, trajs_CP, Tl, n_cell),
        syn_kin_data(model_lab, n_trajs, 5, trajs_CP, Tl, n_cell),
        syn_kin_data(model_lab, n_trajs, 6, trajs_CP, Tl, n_cell),
    )
    # synthesize labeling data (one-shot) at each checkpoint
    uu_one_shot, su_one_shot, ul_one_shot, sl_one_shot, pr_one_shot = (
        np.zeros((n_cell * len(T_CP), gene_num)),
        np.zeros((n_cell * len(T_CP), gene_num)),
        np.zeros((n_cell * len(T_CP), gene_num)),
        np.zeros((n_cell * len(T_CP), gene_num)),
        np.zeros((n_cell * len(T_CP), gene_num)),
    )
    for i, t_cp in enumerate(T_CP):
        C0 = [trajs_CP[j][:, i] for j in range(n_trajs)]
        trajs_T, trajs_C = simulate(
            model_lab, C0=C0, t_span=[0, t_lab], n_traj=n_cell, report=True
        )
        u = [trajs_C[j][(0, 2), -1] for j in range(n_trajs)]
        s = [trajs_C[j][(1, 3), -1] for j in range(n_trajs)]
        w = [trajs_C[j][(4, 6), -1] for j in range(n_trajs)]
        l = [trajs_C[j][(5, 7), -1] for j in range(n_trajs)]
        p = [trajs_C[j][(8, 9), -1] for j in range(n_trajs)]

        (
            uu_one_shot[(i * n_cell) : ((i + 1) * n_cell), :],
            su_one_shot[(i * n_cell) : ((i + 1) * n_cell), :],
            ul_one_shot[(i * n_cell) : ((i + 1) * n_cell), :],
            sl_one_shot[(i * n_cell) : ((i + 1) * n_cell), :],
            pr_one_shot[(i * n_cell) : ((i + 1) * n_cell), :],
        ) = (np.array(u), np.array(s), np.array(w), np.array(l), np.array(p))

    Tl = np.array([0, 1, 2, 4, 8])
    # at the beginning
    deg_begin = syn_deg_data(model_lab, model_unlab, n_trajs, 0, trajs_CP, Tl, n_cell)
    # at the end
    deg_end = syn_deg_data(model_lab, model_unlab, n_trajs, -1, trajs_CP, Tl, n_cell)

    return (
        kin_5,
        kin_40,
        kin_200,
        kin_300,
        (uu_one_shot, su_one_shot, ul_one_shot, sl_one_shot, pr_one_shot),
        deg_begin,
        deg_end,
    )
