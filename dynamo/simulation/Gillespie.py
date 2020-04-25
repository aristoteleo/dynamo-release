#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:21:25 2019

@author: xqiu
"""

from .gillespie_utils import *
from .bif_os_inclusive_sim import sim_diff, sim_osc, simulate, osc_diff_dup
import pandas as pd
import scipy.sparse
from anndata import AnnData


# deterministic as well as noise
def Gillespie(
    a=None,
    b=None,
    la=None,
    aa=None,
    ai=None,
    si=None,
    be=None,
    ga=None,
    C0=np.zeros((5, 1)),
    t_span=[0, 50],
    n_traj=1,
    t_eval=None,
    dt=1,
    method="basic",
    verbose=False,
):
    """A simulator of RNA dynamics that includes RNA bursting, transcription, metabolic labeling, splicing, transcription, RNA/protein degradation

    Parameters
    ----------
        a: `float` or None
            rate of active promoter switches to inactive one
        b: `float` or None
            rate of inactive promoter switches to active one
        la: `float` or None
            lambda_: 4sU labelling rate
        aa: `float` or None
            transcription rate with active promoter
        ai: `float` or None
            transcription rate with inactive promoter
        si: `float` or None
            sigma, degradation rate
        be: `float` or None
            beta, splicing rate
        ga: `float` or None
            gamma: the fraction of labeled u turns to unlabeled s
        C0: `numpy.ndarray` (default: np.zeros((5, 1)))
            A numpy array with dimension of 5 x n_gene. Here 5 corresponds to the five species (s - promoter state, ul,
            uu, sl, su) for each gene.
        t_span:
            list of between and end time of simulation
        n_traj:
            number of simulation trajectory to use
        t_eval: `float` or None
            the time points at which data is simulated
        dt: `float` (default: `1`)
            delta t used in simulation
        method: `str` (default: basic)
            method to simulate the expression dynamics
        verbose: `bool` (default: False)
            whether to report running information

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            an Annodata object containing the simulated data.
    """

    gene_num, species_num = C0.shape[0:2]
    adata_no_splicing, P = None, None

    if method == "basic":
        if t_eval is None:
            steps = (t_span[1] - t_span[0]) // dt  # // int; %% remainder
            t_eval = np.linspace(t_span[0], t_span[1], steps)
        trajs_C = simulate_multigene(
            a=None,
            b=None,
            la=None,
            aa=None,
            ai=None,
            si=None,
            be=None,
            ga=None,
            C0=np.zeros((5, 1)),
            t_span=[0, 50],
            n_traj=1,
            t_eval=None,
            report=verbose,
        )  # unfinished, no need to interpolate now.
        uu, ul, su, sl = [
            np.transpose(trajs_C[:, :, i + 1, :].reshape((gene_num, -1)))
            for i in range(4)
        ]

        u = uu + ul
        s = su + sl
        E = u + s

        layers = {
            "uu": scipy.sparse.csc_matrix(uu.astype(int)),
            "ul": scipy.sparse.csc_matrix(ul.astype(int)),
            "su": scipy.sparse.csc_matrix(su.astype(int)),
            "sl": scipy.sparse.csc_matrix(sl.astype(int)),
            "spliced": scipy.sparse.csc_matrix((s).astype(int)),
            "unspliced": scipy.sparse.csc_matrix((u).astype(int)),
        }  # ambiguous is required for velocyto

        # provide more annotation for cells next:
        cell_ids = [
            "traj_%d_step_%d" % (i, j) for i in range(n_traj) for j in range(steps)
        ]  # first n_traj and then steps
        obs = pd.DataFrame(
            {
                "Cell_name": cell_ids,
                "Trajectory": [i for i in range(n_traj) for j in range(steps)],
                "Step": [j for i in range(n_traj) for j in range(steps)],
            }
        )
        obs.set_index("Cell_name", inplace=True)
    elif method == "simulate_2bifurgenes":
        gene_num = 2
        _, trajs_C = simulate_2bifurgenes(
            a1=20,
            b1=20,
            a2=20,
            b2=20,
            K=20,
            n=3,
            be1=1,
            ga1=1,
            be2=1,
            ga2=1,
            C0=np.zeros(4),
            t_span=t_span,
            n_traj=n_traj,
            report=verbose,
        )  # unfinished, no need to interpolate now.
        u = trajs_C[0][[0, 2], :].T
        s = trajs_C[0][[1, 3], :].T
        E = u + s

        layers = {
            "spliced": scipy.sparse.csc_matrix((s).astype(int)),
            "unspliced": scipy.sparse.csc_matrix((u).astype(int)),
            "ambiguous": scipy.sparse.csc_matrix((E).astype(int)),
        }  # ambiguous is required for velocyto

        steps = u.shape[0]

        # provide more annotation for cells next:
        cell_ids = [
            "traj_%d_step_%d" % (i, j) for i in range(n_traj) for j in range(steps)
        ]  # first n_traj and then steps
        obs = pd.DataFrame(
            {
                "Cell_name": cell_ids,
                "Trajectory": [i for i in range(n_traj) for j in range(steps)],
                "Step": [j for i in range(n_traj) for j in range(steps)],
            }
        )
        obs.set_index("Cell_name", inplace=True)
    elif method == "differentiation":
        gene_num = 2

        # Data Synthesis using Gillespie
        r = 15
        tau = 0.7
        b = 0.5 * r / tau
        c = 0.3 * r
        K = 35 * r
        n = 5
        beta = 0.5 / tau
        gamma = 0.2 / tau
        eta = 0.5 / tau
        delta = 0.05 / tau
        a_ut = 1.0 * r / tau
        a_t = 0.5 * r / tau
        params_untreat_unlab = {
            "a1": a_ut,
            "b1": b,
            "c1": c,
            "a2": a_ut,
            "b2": b,
            "c2": c,
            "a1_l": 0,
            "b1_l": 0,
            "c1_l": 0,
            "a2_l": 0,
            "b2_l": 0,
            "c2_l": 0,
            "K": K,
            "n": n,
            "be1": beta,
            "ga1": gamma,
            "et1": eta,
            "de1": delta,
            "be2": beta,
            "ga2": gamma,
            "et2": eta,
            "de2": delta,
        }
        params_treat_unlab = {
            "a1": a_t,
            "b1": b,
            "c1": c,
            "a2": a_t,
            "b2": b,
            "c2": c,
            "a1_l": 0,
            "b1_l": 0,
            "c1_l": 0,
            "a2_l": 0,
            "b2_l": 0,
            "c2_l": 0,
            "K": K,
            "n": n,
            "be1": beta,
            "ga1": gamma,
            "et1": eta,
            "de1": delta,
            "be2": beta,
            "ga2": gamma,
            "et2": eta,
            "de2": delta,
        }
        params_treat_lab = {
            "a1": 0,
            "b1": 0,
            "c1": 0,
            "a2": 0,
            "b2": 0,
            "c2": 0,
            "a1_l": a_t,
            "b1_l": b,
            "c1_l": c,
            "a2_l": a_t,
            "b2_l": b,
            "c2_l": c,
            "K": K,
            "n": n,
            "be1": beta,
            "ga1": gamma,
            "et1": eta,
            "de1": delta,
            "be2": beta,
            "ga2": gamma,
            "et2": eta,
            "de2": delta,
        }
        model_untreat_unlab = sim_diff(*list(params_untreat_unlab.values()))
        model_treat_unlab = sim_diff(*list(params_treat_unlab.values()))
        model_treat_lab = sim_diff(*list(params_treat_lab.values()))

        # synthesize steady state before treatment
        n_cell = 50
        c0 = np.array(
            [40, 100, 40, 100, 0, 0, 0, 0, 1000, 1000]
        )  # same as the os model after this line of code

        n_species = len(c0)
        trajs_T, trajs_C = simulate(
            model_untreat_unlab,
            C0=[c0] * n_cell,
            t_span=[0, 200],
            n_traj=n_cell,
            report=True,
        )

        kin_5, kin_40, kin_200, kin_300, one_shot, deg_begin, deg_end = osc_diff_dup(
            n_species, trajs_C, model_treat_lab, model_treat_unlab, n_cell
        )

        uu = np.vstack(
            (
                kin_5[0],
                kin_40[0],
                kin_200[0],
                kin_300[0],
                one_shot[0],
                deg_begin[0],
                deg_end[0],
            )
        )
        su = np.vstack(
            (
                kin_5[1],
                kin_40[1],
                kin_200[1],
                kin_300[1],
                one_shot[1],
                deg_begin[1],
                deg_end[1],
            )
        )
        ul = np.vstack(
            (
                kin_5[2],
                kin_40[2],
                kin_200[2],
                kin_300[2],
                one_shot[2],
                deg_begin[2],
                deg_end[2],
            )
        )
        sl = np.vstack(
            (
                kin_5[3],
                kin_40[3],
                kin_200[3],
                kin_300[3],
                one_shot[3],
                deg_begin[3],
                deg_end[3],
            )
        )

        E, New = uu + ul + su + sl, ul + sl
        P = np.vstack(
            (
                kin_5[4],
                kin_40[4],
                kin_200[4],
                kin_300[4],
                one_shot[4],
                deg_begin[4],
                deg_end[4],
            )
        )  # append to .obsm attribute

        layers = {
            "uu": scipy.sparse.csc_matrix((uu).astype(int)),
            "ul": scipy.sparse.csc_matrix((ul).astype(int)),
            "su": scipy.sparse.csc_matrix((su).astype(int)),
            "sl": scipy.sparse.csc_matrix((sl).astype(int)),
        }  # ambiguous is required for velocyto
        layers_no_splicing = {
            "new": scipy.sparse.csc_matrix((New).astype(int)),
            "total": scipy.sparse.csc_matrix((E).astype(int)),
        }  # ambiguous is required for velocyto

        kin_len, one_shot_len, begin_len, end_len = (
            kin_5[0].shape[0],
            one_shot[0].shape[0],
            deg_begin[0].shape[0],
            deg_end[0].shape[0],
        )
        kin_Tl, kin_T_CP, deg_label_t = (
            [0, 0.1, 0.2, 0.4, 0.8],
            [0, 5, 10, 40, 100, 200, 300, 400],
            [0, 1, 2, 4, 8],
        )

        # label time for kinetics experiment is 1 (actually it is one-shot experiment)
        kin_cell_ids, kin_Trajectory, kin_Step = (
            ["kin_traj_%d_time_%f" % (i, j) for j in kin_Tl for i in range(n_cell)],
            [i for j in kin_Tl for i in range(n_cell)],
            [j for j in kin_Tl for i in range(n_cell)],
        )
        one_shot_cell_ids, one_shot_Trajectory, one_shot_Step = (
            [
                "one_shot_traj_%d_time_%d" % (i, j)
                for j in kin_T_CP
                for i in range(n_cell)
            ],
            [i for j in kin_T_CP for i in range(n_cell)],
            [j for j in kin_T_CP for i in range(n_cell)],
        )  # first n_traj and then steps
        begin_cell_ids, begin_Trajectory, begin_Step = (
            [
                "begin_deg_traj_%d_time_%d" % (i, j)
                for j in deg_label_t
                for i in range(n_cell)
            ],
            [i for j in deg_label_t for i in range(n_cell)],
            [j for j in deg_label_t for i in range(n_cell)],
        )  # first n_traj and then steps
        end_cell_ids, end_Trajectory, end_Step = (
            [
                "end_deg_traj_%d_time_%d" % (i, j)
                for j in deg_label_t
                for i in range(n_cell)
            ],
            [i for j in deg_label_t for i in range(n_cell)],
            [j for j in deg_label_t for i in range(n_cell)],
        )  # first n_traj and then steps
        cell_ids, Trajectory, Step = kin_cell_ids * 4, kin_Trajectory * 4, kin_Step * 4
        cell_ids.extend(one_shot_cell_ids)
        Trajectory.extend(one_shot_Trajectory)
        Step.extend(one_shot_Step)
        cell_ids.extend(begin_cell_ids)
        Trajectory.extend(begin_Trajectory)
        Step.extend(begin_Step)
        cell_ids.extend(end_cell_ids)
        Trajectory.extend(end_Trajectory)
        Step.extend(end_Step)

        obs = pd.DataFrame(
            {
                "cell_name": cell_ids,
                "trajectory": Trajectory,
                "time": Step,
                "experiment_type": pd.Series(
                    [
                        "kin_t_5",
                        "kin_t_40",
                        "kin_t_200",
                        "kin_t_300",
                        "one_shot",
                        "deg_beign",
                        "deg_end",
                    ]
                )
                .repeat(
                    [
                        kin_len,
                        kin_len,
                        kin_len,
                        kin_len,
                        one_shot_len,
                        begin_len,
                        end_len,
                    ]
                )
                .values,
            }
        )
        obs.set_index("cell_name", inplace=True)
    elif method == "oscillation":
        gene_num = 2

        # Data Synthesis using Gillespie

        r = 20
        tau = 3
        beta = 0.5 / tau
        gamma = 0.2 / tau
        eta = 0.5 / tau
        delta = 0.05 / tau
        zeta = eta * beta / (delta * gamma)
        a1 = 1.5 * r / tau
        b1 = 1 * r / tau
        a2 = 0.5 * r / tau
        b2 = 2.5 * r / tau
        K = 2.5 * r
        n = 10
        params_unlab = {
            "a1": a1,
            "b1": b1,
            "a2": a2,
            "b2": b2,
            "a1_l": 0,
            "b1_l": 0,
            "a2_l": 0,
            "b2_l": 0,
            "K": K,
            "n": n,
            "be1": beta,
            "ga1": gamma,
            "et1": eta,
            "de1": delta,
            "be2": beta,
            "ga2": gamma,
            "et2": eta,
            "de2": delta,
        }
        params_lab = {
            "a1": 0,
            "b1": 0,
            "a2": 0,
            "b2": 0,
            "a1_l": a1,
            "b1_l": b1,
            "a2_l": a2,
            "b2_l": b2,
            "K": K,
            "n": n,
            "be1": beta,
            "ga1": gamma,
            "et1": eta,
            "de1": delta,
            "be2": beta,
            "ga2": gamma,
            "et2": eta,
            "de2": delta,
        }
        model_unlab = sim_osc(*list(params_unlab.values()))
        model_lab = sim_osc(*list(params_lab.values()))

        # synthesize steady state before treatment
        n_cell = 50
        c0 = np.array(
            [
                70,
                70 * beta / gamma,
                70,
                70 * beta / gamma,
                0,
                0,
                0,
                0,
                70 * zeta,
                70 * zeta,
            ]
        )

        n_species = len(c0)
        trajs_T, trajs_C = simulate(
            model_unlab, C0=[c0] * n_cell, t_span=[0, 100], n_traj=n_cell, report=True
        )

        kin_5, kin_40, kin_200, kin_300, one_shot, deg_begin, deg_end = osc_diff_dup(
            n_species, trajs_C, model_lab, model_unlab, n_cell
        )

        uu = np.vstack(
            (
                kin_5[0],
                kin_40[0],
                kin_200[0],
                kin_300[0],
                one_shot[0],
                deg_begin[0],
                deg_end[0],
            )
        )
        su = np.vstack(
            (
                kin_5[1],
                kin_40[1],
                kin_200[1],
                kin_300[1],
                one_shot[1],
                deg_begin[1],
                deg_end[1],
            )
        )
        ul = np.vstack(
            (
                kin_5[2],
                kin_40[2],
                kin_200[2],
                kin_300[2],
                one_shot[2],
                deg_begin[2],
                deg_end[2],
            )
        )
        sl = np.vstack(
            (
                kin_5[3],
                kin_40[3],
                kin_200[3],
                kin_300[3],
                one_shot[3],
                deg_begin[3],
                deg_end[3],
            )
        )

        E, New = uu + ul + su + sl, ul + sl
        P = np.vstack(
            (
                kin_5[4],
                kin_40[4],
                kin_200[4],
                kin_300[4],
                one_shot[4],
                deg_begin[4],
                deg_end[4],
            )
        )  # append to .obsm attribute

        layers = {
            "uu": scipy.sparse.csc_matrix((uu).astype(int)),
            "ul": scipy.sparse.csc_matrix((ul).astype(int)),
            "su": scipy.sparse.csc_matrix((su).astype(int)),
            "sl": scipy.sparse.csc_matrix((sl).astype(int)),
        }  # ambiguous is required for velocyto
        layers_no_splicing = {
            "new": scipy.sparse.csc_matrix((New).astype(int)),
            "total": scipy.sparse.csc_matrix((E).astype(int)),
        }  # ambiguous is required for velocyto

        kin_len, one_shot_len, begin_len, end_len = (
            kin_5[0].shape[0],
            one_shot[0].shape[0],
            deg_begin[0].shape[0],
            deg_end[0].shape[0],
        )
        kin_Tl, kin_T_CP, deg_label_t = (
            [0, 0.1, 0.2, 0.4, 0.8],
            [0, 5, 10, 40, 100, 200, 300, 400],
            [0, 1, 2, 4, 8],
        )

        # label time for kinetics experiment is 1 (actually it is one-shot experiment)
        kin_cell_ids, kin_Trajectory, kin_Step = (
            ["kin_traj_%d_time_%f" % (i, j) for j in kin_Tl for i in range(n_cell)],
            [i for j in kin_Tl for i in range(n_cell)],
            [j for j in kin_Tl for i in range(n_cell)],
        )
        one_shot_cell_ids, one_shot_Trajectory, one_shot_Step = (
            [
                "one_shot_traj_%d_time_%d" % (i, j)
                for j in kin_T_CP
                for i in range(n_cell)
            ],
            [i for j in kin_T_CP for i in range(n_cell)],
            [j for j in kin_T_CP for i in range(n_cell)],
        )  # first n_traj and then steps
        begin_cell_ids, begin_Trajectory, begin_Step = (
            [
                "begin_deg_traj_%d_time_%d" % (i, j)
                for j in deg_label_t
                for i in range(n_cell)
            ],
            [i for j in deg_label_t for i in range(n_cell)],
            [j for j in deg_label_t for i in range(n_cell)],
        )  # first n_traj and then steps
        end_cell_ids, end_Trajectory, end_Step = (
            [
                "end_deg_traj_%d_time_%d" % (i, j)
                for j in deg_label_t
                for i in range(n_cell)
            ],
            [i for j in deg_label_t for i in range(n_cell)],
            [j for j in deg_label_t for i in range(n_cell)],
        )  # first n_traj and then steps
        cell_ids, Trajectory, Step = kin_cell_ids * 4, kin_Trajectory * 4, kin_Step * 4
        cell_ids.extend(one_shot_cell_ids)
        Trajectory.extend(one_shot_Trajectory)
        Step.extend(one_shot_Step)
        cell_ids.extend(begin_cell_ids)
        Trajectory.extend(begin_Trajectory)
        Step.extend(begin_Step)
        cell_ids.extend(end_cell_ids)
        Trajectory.extend(end_Trajectory)
        Step.extend(end_Step)

        obs = pd.DataFrame(
            {
                "cell_name": cell_ids,
                "trajectory": Trajectory,
                "time": Step,
                "experiment_type": pd.Series(
                    [
                        "kin_t_5",
                        "kin_t_40",
                        "kin_t_200",
                        "kin_t_300",
                        "one_shot",
                        "deg_beign",
                        "deg_end",
                    ]
                )
                .repeat(
                    [
                        kin_len,
                        kin_len,
                        kin_len,
                        kin_len,
                        one_shot_len,
                        begin_len,
                        end_len,
                    ]
                )
                .values,
            }
        )
        obs.set_index("cell_name", inplace=True)
    else:
        raise Exception("method not implemented!")
    # anadata: observation x variable (cells x genes)

    if verbose:
        print("we have %s cell and %s genes." % (E.shape[0], E.shape[1]))

    var = pd.DataFrame(
        {
            "gene_short_name": ["gene_%d" % (i) for i in range(gene_num)],
            "true_beta": [beta, beta],
            "true_gamma": [gamma, gamma],
            "true_eta": [eta, eta],
            "true_delta": [delta, delta],
        }
    )  # use the real name in simulation?
    var.set_index("gene_short_name", inplace=True)

    adata = AnnData(
        scipy.sparse.csc_matrix(E.astype(int)).copy(),
        obs.copy(),
        var.copy(),
        layers=layers.copy(),
    )
    adata_no_splicing = AnnData(
        scipy.sparse.csc_matrix(E.astype(int)).copy(),
        obs.copy(),
        var.copy(),
        layers=layers_no_splicing.copy(),
    )
    if P is not None:
        adata.obsm["protein"] = P
        adata_no_splicing.obsm["protein"] = P
    # remove cells that has no expression
    adata = adata[np.array(adata.X.sum(1)).flatten() > 0, :]
    adata_no_splicing = adata_no_splicing[
        np.array(adata_no_splicing.X.sum(1)).flatten() > 0, :
    ]

    return adata, adata_no_splicing
