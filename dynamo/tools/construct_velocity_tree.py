import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import re

from .DDRTree_py import DDRTree_py


def remove_velocity_points(G, n):
    for nodeid in range(n, 2 * n):
        nb_ids = []
        for nb_id in range(len(G[0])):
            if G[nodeid][nb_id] != 0:
                nb_ids = nb_ids + [nb_id]
        num_nbs = len(nb_ids)

        if num_nbs == 1:
            G[nodeid][nb_ids[0]] = 0
            G[nb_ids[0]][nodeid] = 0
        else:
            min_val = np.inf
            for i in range(len(G[0])):
                if G[nodeid][i] != 0:
                    if G[nodeid][i] < min_val:
                        min_val = G[nodeid][i]
                        min_ind = i
            for i in nb_ids:
                if i != min_ind:
                    new_weight = G[nodeid][i] + min_val
                    G[i][min_ind] = new_weight
                    G[min_ind][i] = new_weight
            # print('Add ege %s, %s\n',G.Nodes.Name {nb_ids(i)}, G.Nodes.Name {nb_ids(min_ind)});
            G[nodeid][nb_ids[0]] = 0
            G[nb_ids[0]][nodeid] = 0

    return G


def calculate_angle(o, y, x):
    yo = y - o
    norm_yo = yo / scipy.linalg.norm(yo)
    xo = x - o
    norm_xo = xo / scipy.linalg.norm(xo)
    angle = np.arccos(norm_yo.T * norm_xo)
    return angle


def construct_velocity_tree_py(X1, X2):

    n = X1.shape[1]

    # merge two data with a given time
    t = 0.5
    X_all = np.hstack((X1, X1 + t * X2))

    # parameter settings
    maxIter = 20
    eps = 1e-3
    sigma = 0.001
    gamma = 10

    # run DDRTree algorithm
    W, Z, stree, Y, R, history = DDRTree_py(
        X_all, maxIter=maxIter, eps=eps, sigma=sigma, gamma=gamma
    )

    # draw velocity figure

    # quiver(Z(1, 1: 100), Z(2, 1: 100), Z(1, 101: 200)-Z(1, 1: 100), Z(2, 101: 200)-Z(2, 1: 100));
    # plot(Z(1, 1: 100), Z(2, 1: 100), 'ob');
    # plot(Z(1, 101: 200), Z(2, 101: 200), 'sr');
    G = stree

    sG = remove_velocity_points(G, n)
    tree = sG
    row = []
    col = []
    val = []
    for i in range(sG.shape[0]):
        for j in range(sG.shape[1]):
            if sG[i][j] != 0:
                row = row + [i]
                col = col + [j]
                val = val + [sG[1][j]]
    tree_fname = "tree.csv"
    # write sG data to tree.csv
    #######
    branch_fname = "branch.txt"
    cmd = "python extract_branches.py" + tree_fname + branch_fname

    branch_cell = []
    fid = open(branch_fname, "r")
    tline = next(fid)
    while isinstance(tline, str):
        path = re.regexp(tline, "\d*", "Match")  ############
        branch_cell = branch_cell + [path]  #################
        tline = next(fid)
    fid.close()

    dG = np.zeros((n, n))
    for p in range(len(branch_cell)):
        path = branch_cell[p]
        pos_direct = 0
        for bp in range(len(path)):
            u = path(bp)
            v = u + n

            # find the shorest path on graph G(works for trees)
            nodeid = u
            ve_nodeid = v
            shortest_mat = shortest_path(
                csgraph=G, directed=False, indices=nodeid, return_predecessors=True
            )
            velocity_path = []
            while ve_nodeid != nodeid:
                velocity_path = [shortest_mat[nodeid][ve_nodeid]] + velocity_path
                ve_nodeid = shortest_mat[nodeid][ve_nodeid]
            velocity_path = [shortest_mat[nodeid][ve_nodeid]] + velocity_path
            ###v_path = G.Nodes.Name(velocity_path)

            # check direction consistency between path and v_path
            valid_idx = []
            for i in velocity_path:
                if i <= n:
                    valid_idx = valid_idx + [i]
            if len(valid_idx) == 1:
                # compute direction matching
                if bp < len(path):
                    tree_next_point = Z[:, path(bp)]
                    v_point = Z[:, v]
                    u_point = Z[:, u]
                    angle = calculate_angle(u_point, tree_next_point, v_point)
                    angle = angle / 3.14 * 180
                    if angle < 90:
                        pos_direct = pos_direct + 1

                else:
                    tree_pre_point = Z[:, path(bp - 1)]
                    v_point = Z[:, v]
                    u_point = Z[:, u]
                    angle = calculate_angle(u_point, tree_pre_point, v_point)
                    angle = angle / 3.14 * 180
                    if angle > 90:
                        pos_direct = pos_direct + 1

            else:

                if bp < len(path):
                    if path[bp + 1] == valid_idx[2]:
                        pos_direct = pos_direct + 1

                else:
                    if path[bp - 1] != valid_idx[2]:
                        pos_direct = pos_direct + 1

        neg_direct = len(path) - pos_direct
        print(
            "branch="
            + str(p)
            + ", ("
            + path[0]
            + "->"
            + path[-1]
            + "), pos="
            + pos_direct
            + ", neg="
            + neg_direct
            + "\n"
        )
        print(path)
        print("\n")

        if pos_direct > neg_direct:
            for bp in range(len(path) - 1):
                dG[path[bp], path[bp + 1]] = 1

        else:
            for bp in range(len(path) - 1):
                dG[path(bp + 1), path(bp)] = 1

    # figure;
    # plot(digraph(dG));
    # title('directed graph') figure; hold on;
    row = []
    col = []
    for i in range(dG.shape[0]):
        for j in range(dG.shape[1]):
            if dG[i][j] != 0:
                row = row + [i]
                col = col + [j]
    for tn in range(len(row)):
        p1 = Y[:, row[tn]]
        p2 = Y[:, col[tn]]
        dp = p2 - p1
        h = plt.quiver(
            p1(1), p1(2), dp(1), dp(2), "LineWidth", 5
        )  ###############need to plot it
        set(h, "MaxHeadSize", 1e3, "AutoScaleFactor", 1)  #############

    for i in range(n):
        plt.text(Y(1, i), Y(2, i), str(i))  ##############
    plt.savefig("./results/t01_figure3.fig")  ##################
