import os
import sys
import glob
import time
import importlib
import networkx as nx
import seaborn as sns
import pandas as pd
import itertools
import collections

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import squareform
from scipy import interpolate

from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from .diffusion import *
from .compute_cell_velocity import compute_cell_velocity
from .plotting.cell import calculate_para_umap
    
def compute_trajectory_displacement(traj):
    traj = np.array(traj)
    return np.linalg.norm(traj[-1,:] - traj[0,:])

def compute_trajectory_length(traj1):
    temp = traj1[:-1,:] - traj1[1:,:]
    length = np.sum(np.sqrt(np.sum(temp**2, axis=1)))
    return length


def compute_trajectory_similarity(traj1, traj2, numPicks=10):
    '''
    Computes the similarity between two curves based on average distance of 
    a selection of closest pairs
    Input: 
    - numpy arrays (nsteps1, 2), (nsteps2, 2); nsteps1 >= nsteps2 
    - numPicks: number of selected points on the shorter curve traj2
    Return: a float number
    '''
    # traj1 is longer than traj2

    if not traj2.size:
    # empty traj2
        print("empty trajectory here!")
        raise 

    # pick numPicks points evenly from traj2
    idx = np.round(np.linspace(0, len(traj2) - 1, numPicks)).astype(int)
    
    # in the case the shorter trajectory is less than numPicks timesteps
    idx = np.unique(idx)
    
    temp = traj1 - traj2[idx, None]
    A = np.min(np.sum(temp**2, axis=2), axis=1)
    
    return np.mean(A**(1/2))


def compute_similarity_matrix(traj, numPicks=10):
    # UNUSED
    # This is only to be used for serious clustering.
    traj_length = np.array([compute_trajectory_length(np.array(i)) 
        for i in traj])
    traj_order = np.argsort(traj_length) # 1d array
    
    ncells = len(traj_order)
    simMat = np.zeros((ncells,ncells))
    for i,j in itertools.combinations(traj_order, 2):
            # length of traj[i] <= traj[j]
            simMat[i,j] = compute_trajectory_similarity(np.array(traj[j]), 
                    np.array(traj[i]), numPicks)
            simMat[j,i] = simMat[i,j]
    return simMat


def truncate_end_state_stuttering(paths, cell_embedding):
    newPaths = [ipath[:np.int32(np.where(
        np.linalg.norm(ipath-ipath[-1], axis=1) < 1e-3)[0][0])] 
        for ipath in paths]
    newPaths = [i for i in newPaths if len(i) > 1]
    return np.array(newPaths, dtype=object)


def extract_long_trajectories(
        path_clusters, 
        cell_clusters, 
        paths, 
        similarity_cutoff, 
        similarity_threshold, 
        nkeep=10):
    '''
    a recursive method to find long paths and group similar paths.
    
    Parameters
    ----------
    
    paths: np.ndarray (N, ntimesteps, 2)
        N paths, sorted by their |displacement|, 
        each trajectory is a (ntimestep, 2) array
        
    similarity_threshold: float
        group trajectories within this similarity threshold
    
    After each iteration, a number of trajectories are popped in the traj list 
    returns a list of clusters
    
    Return
    ------
    path_clusters: a dictionary of np.ndarray (ntimesteps, 2)
    cell_clusters: a dictionary of np.ndarray (2, )
    
    '''
    clusterID = len(path_clusters)
    if not paths.size:
        return path_clusters, cell_clusters
    
    longest = paths[0]
    similarity = np.array([
        compute_trajectory_similarity(np.array(longest), np.array(ipath), 10) 
        for ipath in paths])
    
    sel = (similarity < similarity_cutoff)
    sel_keep = (similarity_threshold <= similarity)
    cluster = paths[sel & sel_keep][:nkeep]
    if len(cluster) == 0:
        #print("this cluster has no traj to keep")
        cluster = paths[0,None]
    elif not np.array_equal(paths[0], cluster[0]):
        #print("concat", cluster[0].shape, paths[0].shape)
        cluster = np.append(cluster, paths[0,None])
    path_clusters[clusterID] = cluster
    cell_clusters[clusterID] = [ipath[0] for ipath in paths[sel]]
    
    paths = paths[~sel]
    return extract_long_trajectories(
            path_clusters, 
            cell_clusters, 
            paths, 
            similarity_cutoff, 
            similarity_threshold, 
            nkeep)

    
def cell_fate_tuning(embedding, cell_clusters, n_neighbors=20):
    '''
    Parameters
    ----------
    embedding: numpy ndarray (ncells, 2)
    
    cell_clusters: dictionary of length n_clusters
        A dictionary of starting cell positions (cluster_size, 2) 
    n_neighbors: float
        
    Return
    ------
    A numpy array (size: ncells) recording fate of each cell in the order of
    cellIDs.
    '''

    # path time to cell time.
    n_clusters = len(cell_clusters)
    n_cells = len(embedding)

    # initialization
    # each cell has a chance to go through the n_clusters fates
    # according to its match with cells in each cluster
    # defined by [p0, p1, ..., p{n_cluster-1}]

    cell_fate = np.zeros((n_cells, n_clusters))
    cluster_index = 0
    clusterIDs = list()
    for cluster, cell_embeddings in cell_clusters.items():
        temp = cell_embeddings - embedding[:,None]

        # the np.where tuple.
        # tuple [0] --> cell indices
        # tuple [1] --> traj indices in the cluster
        for i in np.where(np.sum(temp**2, axis = 2) == 0)[0]:
            cell_fate[i][cluster_index] += 1
        cluster_index += 1
        clusterIDs.append(cluster)
    cell_fate_major = np.argmax(cell_fate, axis=1)
    #print(cell_fate_major)

    # mapping back to clusterIDs
    # cluster_index is the list index of the clusterIDs
    cell_fate_major = np.array([clusterIDs[i] for i in cell_fate_major],
            dtype=int)

    #print(cell_fate_major)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(embedding)
    A = neigh.kneighbors_graph(embedding)
    B = A.toarray()
    cell_fate_tuned = np.array([collections.Counter(B[i][B[i]!=0]* \
            cell_fate_major[B[i]!=0]).most_common()[0][0] \
            for i in range(n_cells)], dtype=int)
#    if set(cell_fate_tuned).issubset(set(cell_fate_major)):
#        del_path_clusters = set(cell_fate_major) - set(cell_fate_tuned)
#        print("Those path clusters are removed: ", del_path_clusters)

    return np.array(cell_fate_tuned)


def projection_cell_time_one_cluster(embedding, rep_path, cluster, cell_fate):
    '''
    Parameters
    ----------
    embedding: numpy ndarray
    
    rep_path: numpy ndarray
        used a reference for cell time projection
    
    clusters: int
        which cluster of cells
        
    cell_fate: numpy 1d array
        cluster number for each cell
        
    Return
    ------
    A dictionary of cell time {cellID: time}
    '''
    cell_time= list()
    #print("Path ", cluster)
    cells = embedding[cell_fate == cluster]
    #print(cells)
    cell_index = np.where(cell_fate == cluster)[0]
    dist = cells[:,None] - rep_path
    cell_time=np.argmin(np.sum(dist**2, axis = 2), axis = 1)
    cell_time_per_cluster = {A: B for A, B in zip(cell_index, cell_time)}
    return cell_time_per_cluster


def closest_distance_between_two_paths(path1, path2, cell_embedding):
    '''
    returns the closest distance and the closest cells.
    '''
   
    if path1.size and path2.size:
        temp = path1 - path2[:, None]
        A = np.sum(temp**2, axis=2)
        pair = np.unravel_index(np.argmin(A), A.shape)
        #print("The closest distance is ", np.sqrt(A[pair]))
        #print("Between ", pair[1], " from refPath1 and ", \
        #        pair[0], " from refPath2.")
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(cell_embedding[:,0], cell_embedding[:,1], alpha = 0.3)
        plt.scatter(path1[:,0], path1[:,1], c=range(len(path1)), s=5)
        plt.text(path1[-1,0], path1[-1,1], "refPath"+str(1), fontsize=12)
        plt.text(path2[-1,0], path2[-1,1], "refPath"+str(2), fontsize=12)        
        plt.scatter(path2[:,0], path2[:,1], c=range(len(path2)), s=5)
        plt.show()
        return np.sqrt(A[pair]), pair[::-1]
    else:
        return np.Inf,(np.nan,np.nan)


def resolve_terminal_cells(terminal_cells, 
                           cell_time_subclusters,
                           sorted_refPaths,
                           cell_embedding, 
                           grid_mass,
                           cell_grid_idx,
                           vel,
                           cluster,
                           cell_fate,
                           dt,
                           t_total,
                           n_repeats,
                           psrng_seeds_diffusion,
                           MAX_ALLOWED_ZERO_TIME_CELLS,
                           MAX_ALLOWED_TERM_CELLS,
                           n_jobs,
                           level=0,
                           NO_ZERO=False,
                           NO_TERM=False,
                           ):
    # get subsampled cell embedding & mass matrix
    #print(f"level is {level}")
    if level > 10:
        print(f"WARNING: Abnormally too many times ({level}) to resolve terminal cells.")
        print(f"WARNING: This is likely due to a too large dt ({dt}).")
        print(f"WARNING: You can either use this result if you think your dt is reasonable; or rerun with a smaller dt.")
        return None
    sub_embedding = cell_embedding[terminal_cells]
    sub_grid_mass = np.zeros_like(grid_mass)
    for cell in terminal_cells:
        i = tuple(cell_grid_idx[cell])
        sub_grid_mass[i] = grid_mass[i]

   # generate new trajectories starting from the terminal cells
   #print("Sample trajs for terminal cells in cluster ", cluster, " ...")
    sub_traj = run_diffusion(
            cell_embedding, 
            vel=vel,
            grid_mass=sub_grid_mass, 
            dt=dt, 
            t_total=t_total, 
            eps=1e-3, 
            off_cell_init=False, 
            init_cell=terminal_cells, 
            n_repeats=n_repeats, 
            psrng_seeds_diffusion=psrng_seeds_diffusion,
            n_jobs=n_jobs)
        
    # Find the longest trajectory
    newPaths = truncate_end_state_stuttering(sub_traj, cell_embedding)
    traj_displacement = np.array([compute_trajectory_displacement(ipath) for ipath in newPaths])

    order = np.argsort(traj_displacement)[::-1]
    sorted_traj = newPaths[order]
    ref_path = sorted_traj[0]

    # re-assign time for zero time cells
    sub_cell_time = projection_cell_time_one_cluster(
            cell_embedding, 
            ref_path, 
            cluster, 
            cell_fate)

    sorted_refPaths.append(ref_path)
        
    term_out = recur_cell_time_assignment_intracluster(
            {cell: sub_cell_time[cell] for cell in terminal_cells},
            cell_time_subclusters, 
            cluster, 
            sorted_refPaths, 
            cell_fate,
            cell_embedding,
            vel, 
            cell_grid_idx, 
            grid_mass, 
            dt=dt, 
            t_total=t_total, 
            n_repeats=n_repeats, 
            n_jobs=n_jobs,
            psrng_seeds_diffusion=psrng_seeds_diffusion,
            MAX_ALLOWED_ZERO_TIME_CELLS=MAX_ALLOWED_ZERO_TIME_CELLS,
            MAX_ALLOWED_TERM_CELLS=MAX_ALLOWED_TERM_CELLS,
            NO_ZERO=NO_ZERO,
            NO_TERM=NO_TERM,
            level=level+1)

    cell_time_subclusters, sorted_refPaths = term_out[0], term_out[1]
    return cell_time_subclusters, sorted_refPaths

    
def recur_cell_time_assignment_intracluster(
    unresolved_cell_time_cluster, 
    cell_time_subclusters,
    cluster, 
    sorted_refPaths, 
    cell_fate,
    cell_embedding, 
    vel, 
    cell_grid_idx, 
    grid_mass, 
    dt=0.001, 
    t_total=10000, 
    n_repeats=10, 
    n_jobs=-1,
    n_recur=0,
    psrng_seeds_diffusion=None,
    MAX_ALLOWED_ZERO_TIME_CELLS=0.05,
    MAX_ALLOWED_TERM_CELLS=0.05,
    NO_ZERO=False,
    NO_TERM=False,
    level=0):
    '''
    Recursive function to consolidate cell time within a cluster.
    
    Parameters
    ----------
    unresolved_cell_time_cluster: dictionary {cellID : time}
        cellIDs and corresponding unresolved time for a specific cluster
    cell_time_subclusters: list of dictionaries {cellID : time}
        resolved cell time (yet to be adjusted between subclusters)
    
    sorted_refPaths: list
        list of paths in a cluster ordered from long to short in displacement. 
        
    cell_embedding, vel, grid_mass: a set of parameters from the diffusion simulations.
    
    Return
    ------
    resolved_cell_time_cluster: dictionary
    
    cell_time_subclusters: list
    sorted_refPaths: list
        a list of longest trajectories used for cell time projection
    level: integer
        times that the function runs.
    '''
    
    # print("resolving intraCluster ", cluster)
    # print(len(cell_time_subclusters), len(sorted_refPaths))

    ZERO = 0
    TERMINAL_TIME = max(unresolved_cell_time_cluster.values())

    if isinstance(MAX_ALLOWED_ZERO_TIME_CELLS, float):
        MAX_ALLOWED_ZERO_TIME_CELLS = int(MAX_ALLOWED_ZERO_TIME_CELLS \
                * len(unresolved_cell_time_cluster))
    MAX_ALLOWED_ZERO_TIME_CELLS = max(MAX_ALLOWED_ZERO_TIME_CELLS, 10)

    if isinstance(MAX_ALLOWED_TERM_CELLS, float):
        MAX_ALLOWED_TERM_CELLS = int(MAX_ALLOWED_TERM_CELLS \
                * len(unresolved_cell_time_cluster))
    MAX_ALLOWED_TERM_CELLS = max(MAX_ALLOWED_TERM_CELLS, 10)

    #print("Total cells in the cluster: ", len(unresolved_cell_time_cluster))
    #print("MAX allowed zero time cells: ", MAX_ALLOWED_ZERO_TIME_CELLS)
    #print("MAX allowed terminal cells: ", MAX_ALLOWED_TERM_CELLS)

    zero_time_cells = [cellid for cellid, celltime in 
            unresolved_cell_time_cluster.items() if celltime <= ZERO]

    terminal_cells = [cellid for cellid, celltime in 
            unresolved_cell_time_cluster.items() if celltime >= TERMINAL_TIME]

    # non-zero/non-terminal time cells form a subcluster.
    cell_time_mid=dict()
    NO_ZERO = (len(zero_time_cells) < MAX_ALLOWED_ZERO_TIME_CELLS) or NO_ZERO
    NO_TERM = (len(terminal_cells) < MAX_ALLOWED_TERM_CELLS) or NO_TERM
    if NO_ZERO:
        #print("Only ", len(zero_time_cells), " zero cells left. ")
        #print(zero_time_cells)
        for i in zero_time_cells:
            cell_time_mid[i] = unresolved_cell_time_cluster[i]
        zero_time_cells = list()

    if NO_TERM:
        #print("Only ", len(terminal_cells), " terminal cells left.")
        #print(terminal_cells)
        for i in terminal_cells:
            cell_time_mid[i] = unresolved_cell_time_cluster[i]
        terminal_cells = list()

    for i in unresolved_cell_time_cluster:
        if (i not in zero_time_cells) and (i not in terminal_cells):
            cell_time_mid[i] = unresolved_cell_time_cluster[i]
    if len(cell_time_mid) > 0:
        cell_time_subclusters.append(cell_time_mid) 

    # This is where the recursive method ends.
    #if NO_ZERO and NO_TERM:
    #    return cell_time_subclusters, sorted_refPaths, level, NO_ZERO, NO_TERM

    if not NO_ZERO:
        #print(len(zero_time_cells), " ZERO cells left.")
        resolve_out = resolve_terminal_cells(
                zero_time_cells, 
                cell_time_subclusters,
                sorted_refPaths,
                cell_embedding, 
                grid_mass,
                cell_grid_idx,
                vel,
                cluster,
                cell_fate,
                dt,
                t_total,
                n_repeats,
                psrng_seeds_diffusion,
                MAX_ALLOWED_ZERO_TIME_CELLS,
                MAX_ALLOWED_TERM_CELLS,
                n_jobs,
                level=level,
                NO_ZERO=NO_ZERO,
                NO_TERM=NO_TERM)

        if isinstance(resolve_out, tuple):
            cell_time_subclusters, sorted_refPaths = resolve_out
        #else:
        NO_ZERO = True
        level = 0


    if not NO_TERM:
        #print(len(terminal_cells), " TERMINAL cells left.")
        resolve_out = resolve_terminal_cells(
                terminal_cells, 
                cell_time_subclusters,
                sorted_refPaths,
                cell_embedding, 
                grid_mass,
                cell_grid_idx,
                vel,
                cluster,
                cell_fate,
                dt,
                t_total,
                n_repeats,
                psrng_seeds_diffusion,
                MAX_ALLOWED_ZERO_TIME_CELLS,
                MAX_ALLOWED_TERM_CELLS,
                n_jobs,
                level=level,
                NO_ZERO=NO_ZERO,
                NO_TERM=NO_TERM)
        if isinstance(resolve_out, tuple):
            cell_time_subclusters, sorted_refPaths = resolve_out
        #else:
        NO_TERM = True
        level = 0

    return cell_time_subclusters, sorted_refPaths, level, NO_ZERO, NO_TERM
    
    
def cell_time_assignment_intercluster(
        unresolved_cell_time, 
        cell_fate_dict, 
        cell_embedding, 
        tau = 0.05):
    '''
    Consolidate cell time between clusters according to the intersection
    between cells from any two clusters.
    Assumption: No cyclic behavior between clusters. Else, return None.
    We construct a directed graph to detect cycles.
    CT --> inter-cluster time gap (transfer)
    Parameters
    ----------
    unresolved_cell_time: list
        a list of dictionary {cellID : time}
        cellIDs and corresponding unresolved time for all cells.
    cell_fate_dict: dictionary
        {cell index:cluster}
    cell_embedding: numpy ndarray
        all downsampled cell embedding
    Return
    ------
    resolved_cell_time_cluster: list
        a list of dictionaries {cellID : time}
    '''

    #print("\nintercluster cell time adjustment")
    #print("number of cells: ", len(cell_fate_dict))

    #print(cell_fate_dict)
    clusterIDs = sorted(np.unique(list(cell_fate_dict.values())))

    cutoff = overlap_crit_intracluster(cell_embedding, cell_fate_dict, tau)
    #print("Cutoff is ", cutoff)

    CT = nx.DiGraph()
    for cluster in clusterIDs:
        CT.add_node(cluster)

    # nodes
    nodes = clusterIDs
    n_nodes = len(nodes)
    #print("Number of nodes: ", n_nodes)
    # paths
    paths = list()
    # weights
    w = list()
    
    # MAX_IGNORED_TIME_SHIFT is set to 50% of the shortest cluster.
    durations = list()
    for cluster_index in range(len(clusterIDs)):
        time_cluster = unresolved_cell_time[cluster_index].values()
        duration_cluster = max(time_cluster)-min(time_cluster)
        durations.append(duration_cluster)
    MAX_IGNORED_TIME_SHIFT = 0.5 * min(durations)

    # Always no cycles if n_nodes <=2
    if n_nodes < 3:
        MAX_IGNORED_TIME_SHIFT = 0

    # good chance no cycles if n_nodes = 3
    if n_nodes == 3:
        MAX_IGNORED_TIME_SHIFT = 0.1 * MAX_IGNORED_TIME_SHIFT

    # i, j are the clusterIDs
    # they have to be converted to cluster_index to be used by
    # overlap_intercluster
    #print("clusterIDs: ", clusterIDs)
    for i,j in itertools.combinations(clusterIDs, 2):
        cluster0_index = clusterIDs.index(i)
        cluster1_index = clusterIDs.index(j)
        shiftT, overlap_cells = overlap_intercluster(
                cell_embedding,
                cell_fate_dict, 
                unresolved_cell_time, 
                clusterIDs,
                cluster0_index,
                cluster1_index,
                cutoff)

        if shiftT is not None:
            shiftT = 0 if abs(shiftT) < MAX_IGNORED_TIME_SHIFT else shiftT

            #print("Time shift is: ", shiftT)
            #print("The overlapping cells are:",
                    #"\ncell ", overlap_cells[0], " from cluster ", i, " and ", 
                    #overlap_cells[1], " from cluster ", j)

            if shiftT > 0:
                CT.add_edge(i, j, weight = shiftT)
                w.append(shiftT)
                paths.append([i,j])

            if shiftT < 0:
                CT.add_edge(j, i, weight = -shiftT)
                w.append(-shiftT)
                paths.append([j,i])

            if shiftT == 0:
                CT.add_edge(i, j, weight = MAX_IGNORED_TIME_SHIFT)
                w.append(shiftT)
                paths.append([i,j])

    if len(paths) == 0:
        return unresolved_cell_time

    pos = nx.planar_layout(CT)
#    plt.figure(figsize=(5,5))
#    nx.draw(CT, 
#            pos=pos, 
#            with_labels = True, 
#            node_size=500, 
#            node_color = 'b',
#            style=':',
#            font_size = 18, 
#            font_color = 'w')

#    weights = nx.get_edge_attributes(CT,'weight')
    #labels = {i:int(1/weights[i]) for i in weights}
#    labels = weights
#    nx.draw_networkx_edge_labels(CT,pos,edge_labels=labels)
#    plt.show()
    
    if True:
#    if not nx.is_forest(CT):
#        print("There exists a cycle in the cluster graph.")
#        print("Unable to consolidate cells times in this case.")
#        return unresolved_cell_time
#    else:
        w_cumm = {node:0 for node in nodes}
#        p_w = zip(paths, w)
#        print("Paths are: ", paths)
        for tree_nodes in nx.weakly_connected_components(CT):
            #print("Connected components: ", tree_nodes)
            temp_p = []
            temp_w = []
            for i in tree_nodes:
                for j, k in zip(paths, w):
                    if (i in j) and (j not in temp_p):
                        temp_p.append(j)
                        temp_w.append(k)
#            print(temp_p)
            temp_w_cumm = relative_time_in_tree(temp_p, temp_w)
            if len(temp_w_cumm) > 1:
                for node, time_adj in temp_w_cumm.items(): 
                    w_cumm[node] = time_adj

    #print("All nodes adjustment: ", w_cumm)
    # update pseudotime
    #print("Before:\n", unresolved_cell_time)
    pseudotime = np.array(unresolved_cell_time, dtype=object)
    for node_idx in range(n_nodes):
        cells = pseudotime[node_idx]
        node = nodes[node_idx]
        for cell in cells:
            cells[cell] += w_cumm[node]
    #print("After:\n", unresolved_cell_time)
    return pseudotime

def relative_time_in_tree(paths, w):
    paths = np.array(paths)
    w = np.array(w)

    # Hence starting from the longest path
    nodes = sorted(set(np.array(paths).flatten()))
    # print(nodes)
    n_nodes = len(nodes)
    flag = {node:0 for node in nodes}
    w_cumm = {node:0 for node in nodes}

    if len(paths) == 0:
        return w_cumm

    MAX_ITER = 10*n_nodes
    # by Guangyu Wang
    node = nodes[0]
    flag[node] = 1
    iterations = 0
    while 0 in flag.values():
        iterations += 1
        if iterations > MAX_ITER:
            print("There are cycle(s), forcing a break.")
            break
        for path in paths:
            if path[0]==node:
                #print("Forward: "+str(node)+" -> "+str(path[1]))
                w_cumm[path[1]] = w_cumm[node]+w[np.all(paths==path, axis=1)][0]
                node = path[1]
                flag[node] = 1

            elif path[1]==node:
                #print("Backward: "+str(node)+" -> "+str(path[0]))
                w_cumm[path[0]] = w_cumm[node]-w[np.all(paths==path, axis=1)][0]
                node=path[0]
                flag[node] = 1
            else:
                #print("Pass: "+str(node))
                pass
#    print(w_cumm)
    return w_cumm

# combine cell time from clusters
def combine_clusters(cell_time_per_cluster):
    cell_time = dict()
    for d in cell_time_per_cluster:
        for k, v in d.items():
            cell_time[k] = v
    return cell_time


def interpolate_all_cell_time(cell_time, all_cell_embedding, sampling_ixs,
        n_grids):
    x = all_cell_embedding[sampling_ixs,0]
    y = all_cell_embedding[sampling_ixs,1]

    xx = np.linspace(min(x), max(x), n_grids[0]+1)
    yy = np.linspace(min(y), max(y), n_grids[1]+1)
    xx, yy = np.meshgrid(xx, yy)
    
    points = np.transpose(np.vstack((x, y)))
    interp = interpolate.griddata(points, cell_time, (xx, yy), method='nearest')
    all_cell_time = list()
    for cell_coord in all_cell_embedding:
        gd = discretize(cell_coord, xmin=(min(x), min(y)), xmax=(max(x),max(y)),
                n_grids=n_grids, capping = True)[0]
        all_cell_time.append(interp[gd[1], gd[0]])

    # drop the top 5 percentile
    all_cell_time = np.array(all_cell_time)
    all_cell_time[all_cell_time>np.quantile(all_cell_time, 0.95)]=np.quantile(all_cell_time, 0.95)
    
    # smoothing the data using the nearest neighbours
    n_neighbors = int(len(all_cell_time)*0.05)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=1, n_jobs=-1)
    neigh.fit(all_cell_embedding)
    A = neigh.kneighbors_graph(all_cell_embedding)
    B = A.toarray()

    all_cell_time_smooth = [np.mean(all_cell_time[B[i]!=0]) for i in
            range(len(all_cell_time))]
    all_cell_time_smooth -= np.min(all_cell_time_smooth)
    all_cell_time_smooth = all_cell_time_smooth/np.max(all_cell_time_smooth)
    return all_cell_time_smooth


def export_cell_time(cell_time, cell_fate, sampling_ixs, filename): 
    sample = np.array([True if i in sampling_ixs else False for i in
        range(len(cell_fate))], dtype=bool)
    data = np.vstack((range(len(cell_fate)), cell_fate, cell_time, sample)).T
    df = pd.DataFrame(data, columns = ['cellindex', 'traj_cluster', 'pseudotime',
    'downsampled'])
    df = df.astype({"cellindex": int, "traj_cluster": int, "pseudotime": float,
        "downsampled": bool})
    df.to_csv(filename, index=False)
        

def overlap_crit_intracluster(cell_embedding, cell_fate_dict, quant):
    """
    Calculate the cutoff distance (in embedding space) according to intracluster distances.
    """
    cutoff = list()

    cell_ID = [cell for cell, fate in cell_fate_dict.items()]
    cell_fate = [cell_fate_dict[cell] for cell in cell_ID]
    cell_embedding_sub = cell_embedding[cell_ID]
    for cluster_ID in np.unique(cell_fate):
        cell_cluster = cell_embedding_sub[cell_fate == cluster_ID]
        temp1 = cell_cluster - cell_cluster[:, None]
        temp2 = np.linalg.norm(temp1, axis=-1)
        
        # drop the self distances
        temp3 = temp2[~np.eye(temp2.shape[0], dtype=bool)]
        cutoff.append(np.quantile(temp3, quant))
    return max(cutoff)


def find_uniq_cell_pairs(pairs, distances):
    '''
    Parameters
    ----------
    pairs: tuple (np.where output format) 
    distances: 1d np.array <- pairwise distance 
    Return
    ------
    np.ndarray
    '''
    
    order = np.argsort(distances)
    ordered_pairs = np.array(pairs).T[order]

    fbd_cell_a = list()
    fbd_cell_b = list()
    uniq_pair = list()
    for pair in ordered_pairs:
        if pair[0] in fbd_cell_a or pair[1] in fbd_cell_b:
            continue
        else:
            uniq_pair.append(pair)
            fbd_cell_a.append(pair[0])
            fbd_cell_b.append(pair[1])
    return np.array(uniq_pair)


def overlap_intercluster(
        cell_embedding, 
        cell_fate_dict, 
        cell_time_per_cluster, 
        clusterIDs,
        cluster0_index, 
        cluster1_index, 
        cutoff, 
        BARELY_OVERLAP=5, 
        peak_mode='most_frequent_shift'):
    '''
    returns the indices of overlapping cells in pairs and dT.
    '''

    cluster0_ID = clusterIDs[cluster0_index]
    cluster1_ID = clusterIDs[cluster1_index]
    #print("\nConsolidating time between clusters ", 
    #        cluster0_ID, " and ", cluster1_ID, "...")

    cluster0_cellID = list()
    cluster1_cellID = list()
    for cell, fate in cell_fate_dict.items():
        if fate == cluster0_ID:
            cluster0_cellID.append(cell)
        if fate == cluster1_ID:
            cluster1_cellID.append(cell)

    cluster0_cellID = np.array(cluster0_cellID)
    cluster1_cellID = np.array(cluster1_cellID)

    cell_cluster0 = cell_embedding[cluster0_cellID]
    cell_cluster1 = cell_embedding[cluster1_cellID]
    if cell_cluster0.size and cell_cluster1.size:
        temp1 = cell_cluster0 - cell_cluster1[:, None]
        temp2 = np.linalg.norm(temp1, axis=-1)
        closePairs = np.where(temp2 < cutoff)
        
        #print(closePairs)
        if len(closePairs[0]) == 0:
            #print("No close cells between clusters\t", 
            #        (cluster0_ID, cluster1_ID))
            return None, []
        #print("\ncells in ", cluster0_index, ": \n", cluster0_cellID[closePairs[1]])
        #print("\ncells in ", cluster1_index, ": \n", cluster1_cellID[closePairs[0]])
        
        # 1 cell can pair maximum 1 cell.
        idx = find_uniq_cell_pairs(closePairs, temp2[closePairs])

        # Swap so that:
        # first column -> cluster0
        # second column -> cluster1
        idx[:,[1,0]] = idx[:,[0,1]]
    
        #print(cell_time_per_cluster)
        deltaT = dict()
        for pair in idx:
            pair_cellIDs=(cluster0_cellID[pair[0]], cluster1_cellID[pair[1]])
            #print("pair_cellIDs: ", pair_cellIDs)
            deltaT[pair_cellIDs] = \
                    cell_time_per_cluster[cluster0_index][pair_cellIDs[0]] \
                    -cell_time_per_cluster[cluster1_index][pair_cellIDs[1]] 

        deltaT_values = np.array(list(deltaT.values()))
        
        # If there are too few overlaps, 
        # use the pair with smallest absolute time difference.
        if len(deltaT_values) < BARELY_OVERLAP:
            peak_mode = 'least_shift'

        #print("\nPeak mode: ", peak_mode)
        if peak_mode in ['least_shift']:
            #fig, axes = plt.subplots(nrows=1, ncols=2, 
            #        gridspec_kw={'width_ratios':[1,1]}, figsize=(8,4))
            #axes[0].title.set_text('overlapping cells between 2 clusters')
            #axes[0].scatter(cell_embedding[:,0], cell_embedding[:,1], s=1, alpha=0.3)
            #axes[0].scatter(cell_cluster0[:,0], cell_cluster0[:,1], s=5,
            #        alpha=0.3, color='orange')
            #axes[0].scatter(cell_cluster1[:,0], cell_cluster1[:,1], s=5,
            #        alpha=0.3, color='cyan')
            #axes[0].scatter(cell_cluster0[idx[:,0]][:,0],
            #        cell_cluster0[idx[:,0]][:,1], color='orange',
            #        edgecolors='k')
            #axes[0].scatter(cell_cluster1[idx[:,1]][:,0],
            #        cell_cluster1[idx[:,1]][:,1], color='cyan',
            #        edgecolors='k')
            #axes[1].title.set_text('histogram of overlapping time difference')
            #sns.histplot(ax=axes[1], data=deltaT_values, kde=False, color='skyblue')
            #plt.show()

            shiftT = deltaT_values[np.argmin(np.abs(deltaT_values))]
            closest_pair = list(deltaT.keys())[list(deltaT.values()).index(shiftT)]

        elif peak_mode in ['most_frequent_shift']:
#            fig, axes = plt.subplots(nrows=1, ncols=2, 
#                    gridspec_kw={'width_ratios':[1,1]}, figsize=(8,4))

            #print("Unique close pairs\n", idx)
#            axes[0].title.set_text('overlapping cells between 2 clusters')
#            axes[0].scatter(cell_embedding[:,0], cell_embedding[:,1], s=1, alpha=0.3)
#            axes[0].scatter(cell_cluster0[:,0], cell_cluster0[:,1], s=5,
#                    alpha=0.3, color='orange')
#            axes[0].scatter(cell_cluster1[:,0], cell_cluster1[:,1], s=5,
#                    alpha=0.3, color='cyan')
#            axes[0].scatter(cell_cluster0[idx[:,0]][:,0],
#                    cell_cluster0[idx[:,0]][:,1], color='orange',
#                    edgecolors='k')
#            axes[0].scatter(cell_cluster1[idx[:,1]][:,0],
#                    cell_cluster1[idx[:,1]][:,1], color='cyan',
#                    edgecolors='k')
#            axes[1].title.set_text('histogram of overlapping time difference')
#            sns.histplot(ax=axes[1], data=deltaT_values, kde=True, color='skyblue')
#            try:
#                kdeline = axes[1].lines[0]
#                x = kdeline.get_xdata()
#                y = kdeline.get_ydata()
#                mode_idx = np.argmax(y)
#                axes[1].vlines(x[mode_idx], 0, y[mode_idx], color='tomato', ls='--', lw=5)
#                shiftT = x[mode_idx]
#            except IndexError: 
            # When there is only one bin, kdeline does not exist.
#                for p in axes[1].patches:
#                    shiftT = p.get_x()
#                axes[1].vlines(p.get_x(), 0, p.get_height(), color='tomato', ls='--', lw=5)
#            plt.show()

            # find the pair ~ shiftT
#            shiftT = deltaT_values[np.argmin(np.abs(deltaT_values-shiftT))]
            dt_numpy = np.array(sorted(list(deltaT_values)))[:,None]
            kde = KernelDensity(kernel="gaussian").fit(dt_numpy)
            log_density = kde.score_samples(dt_numpy)
            shiftT = dt_numpy[np.argmax(log_density)][0]
            closest_pair = list(deltaT.keys())[list(deltaT.values()).index(shiftT)]
        return shiftT, closest_pair


def assign_all_cell_fate(embedding, sampling_ixs, cell_fate):
    #print(len(cell_fate))
    #print(len(sampling_ixs))
    neigh = NearestNeighbors(n_neighbors=1, radius=20, n_jobs=-1)
    neigh.fit(embedding[sampling_ixs])
    A = neigh.kneighbors_graph(embedding)
    B = A.toarray()
    all_cell_fate = np.array([(B[i][B[i]!=0]*cell_fate[B[i]!=0])[0]
                           for i in range(len(B))], dtype=int)
    return all_cell_fate


def compute_cell_time(
        cellDancer_df, 
        embedding, 
        cell_embedding, 
        path_clusters, 
        cell_fate,
        vel_mesh, 
        cell_grid_idx, 
        grid_mass,
        sampling_ixs, 
        n_grids,
        dt=0.001, 
        t_total=10000, 
        eps=1e-3,
        n_repeats = 10, 
        n_jobs = -1,
        psrng_seeds_diffusion = None):
    
    clusters = np.unique(cell_fate)

    # cell_fate is a list here
    # [cell_0_cluster, cell_1_cluster, ..., cell_n_cluster]
    n_clusters = len(clusters)

    print("There are %d cluster of cells according to the endpoints of the velocity-driven paths." % (n_clusters))
    
    cell_time_per_cluster = [projection_cell_time_one_cluster(cell_embedding, 
        path_clusters[cluster][0], cluster, cell_fate) for cluster in clusters]

    repPath_clusters = [path_clusters[cluster][0] for cluster in clusters] 

    #plot_celltime_clusters(cell_time_per_cluster, repPath_clusters, cell_embedding)

    # intra-cluster time assignment
    clusterIndex=0
    # It is trickly as cell clusters do not 1-to-1 map to the path clusters
    # The clusterIndex is 0, 1, 2, ..., without skips
    # It is used for lists cell_time_per_cluster
    # The cluster could be 0, 2, 3, 6, ... with skips.
    # It is used for path_clusters (because we didn't merge/drop
    # path clusters)

    for clusterID in clusters:
        cell_time_subclusters = list()
        cluster_out = recur_cell_time_assignment_intracluster(
                cell_time_per_cluster[clusterIndex], 
                cell_time_subclusters,
                clusterID, 
                [path_clusters[clusterID][0]], 
                cell_fate, 
                cell_embedding, 
                vel_mesh, 
                cell_grid_idx, 
                grid_mass,
                dt=dt, 
                t_total=t_total, 
                n_repeats=n_repeats, 
                n_jobs=n_jobs,
                psrng_seeds_diffusion = psrng_seeds_diffusion,
                MAX_ALLOWED_ZERO_TIME_CELLS = 0.05,
                MAX_ALLOWED_TERM_CELLS = 0.05)
        cell_time_subclusters, refPaths = cluster_out[0], cluster_out[1]
        # print("Summarizing:")
        # print("ClusterID: ", clusterID)
        # print("number of paths: ", len(refPaths))
        # print("number of subclusters: ", len(cell_time_subclusters))
        # print("\n\n")

        #print("\nDisplay reference paths for cluster", clusterID)
        #fig, ax = plt.subplots(figsize=(6,6))
        #plt.scatter(cell_embedding[:,0], cell_embedding[:,1], alpha = 0.3)
        refPaths = refPaths[:len(cell_time_subclusters)]
        for j in range(len(refPaths)):
            path = refPaths[j]
            #plt.scatter(path[:,0], path[:,1], c=range(len(path)), s=5)
            #plt.text(path[-1,0], path[-1,1], "refPath"+str(j), fontsize=12)
            scell = cell_time_subclusters[j]
            scell = [i for i in scell]
            scell = cell_embedding[scell]
            #plt.scatter(scell[:,0], scell[:,1], s=10, alpha=1)
        #plt.axis('off')
        #plt.show()

        # Need to do inter-subcluster adjustment for all the subclusters in a cluster
        if len(cell_time_subclusters) > 1:
            cells_subclusters = list()
            cell_fate_subclusters_dict = dict()

            # subclusterID starts from 0
            subclusterID = 0
            for subcluster in cell_time_subclusters:
                for cell in subcluster:
                    cells_subclusters.append(cell)
                    cell_fate_subclusters_dict[cell] = subclusterID
                subclusterID += 1

            cell_time_per_cluster[clusterIndex]= \
                    cell_time_assignment_intercluster(
                        cell_time_subclusters,
                        cell_fate_subclusters_dict, 
                        cell_embedding, 
                        tau = 0.05)
            cell_time_per_cluster[clusterIndex] = \
                    combine_clusters(cell_time_per_cluster[clusterIndex])

        clusterIndex += 1

    #print("\n\n\nAll intra cluster cell time has been resolved.\n\n\n")
    
    # inter-cluster time alignment
    cell_fate_dict = {i:cell_fate[i] for i in range(len(cell_fate))}
    resolved_cell_time = cell_time_assignment_intercluster(
            cell_time_per_cluster, 
            cell_fate_dict, 
            cell_embedding, 
            tau = 0.05)

    #print("\n\nAll inter cluster cell time has been resolved.\n\n\n")
    cell_time = combine_clusters(resolved_cell_time)
    ordered_cell_time = np.array([cell_time[cell] for cell in sorted(cell_time.keys())])

    # interpolate to get the time for all cells.
    if n_grids is not None:
        all_cell_time=interpolate_all_cell_time(
                ordered_cell_time, 
                embedding, 
                sampling_ixs, 
                n_grids)
    else:
        all_cell_time=ordered_cell_time

    all_cell_fate = assign_all_cell_fate(embedding, sampling_ixs, cell_fate)
    #print("There are %d cells." % (len(all_cell_fate)))
    #plot_cell_clusters(all_cell_fate, embedding)
    
    # write cell time to cellDancer_df
    gene_names = cellDancer_df['gene_name'].drop_duplicates().to_list()
    if len(cellDancer_df) == len(gene_names) * len(all_cell_time):
        cellDancer_df['pseudotime'] = np.tile(all_cell_time, len(gene_names))
        cellDancer_df = cellDancer_df.astype({"pseudotime": float})
    return cellDancer_df


def pseudo_time(
        cellDancer_df, 
        grid=None, 
        dt=0.05, 
        t_total=200, 
        n_repeats=10,
        psrng_seeds_diffusion=None,
        n_jobs=-1,
        speed_up=(60, 60), 
        n_paths=5,
        plot_long_trajs=False,
        save=False, 
        output_path=None):

    """Compute the gene-shared pseudotime based on the projection of the RNA velocity on the embedding space.
    Arguments
    ---------
    cellDancer_df: `pandas.DataFrame`
        Dataframe of velocity estimation results.
        Columns=['cellIndex', 'gene_name', unsplice', 'splice',
        'unsplice_predict', 'splice_predict',
        'alpha', 'beta', 'gamma', 'loss', 'cellID, 'clusters',
        'embedding1', 'embedding2', 'velocity1', 'velocity2']
    grid: optional, `tuple` (default: `None`)
        (n_x, n_y), where n_x, n_y are integers.
        The embedding space (2d, [xmin, xmax] x [ymin, ymax]) is divided into
        n_x * n_y grids. The cells in the same grid share the same velocity
        (mean), however, they may not share the pseudotime.
        If it's set to `None`, then a recommended value for n_x and n_y is
        the square root of the number of selected cells (rounded to the nearest
        tenth.)
    dt: optional, `float` (default: 0.05)
        Time step used to advance a cell on the embedding for generation of
        cell diffusion trajectories. Parameter `dt` should be set together
        with `t_total`. Excessively small values of `dt` demand large `t_total`,
        and drastically increase computing time; Excessively large values of
        `dt` lead to low-resolution and unrealistic pseudotime estimation. 
    t_total: optional, `float` (default: 200)
        Total number of time steps used for generation of cell diffusion
        trajectories.
        The diffusion is stopped by any of the criteria:
        - reach `t_total`;
        - the magnitude of the velocity is less than a cutoff `eps`;
        - the cell goes to where no cell resides;
        - the cell is out of the diffusion box (the `grid`)
    n_repeats: optional, `int` (default: 10)
        Number of repeated diffusion of each cell used for generation of cell
        diffusion trajectories.
    psrng_seeds_diffusion: optional, `list-like` (default: `None`)
        Pseudo random number generator seeds for all the replicas in the generation
        of cell diffusion trajectories. Its length = `n_repeats`. Set this for
        reproducibility.
    speed_up: optional, `tuple` (default: (60,60))
        The sampling grid used in *compute_cell_velocity.compute*.
        This grid is used for interpolating pseudotime for all cells.
    n_jobs: optional, `int` (default: -1)
        Number of threads or processes used for cell diffusion. It follows the
        scikit-learn convention. -1 means all possible threads.
    n_paths: optional, `int` (default: 5)
        Number of long paths to extract for cell pseudotime estimation.
        Note this parameter is very sensitive. For the best outcome, please set the
        number based on biological knowledge about the cell embedding.
    plot_long_trajs: optional, `bool`(default: False)
        Whether to show the long trajectories whose traverse lengths are
        local maximums.
    save: `bool` (default: `False`)
        Whether to save the pseudotime-included `cellDancer_df` as .csv file.
    output_path: optional, `str` (default: `None`)
        Save file path. By default, the .csv file is saved in the current directory.
        
    Returns
    -------
    cellDancer_df: `pandas.DataFrame`
        The updated cellDancer_df with additional columns ['velocity1', 'velocity2'].
    """

    if output_path is None:
        output_path = os.getcwd()

    start_time = time.time()

    gene_choice = cellDancer_df[~cellDancer_df['velocity1'].isna()]['gene_name']
    gene_choice = gene_choice.drop_duplicates()
    one_gene = gene_choice.to_list()[0]
    embedding = cellDancer_df[cellDancer_df['gene_name'] == 
            one_gene][['embedding1', 'embedding2']]
    embedding = embedding.to_numpy()

    # This could be problematic if it's not in the gene_choice
    velocity_embedding = cellDancer_df[cellDancer_df.gene_name ==
            one_gene][['velocity1', 'velocity2']].dropna()
    sampling_ixs = velocity_embedding.index

    cell_embedding, normalized_embedding = embedding_normalization(
        embedding[sampling_ixs], embedding, mode='minmax', NORM_ALL_CELLS=True)

    velocity = velocity_normalization(velocity_embedding, mode='max')

    activate_umap_paths_divider=False # for development - Whether to use UMAP embedding (calculated from alpha, beta, and gamma) in generation of cell diffusion trajectories.
    if activate_umap_paths_divider:
        cellDancer_df = calculate_para_umap(
                cellDancer_df, 'alpha_beta_gamma')

        abr_umap = cellDancer_df[cellDancer_df['gene_name'] ==
                one_gene][['alpha_beta_gamma_umap1', 'alpha_beta_gamma_umap2']]
        _, normalized_abr_umap = embedding_normalization(
                abr_umap.loc[sampling_ixs], abr_umap, mode='minmax', NORM_ALL_CELLS=True)
    else:
        normalized_abr_umap = None


    if grid is None:
        grid = (int(np.sqrt(len(cell_embedding))), int(np.sqrt(len(cell_embedding))))

    __ = generate_grid(cell_embedding, 
            normalized_embedding,
            velocity, 
            normalized_abr_umap, 
            n_grids=grid)

    vel_mesh = __[0] 
    grid_mass = __[1]
    grid_umap = __[2]
    cell_grid_idx = __[3] 
    cell_grid_coor = __[4]
    all_grid_idx = __[5] 
    all_grid_coor = __[6]
    
    if activate_umap_paths_divider:
        path_divider_matrix = compute_path_divider_matrix(grid_umap, cutoff=0.1)
    else:
        path_divider_matrix = None

    # v_eps is used to stop a trajectory if mag of velocity < v_eps
    v_eps = 1e-3
    if psrng_seeds_diffusion is not None:
        print("Pseudo random number generator seeds are set to: ", \
                psrng_seeds_diffusion)

    paths=run_diffusion(cell_embedding, 
                        vel_mesh, 
                        grid_mass, 
                        dt=dt, 
                        t_total=t_total,
                        eps=v_eps, 
                        off_cell_init=False, 
                        n_repeats = n_repeats, 
                        psrng_seeds_diffusion = psrng_seeds_diffusion,
                        path_divider_matrix = path_divider_matrix,
                        n_jobs = n_jobs)
    
    newPaths = truncate_end_state_stuttering(paths, cell_embedding) 
    traj_displacement = np.array([compute_trajectory_displacement(ipath) for \
            ipath in newPaths])

    # sorted from long to short
    order = np.argsort(traj_displacement)[::-1]
    sorted_traj = newPaths[order]
    traj_displacement=traj_displacement[order]

    def decide_cell_fate(path_similarity):
        path_clusters = dict()
        cell_clusters = dict()
        __ = extract_long_trajectories(
            path_clusters, 
            cell_clusters, 
            sorted_traj, 
            similarity_cutoff=path_similarity,
            similarity_threshold=0, 
            nkeep=-1)
        path_clusters, cell_clusters = __

        # This step could cause a drop in the number of path clusters.
        cell_fate = cell_fate_tuning(cell_embedding, cell_clusters)
        clusters = np.unique(cell_fate)
        n_clusters = len(clusters)
        return n_clusters, cell_fate, path_clusters

    def binary_search(s_high, s_low, n_path_min, n_path_max):
        if n_path_min >= n_paths:
            return s_high
        if n_path_max < n_paths:
            return s_low

        s_mid = (s_high + s_low)/2.
        n_path_mid = decide_cell_fate(s_mid)[0]
        if n_path_mid == n_paths:
            return s_mid
        elif n_path_mid < n_paths:
           return binary_search(s_mid, s_low, n_path_mid, n_path_max)
        else:
            return binary_search(s_high, s_mid, n_path_min, n_path_mid)

    s_high = 0.4
    s_low = 0.1
    n_path_min = decide_cell_fate(s_high)[0] 
    n_path_max = decide_cell_fate(s_low)[0] 
    path_similarity = binary_search(s_high, s_low, n_path_min, n_path_max)

    #print("use path_similarity: ", path_similarity)
    n_clusters, cell_fate, path_clusters = decide_cell_fate(path_similarity)

    # make sure cluster id is continuous from 0 to n_clusters-1
    #cluster_map = dict(zip(np.unique(cell_fate), np.array(range(n_clusters))))
    #cell_fate = [cluster_map[i] for i in cell_fate]

    # show path clusters
    if plot_long_trajs:
        plot_path_clusters(path_clusters, cell_embedding, output_path=output_path)    
    cellDancer_df = compute_cell_time(
        cellDancer_df,
        normalized_embedding, 
        cell_embedding, 
        path_clusters, 
        cell_fate,
        vel_mesh, 
        cell_grid_idx=cell_grid_idx, 
        grid_mass=grid_mass, 
        sampling_ixs=sampling_ixs, 
        n_grids=speed_up,
        dt=dt, 
        t_total=t_total, 
        n_repeats=n_repeats, 
        eps=v_eps,
        n_jobs=n_jobs,
        psrng_seeds_diffusion=psrng_seeds_diffusion)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    if save:
        outname = 'cellDancer'+\
                '_pseudo_time'+ \
                '__grid' + str(grid[0])+'x'+str(grid[1])+ \
                '__dt' + str(dt)+ \
                '__ttotal' + str(t_total)+ \
                '__nrepeats' + str(n_repeats) + \
                '.csv'
        outfile = os.path.join(output_path, outname)

        print("\nExporting data to:\n ", outfile)
        cellDancer_df.to_csv(outfile, index=False)
    return cellDancer_df

    


# all plot functions
def plot_path_clusters(path_clusters, cell_embedding, save=True, output_path=None):    
    '''
    path_clusters: a dictionary of paths (each path is ntimestep x 2 dim)
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    #plt.scatter(cell_embedding[:,0], cell_embedding[:,1], c='silver', s=10, alpha = 0.3)
    n_clusters = len(path_clusters)
    
    cmap = ListedColormap(sns.color_palette("bright", n_colors = n_clusters))
    colormaps = [ListedColormap(sns.light_palette(cmap.colors[i],
        n_colors=100)) for i in range(n_clusters)]

    # find the nearest cell (terminal cell) to the end of each leading path.
    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neigh.fit(cell_embedding)

    cluster_cnt = 0
    for cluster in path_clusters:
        cl = colormaps[cluster_cnt]
        lead_path=path_clusters[cluster][0]
        terminal_cell=lead_path[-1]
        A = neigh.kneighbors_graph(np.array([terminal_cell]))
        B = A.toarray()
        terminal_cell = np.matmul(B, cell_embedding)

        # cells in the cluster
        for _path in path_clusters[cluster]:
            plt.scatter(_path[0,0], _path[0,1], 
                        s=1, alpha=0.5,
                    color=cmap.colors[cluster_cnt])

        # Annotation
#        plt.text(lead_path[-1,0], lead_path[-1,1], 
#                 "Path: "+str(cluster+1), fontsize=12,
#                 bbox=dict(facecolor='white', 
#                           alpha=0.5, 
#                           edgecolor='white', 
#                           boxstyle='round,pad=1')
#                 )

        # path
        plt.scatter(lead_path[:,0], lead_path[:,1], 
                    s = 50,
                    c = range(len(lead_path)), 
                    cmap = colormaps[cluster_cnt]
                    )

        # zoom in the terminal cell
#        plt.scatter(terminal_cell[:,0], terminal_cell[:,1], 
#                    s = 30, alpha = 1,
#                    color = cmap.colors[cluster_cnt])

        cluster_cnt += 1

    plt.axis('off')
    if save:
        save_path = os.path.join(output_path, "pseudotime_rep_trajs.pdf")
        plt.savefig(save_path, dpi=300)
    plt.show()




# Those functions are for debugging purpose
def pseudotime_cell_plot():
    print("\n\n\nPlotting estimated pseudotime for all cells ...")
    fig, ax = plt.subplots(figsize=(6,6))
    im = plt.scatter(all_cell_embedding[:,0], all_cell_embedding[:,1],
            c=all_cell_time_smooth, alpha = 1, s = 1)

    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    cax = plt.colorbar(im,fraction=0.03, pad=0., location='bottom')
    cax.set_label('normalized pseudotime')
    plt.axis('off')
    plt.show()
    
    
def plot_cell_clusters(cell_fate, cell_embedding):
    clusters = np.unique(cell_fate)
    n_clusters = len(clusters)
    cluster_map = dict(zip(clusters, np.array(range(n_clusters))))
    
    cmap = ListedColormap(sns.color_palette("tab10", n_colors = n_clusters))
    fig, ax1 = plt.subplots(figsize=(6, 6))
    img1=ax1.scatter(
            cell_embedding[:,0], 
            cell_embedding[:,1],
            c=[cluster_map[i] for i in cell_fate],
            s=1, 
            alpha=1, 
            cmap=cmap)

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("cell fate: majority votes")
    ax1.axis("off")

    bounds = np.linspace(0, n_clusters, n_clusters+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax3 = fig.add_axes([0.9, 0.3, 0.02, 0.3])
    cb = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, 
            spacing='proportional', 
            boundaries=bounds, 
            norm=norm, 
            format='%1i')

    labels = ["cluster "+str(i) for i in clusters]

    cb.ax.get_yaxis().set_ticks([])
    for i, label in enumerate(labels):
        cb.ax.text(4.5, i + 0.5 , label, ha='center', va='center')
    plt.show()


def plot_celltime_clusters(cell_time_per_cluster, path_clusters, embedding):

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(embedding[:,0],embedding[:,1], c='silver', alpha = 0.3)

    n_paths = len(path_clusters)
    cmap = ['viridis'] * n_paths

    for i in range(n_paths):
        colormap = cmap[i]
        cell_index = list(cell_time_per_cluster[i].keys())
        cell_time = list(cell_time_per_cluster[i].values())
        cells = embedding[cell_index]
        lead_path = path_clusters[i]
        
        plt.scatter(cells[:,0], cells[:,1], 
                c=cell_time, s=20, cmap = colormap, alpha =0.8)
        plt.scatter(lead_path[:,0], lead_path[:,1], 
                c=range(len(lead_path)), s=5, cmap = 'Reds_r')
        ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()
