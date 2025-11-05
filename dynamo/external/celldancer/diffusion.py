#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import multiprocessing as mp

import numpy as np
from sklearn import preprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt

def embedding_normalization(cell_embedding, embedding=None, mode="minmax", NORM_ALL_CELLS=False):
    '''
    Normalize by the maximum absolute value.
    
    Parameters
    ----------
    embedding: 2D numpy array (n_cells, 2)
    mode: string
          'maxabs', "minmax"
    maxabs is meant for sparse data and/or centered at 0. 
    Note in this program (ML velocity), it is pretty safe to do maxabs normalization
    since the data are free of extreme outliers.
     
    '''
    if mode in ['max', 'maximum', 'maxabs']:
        transformer = preprocessing.MaxAbsScaler().fit(cell_embedding)
    elif mode in ['minmax']:
        transformer = preprocessing.MinMaxScaler().fit(cell_embedding)
    em = transformer.transform(cell_embedding)
    if NORM_ALL_CELLS:
        try:
            em_all = transformer.transform(embedding)
        except ValueError:
            print("ERROR! Missing embedding for all cells.")
            raise
        return em, em_all
    else:
        return em
    
def velocity_normalization(downsampled_vel, all_vel=None, mode="max", NORM_ALL_CELLS=False):
    '''
    Normalize by the maximum absolute value in the downsampled_vel.
    
    Parameters
    ----------
    downsampled_vel: 2D numpy array (n_cells, 2)
    mode: 'maxabs'
    
    maxabs is meant for sparse data and/or centered at 0. 
    
    Note in this program, it is pretty safe to do maxabs normalization
    since the data are free of extreme outliers.
     
    '''
    # add v_prime to vel of each cell without changing their directions.
    v_mag = np.linalg.norm(downsampled_vel, axis=1)
    v_prime = 0.1*np.std(v_mag)

    # for 0 velocity cell, nothing changed.
    v_prime = np.divide(v_prime, v_mag, where=v_mag > 0)
    downsampled_vel = downsampled_vel*(v_prime + 1)[:,None]

    if mode in ['max', 'maximum', 'maxabs']:
        transformer = preprocessing.MaxAbsScaler().fit(downsampled_vel)
    em = transformer.transform(downsampled_vel)
    if NORM_ALL_CELLS:
        em_all = transformer.transform(all_vel)
        return em, em_all
    else:
        return em
    

def discretize(coordinate, xmin, xmax, n_grids, capping=False):
    '''
    '''
    grid_size = np.array(xmax) - np.array(xmin)
    grid_size = grid_size / np.array(n_grids)

    grid_idx = np.int64(np.floor((coordinate-xmin)/grid_size))
    
    if capping:
        grid_idx = np.where(grid_idx > n_grids, n_grids, grid_idx)
        grid_idx = np.where(grid_idx <0, 0, grid_idx)
    
    grid_coor = xmin + grid_size * (grid_idx+0.5)
    return grid_idx, grid_coor 


def generate_grid(
        cell_embedding, 
        embedding, 
        velocity_embedding, 
        abr_umap = None, 
        n_grids = None):

    xmin = np.min(cell_embedding, axis=0)
    xmax = np.max(cell_embedding, axis=0)
    n_grids = np.array(n_grids, dtype=int)

    cell_grid_idx, cell_grid_coor = discretize(cell_embedding, 
            xmin=xmin, 
            xmax=xmax, 
            n_grids=n_grids)

    # The actual n_grids need to allow a leeway +1 in each dimension.
    mesh = np.zeros(np.append(n_grids+1,len(n_grids)))

    cnt = np.zeros(n_grids+1)
    for index in range(cell_grid_idx.shape[0]):
        grid_index = cell_grid_idx[index]
        if np.any(grid_index > n_grids) or np.any(grid_index < 0):
            continue
        grid_index = toTuple(grid_index)
        mesh[grid_index] += velocity_embedding[index]
        cnt[grid_index] += 1
    cnt = cnt[:,:,None]
    mesh = np.divide(mesh, cnt, out=np.zeros_like(mesh), where=cnt>0.1)
    
    # the all cell embedding is used to generate mass
    mass = np.zeros(n_grids+1)
    all_cells_grid_idx, all_cells_grid_coor = \
            discretize(embedding, xmin=xmin, xmax=xmax, n_grids=n_grids)
    n_cells = all_cells_grid_idx.shape[0]

    for index in range(n_cells):
        all_cells_grid_index = all_cells_grid_idx[index]
        
        # mass outside the grid is not needed.
        if np.any(all_cells_grid_index > n_grids) or np.any(all_cells_grid_index < 0):
            continue
        all_cells_grid_index = toTuple(all_cells_grid_index)
        mass[all_cells_grid_index] += 1

    # the all cell embedding is used to generate grid_umap
    if abr_umap is not None:
        grid_umap = np.full_like(mesh, np.NAN)
        n_umap_dims = all_cells_grid_idx.shape[-1]
        for index in range(n_cells):
            all_cells_grid_index = all_cells_grid_idx[index]
            if np.any(all_cells_grid_index > n_grids) or np.any(all_cells_grid_index < 0):
                all_cells_grid_index = toTuple(all_cells_grid_index)
                grid_umap[all_cells_grid_index] = np.full((1,n_umap_dims), np.NAN)
                pass
            all_cells_grid_index = toTuple(all_cells_grid_index)
            if np.any(np.isnan(grid_umap[all_cells_grid_index])):
                grid_umap[all_cells_grid_index] = np.full((1,n_umap_dims), 0)
            else:
                grid_umap[all_cells_grid_index] += abr_umap[index,:]

        # divide by 0 does not happen
        # because where-ever mass is 0, grid_umap is nan. nan/0 -> nan
        grid_umap = np.divide(grid_umap, mass[:,:,None])

    else:
        grid_umap = None

    return mesh, mass, grid_umap, \
            cell_grid_idx, cell_grid_coor, all_cells_grid_idx, all_cells_grid_coor


def toTuple(arr):
    '''
    Parameters
    ----------
    arr: numpy ndarray or list

    Return
    ------
    A tuple (of nested tuples)

    '''

    try:
        return tuple(toTuple(i) for i in arr)
    except TypeError:
        return arr


def compute_path_divider_matrix(fmat, cutoff=0.3):


    print("The cutoff for banning a path is ", cutoff)
    ngrids = fmat.shape[:-1]
    flat_length = np.multiply(*ngrids)
    temp = fmat.reshape(flat_length, fmat.shape[-1])

    temp2 = temp-temp[:,None]
    temp2 = np.linalg.norm(temp2, axis=-1)

    ban = temp2.reshape(ngrids+ngrids)

    path_divider_matrix = ban < cutoff
    return path_divider_matrix


def plot_velocity(embedding, velocity_embedding):
    fig, ax = plt.subplots(figsize=(6,6))
    plt.quiver(embedding[:, 0],embedding[:, 1],
              velocity_embedding[:,0], velocity_embedding[:,1], 
              color='Blue')
    plt.show()

def plot_mesh_velocity(mesh, grid_mass):
    x=list()
    y=list()
    vx=list()
    vy=list()
    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            x.append(i)
            y.append(j)
            vx.append(mesh[i,j][0])
            vy.append(mesh[i,j][1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(x,y,vx,vy,color='red',scale = 10)
    plt.imshow(grid_mass.T, interpolation=None, origin='lower',cmap="Greys")
    plt.show()

def velocity_add_random(velocity, theta):
    '''
    Rotate the velocity according to a randomized kicks on the perpendicular direction.
    The direction is determined by the sign of a random number. 
    The magnitude of the perpendicular kick is determined by the random number 
    from a normal distribution N(0, theta).
    Magnitude of the velocity is kept the same to conserve energy (temperature) of the system.
    
    Parameters
    ----------
    velocity
        velocity of the grid
    theta
        the angular range that the noise could be affecting the direction of the velocity
        
    WARNING
        at a rare chance, the rotation angle (magnitude) could be much larger than theta.
        
    Return
    ------
    Adjusted velocity for the interested cell
        
    '''
    r = np.random.normal(0, theta, 1)
#    print(mp.current_process(), r)

    cosine = np.cos(r)[0]
    sine = np.sin(r)[0]
    
    # Rotation matrix
    R = np.array([[cosine, sine],[-sine, cosine]])
    velocity = np.dot(velocity, R)
    return velocity

def velocity_rotation(velocity, theta):
    '''
    Rotate the velocity clockwise by angle theta
    
    Parameters
    ----------
    velocity
        velocity of the grid
    theta
        the angular range that the noise could be affecting the direction of the velocity
        
    Return
    ------
    Adjusted velocity for the interested cell
        
    '''
    cosine = np.cos(theta)
    sine = np.sin(theta)
    
    # Rotation matrix
    R = np.array([[cosine, sine],[-sine, cosine]])
    velocity = np.dot(velocity, R)
    return velocity


def diffusion_off_grid_wallbound(
        cell_embedding, 
        vel, 
        init, 
        grid_mass,
        dt = 0.001, 
        t_total = 10000, 
        eps = 1e-5,
        random_seed = None,
        pdm = None):
    
    '''
    Simulate the diffusion of a cell in the velocity field (off grid), the
    cell's velocity will turn 30 degrees
    if it hits the boundary the next timestep.

    The diffusion is stopped by any of the criteria:
    - reach t_total
    - the magnitude of the velocity is less than eps.
    - the cell goes to places where the cell mass <= MAX_IGNORED_MASS even after turning.
    - the cell is out of the simulation box

    Parameters
    ----------
    
    cell_embedding: numpy ndarray (n_cells x n_dims)
        embedding coordinate for all the cells (downsampled)

    vel: numpy ndarray (n_grids x n_dims)
        pre-assigned velocity of each grid

    init: numpy ndarray (n_cells x n_dims)
        The initial position (cell_embedding)

    dt: float 
        Step size of each integration time step

    t_total: int
        Total number of time steps

    grid_mass: numpy ndarray (n_grids x n_dims)
        mass of cells.

    eps 
        Criterion to stop a trajectory before t_total (v_net < eps)

    
    Return
    ------
        a numpy ndarray of coordinates in the trajectory, shape:
        (real_n_time_steps, n_dims)
    '''
    
    np.random.seed(seed = random_seed)
#    print("random seed is set to, ", random_seed)
    THETA = np.pi/6
    
    XMIN = np.min(cell_embedding, axis=0)
    XMAX = np.max(cell_embedding, axis=0)
    N_GRIDS=(vel.shape[0]-1,vel.shape[1]-1)

    # lower 5% nonzero mass set to 0.
    #MAX_IGNORED_MASS= np.percentile(grid_mass[grid_mass>0], 5)
    MAX_IGNORED_MASS = 2
    
    def no_cells_around(xcur, xcur_d, vcur):
        xnxt = xcur + vcur*dt
        xnxt_d, dummy = discretize(xnxt, xmin=XMIN, xmax=XMAX, n_grids=N_GRIDS)
        try:
            mass = grid_mass[xnxt_d[0], xnxt_d[1]]
        except IndexError:
            return True
        return mass <= MAX_IGNORED_MASS
   
    x0 = init
    x0_d, dummy = discretize(x0, xmin=XMIN, xmax=XMAX, n_grids=N_GRIDS)
    v0 = vel[x0_d[0],x0_d[1]]
    v0 = velocity_add_random(v0, THETA)
    trajectory = [x0]
    
    for i in range(int(t_total)):
    
        if np.linalg.norm(v0) < eps:
            #print("Velocity is too small")
            return np.array(trajectory)
        if no_cells_around(x0, x0_d, v0):
            v0_cc = velocity_rotation(v0, THETA)
            v0_c = velocity_rotation(v0, -THETA)

            # nowhere to go but null
            CC = no_cells_around(x0, x0_d, v0_cc) 
            C = no_cells_around(x0, x0_d, v0_c)

            if CC and C:
                return np.array(trajectory)
            elif not C:
                v0 = v0_c
            else:
                v0 = v0_cc
                
        else:
            x = x0 + v0*dt
            x_d, dummy = discretize(x, xmin=XMIN, xmax=XMAX, n_grids=N_GRIDS)
            if (pdm is None) or (pdm[toTuple(x0_d)+toTuple(x_d)]):
                try:
                    v = vel[x_d[0],x_d[1]]
                    mass = grid_mass[x_d[0],x_d[1]]
                    v = velocity_add_random(v, THETA)
                except IndexError:
                    break
            
                trajectory.append(x)
                x0 = x
                v0 = v

    return np.array(trajectory)


def diffusion_on_grid_wallbound(
        cell_embedding, 
        vel, 
        init, 
        grid_mass,
        dt=0.001, 
        t_total=10000, 
        eps = 1e-5):
    
    '''
    same as diffusion_off_grid_wallbound, however, it returns the coordinates
    of the grid traversed by the cell, instead of the position of the cell.

    The diffusion is stopped by any of the criteria:
    1. reach t_total
    2. the magnitude of the velocity is less than eps.
    3. the cell goes to places where the cell mass = 0 even after turning.
    4. the cell is out of the simulation box

    Parameters
    ----------
    
    cell_embedding: numpy ndarray (n_cells x n_dims)
        embedding coordinate for all the cells (downsampled)

    vel: numpy ndarray (n_grids x n_dims)
        pre-assigned velocity of each grid

    init: numpy ndarray (n_cells x n_dims)
        The initial position (cell_embedding)

    dt: float 
        Step size of each integration time step

    t_total: int
        Total number of time steps

    grid_mass: numpy ndarray (n_grids x n_dims)
        mass of cells.

    eps 
        Criterion to stop a trajectory before t_total (v_net < eps)

    
    Return
    ------
        a numpy ndarray of coordinates in the trajectory, shape:
        (real_n_time_steps, n_dims)
    '''
    
    THETA = np.pi/6
    
    XMIN = np.min(cell_embedding, axis=0)
    XMAX = np.max(cell_embedding, axis=0)
    N_GRIDS=(vel.shape[0]-1,vel.shape[1]-1)
    
    # lower 5% nonzero mass set to 0.
    MAX_IGNORED_MASS= np.percentile(grid_mass[grid_mass>0],5)

    def no_cells_around(xcur, xcur_d, vcur):
        xnxt = xcur + vcur*dt
        xnxt_d, dummy = discretize(xnxt, xmin=XMIN, xmax=XMAX, n_grids=N_GRIDS)
        try:
            mass = grid_mass[xnxt_d[0], xnxt_d[1]]
        except IndexError:
            return True
        return mass < MAX_IGNORED_MASS
   
    x0 = init
    x0_d, x0_d_coor = discretize(x0, xmin=XMIN, xmax=XMAX, n_grids=N_GRIDS)
    v0 = vel[x0_d[0],x0_d[1]]
    v0 = velocity_add_random(v0, THETA)
    trajectory = [x0_d_coor]
    
    for i in range(int(t_total)):
    
        if np.linalg.norm(v0) < eps:
            #print("Velocity is too small")
            return np.array(trajectory)
        if no_cells_around(x0_d_coor, x0_d, v0):
            v0_cc = velocity_rotation(v0, np.pi/2)
            v0_c = velocity_rotation(v0, -np.pi/2)
            # nowhere to go but null
            CC = no_cells_around(x0_d_coor, x0_d, v0_cc) 
            C = no_cells_around(x0_d_coor, x0_d, v0_c)
            if CC and C:
                return np.array(trajectory)
            elif not C:
                v0 = v0_c
            else:
                v0 = v0_cc
                
        else:
            x = x0_d_coor + v0*dt
            x_d, x_d_coor = discretize(x, xmin=XMIN, xmax=XMAX, n_grids=N_GRIDS)
            try:
                v = vel[x_d[0],x_d[1]]
                v = velocity_add_random(v, THETA)
            except IndexError:
                break
            
            trajectory.append(x_d_coor)
            x0 = x_d
            x0_d_coor = x_d_coor
            v0 = v

    return np.array(trajectory)


def run_diffusion(
        cell_embedding, 
        vel, 
        grid_mass, 
        dt, 
        t_total = 10000, 
        eps = 1e-5, 
        off_cell_init = False, 
        init_cell = [], 
        n_repeats = 10, 
        n_jobs = 8, 
        psrng_seeds_diffusion = None,
        path_divider_matrix=None):
    '''
    Simulation of diffusion of a cell in the velocity field (on grid), 
    the cell's velocity will turn 90 degrees if it hits the boundary the next timestep.
    Embarrassingly parallel (process) are employed.
    
    Parameters
    ----------
    
    cell_embedding: numpy.ndarray (n_cells, 2)
        embedding coordinate for all the cells (downsampled)
        
    vel: numpy.ndarray (ngrid, ngrid, 2)
        pre-assigned velocity of each grid
    
    dt: float
        Step size of each integration time step
    
    t_total: int
        Total number of time steps
    
    eps: float
        Criterion to stop a trajectory before t_total (v_net < eps)
    
    off_cell_init: Boolean
        Whether to spawn initial coordinates from the neighbouring space around a cell
        
    init_cell: list
        List of initial cell indices. If empty list, use all cell indices in the given cell_embedding.
    
    n_repeats: init
        Number of repeats (either on or off the cells)
    
    n_jobs: int
        Number of threads
    
    Return
    ------
        a numpy array of trajectorys,  shape: (num_trajs, *n_time_steps, 2)
    '''
    import tqdm

    if psrng_seeds_diffusion is None:
        psrng_seeds_diffusion = [i*100+11 for i in range(n_repeats)]
        
    assert len(psrng_seeds_diffusion) >= n_repeats

    if n_jobs >= mp.cpu_count():
        n_jobs = mp.cpu_count()

    if n_jobs < 0:
        n_jobs = mp.cpu_count() + 1 + n_jobs

    TASKS = list()
    # Setting up the TASKS
    n_cells = cell_embedding.shape[0]
    
    if not init_cell:
        init_cell = list(range(n_cells))

    embedding_range = cell_embedding.max(axis=0) - cell_embedding.min(axis=0)
    n_grids = np.array([vel.shape[0], vel.shape[1]])
    grid_size = embedding_range/n_grids
    
    n_trajs = 0 
    for i in init_cell:
        for j in range(n_repeats):
            n_trajs += 1
            if off_cell_init:
                init_position = cell_embedding[i] + grid_size * np.random.uniform(-0.5,0.5,2)
            else:
                init_position = cell_embedding[i]
            TASKS.append((cell_embedding, vel, init_position, grid_mass, dt,
                t_total, 1e-5, psrng_seeds_diffusion[n_trajs % n_repeats],
                path_divider_matrix))
    
    with mp.Pool(n_jobs) as pool:
        n_total = len(init_cell)*n_repeats
        if n_total > 5000:
            paths = pool.starmap(diffusion_off_grid_wallbound, 
                    tqdm.tqdm(TASKS, total=n_total, 
                        desc="Generating Trajectories", 
                        colour="blue")
                    )
        else:
            paths = pool.starmap(diffusion_off_grid_wallbound, TASKS) 
    return np.array(paths, dtype=object)
