import os
import sys
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sys.path.append('.')
    from sampling import *
else:
    try:
        from .sampling import *
    except ImportError:
        from sampling import *


def compute_cell_velocity(
    cellDancer_df,
    gene_list=None,
    speed_up=(60,60),
    expression_scale=None,
    projection_neighbor_size=200,
    projection_neighbor_choice='embedding'):

    """Project the RNA velocity onto the embedding space.
        
    Arguments
    ---------
    cellDancer_df: `pandas.DataFrame`
        Dataframe of velocity estimation results. Columns=['cellIndex', 'gene_name', unsplice', 'splice', 'unsplice_predict', 'splice_predict', 'alpha', 'beta', 'gamma', 'loss', 'cellID, 'clusters', 'embedding1', 'embedding2']
    gene_list: optional, `list` (default: None)
        Genes selected to calculate the cell velocity. `None` if all genes in the cellDancer_df are to be used.
    speed_up: optional, `tuple` (default: (60,60))
        Speed up by giving the sampling grid to downsample cells. 
        `None` if all cells are used to compute cell velocity. 
    expression_scale: optional, `str` (default: None)
        `None` if no expression scale is to be used. 
        `'power10'` if the 10th power is been used to scale spliced and unspliced reads.
    projection_neighbor_size: optional, `int` (default: '200')
        The number of neighboring cells used for the transition probability matrix for one cell.
    projection_neighbor_choice: optional, `str` (default: 'embedding')
        `'embedding'` if using the embedding space to obtain the neighbors. 
        `'gene'` if using the spliced reads of all genes to obtain the neighbors.

    Returns
    -------
    cellDancer_df: `pandas.DataFrame`
        The updated cellDancer_df with additional columns ['velocity1', 'velocity2'].
    """

    def velocity_correlation(cell_matrix, velocity_matrix):
        """Calculate the correlation between the predict velocity (velocity_matrix[:,i])
        and the difference between a cell and every other (cell_matrix - cell_matrix[:, i])

        Arguments
        ---------
        cell_matrix: np.ndarray (ngenes, ncells)
            gene expression matrix
        velocity_matrix: np.ndarray (ngenes, ncells)
        Return
        ---------
        c_matrix: np.ndarray (ncells, ncells)
        """
        c_matrix = np.zeros((cell_matrix.shape[1], velocity_matrix.shape[1]))
        for i in range(cell_matrix.shape[1]):
            c_matrix[i, :] = corr_coeff(cell_matrix, velocity_matrix, i)[0, :]
        np.fill_diagonal(c_matrix, 0)
        return c_matrix


    def velocity_projection(cell_matrix, velocity_matrix, embedding, knn_embedding):
        '''
        cell_matrix: np.ndarray (ngenes, ncells)
            gene expression matrix
        velocity_matrix: np.ndarray (ngenes, ncells)
        '''
        # cell_matrix = np_splice[:,sampling_ixs]
        # velocity_matrix = np_dMatrix[:,sampling_ixs]
        sigma_corr = 0.05
        cell_matrix[np.isnan(cell_matrix)] = 0
        velocity_matrix[np.isnan(velocity_matrix)] = 0
        corrcoef = velocity_correlation(cell_matrix, velocity_matrix)
        probability_matrix = np.exp(corrcoef / sigma_corr)*knn_embedding.A
        probability_matrix /= probability_matrix.sum(1)[:, None]
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
            np.fill_diagonal(unitary_vectors[0, ...], 0)
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        velocity_embedding = (probability_matrix * unitary_vectors).sum(2)
        velocity_embedding -= (knn_embedding.A * unitary_vectors).sum(2) / \
            knn_embedding.sum(1).A.T  # embedding_knn.A *
        velocity_embedding = velocity_embedding.T
        return velocity_embedding
    
    # remove invalid prediction
    is_NaN = cellDancer_df[['alpha','beta']].isnull()
    row_has_NaN = is_NaN. any(axis=1)
    cellDancer_df = cellDancer_df[~row_has_NaN].reset_index(drop=True)
    
    if 'velocity1' in cellDancer_df.columns:
        del cellDancer_df['velocity1']
    if 'velocity2' in cellDancer_df.columns:
        del cellDancer_df['velocity2']
    
    if gene_list is None:
        gene_list=cellDancer_df.gene_name.drop_duplicates()


    # This creates a new dataframe
    cellDancer_df_input = cellDancer_df[cellDancer_df.gene_name.isin(gene_list)].reset_index(drop=True)
    np_splice_all, np_dMatrix_all= data_reshape(cellDancer_df_input)
    # print("(genes, cells): ", end="")
    # print(np_splice_all.shape)
    n_genes, n_cells = np_splice_all.shape

    # This creates a new dataframe
    data_df = cellDancer_df_input.loc[:, 
            ['gene_name', 'unsplice', 'splice', 'cellID','embedding1', 'embedding2']]
    # random.seed(10)
    embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(data_df,
                                                                                 para='neighbors',
                                                                                 target_amount=0,
                                                                                 step=speed_up,
                                                                                 n_neighbors=projection_neighbor_size,
                                                                                 projection_neighbor_choice=projection_neighbor_choice,
                                                                                 expression_scale=expression_scale,
                                                                                 pca_n_components=None,
                                                                                 umap_n=None,
                                                                                 umap_n_components=None)
    

    # projection_neighbor_choice only provides neighborlist, use embedding(from raw data) to compute cell velocity
    embedding = cellDancer_df_input[cellDancer_df_input.gene_name == 
            gene_list[0]][['embedding1', 'embedding2']]
    embedding = embedding.to_numpy()
    velocity_embedding = velocity_projection(
            np_splice_all[:, sampling_ixs], 
            np_dMatrix_all[:, sampling_ixs], 
            embedding[sampling_ixs, :], 
            knn_embedding)

    if set(['velocity1','velocity2']).issubset(cellDancer_df.columns):
        print("Caution! Overwriting the \'velocity\' columns.") 
        cellDancer_df.drop(['velocity1','velocity2'], axis=1, inplace=True)

    sampling_ixs_all_genes = cellDancer_df_input[cellDancer_df_input.cellIndex.isin(sampling_ixs)].index
    cellDancer_df_input.loc[sampling_ixs_all_genes,'velocity1'] = np.tile(velocity_embedding[:,0], n_genes)
    cellDancer_df_input.loc[sampling_ixs_all_genes,'velocity2'] = np.tile(velocity_embedding[:,1], n_genes)
    # print("After downsampling, there are ", len(sampling_ixs), "cells.")
    return(cellDancer_df_input)

def corr_coeff(ematrix, vmatrix, i):
        '''
        Calculate the correlation between the predict velocity (velocity_matrix[:,i])
        and the displacement between a cell and every other (cell_matrix - cell_matrix[:, i])
        ematrix = cell_matrix
        vmatrix = velocity_matrix
        '''
        ematrix = ematrix.T
        vmatrix = vmatrix.T
        ematrix = ematrix - ematrix[i, :]
        vmatrix = vmatrix[i, :][None, :]
        ematrix_m = ematrix - ematrix.mean(1)[:, None]
        vmatrix_m = vmatrix - vmatrix.mean(1)[:, None]

        # Sum of squares across rows
        ematrix_ss = (ematrix_m**2).sum(1)
        vmatrix_ss = (vmatrix_m**2).sum(1)
        cor = np.dot(ematrix_m, vmatrix_m.T)
        N = np.sqrt(np.dot(ematrix_ss[:, None], vmatrix_ss[None]))
        cor=np.divide(cor, N, where=N!=0)
        return cor.T


def data_reshape(cellDancer_df): # pengzhi version
    '''
    load detail file
    return expression matrix and velocity (ngenes, ncells)
    '''
    psc = 1
    gene_names = cellDancer_df['gene_name'].drop_duplicates().to_list()
    # PZ uncommented this.
    cell_number = cellDancer_df[cellDancer_df['gene_name']==gene_names[0]].shape[0]
    cellDancer_df['index'] = np.tile(range(cell_number),len(gene_names))

    splice_reshape = cellDancer_df.pivot(
        index='gene_name', values='splice', columns='index')
    splice_predict_reshape = cellDancer_df.pivot(
        index='gene_name', values='splice_predict', columns='index')
    dMatrix = splice_predict_reshape-splice_reshape
    np_splice_reshape = np.array(splice_reshape)
    np_dMatrix = np.array(dMatrix)
    np_dMatrix2 = np.sqrt(np.abs(np_dMatrix) + psc) * \
        np.sign(np_dMatrix)
    return(np_splice_reshape, np_dMatrix2)

