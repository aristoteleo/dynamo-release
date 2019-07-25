import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.stats import norm
import umap

from .psl import *

def reduceDimension(adata, n_components = 50, normalize_components = True, n_neighbors = 50, reduction_method=None): # c("UMAP", 'tSNE', "DDRTree", "ICA", 'none')
    """Compute a low dimension reduction projection of an annodata object first with PCA, followed by non-linear dimension reduction methods

    :param adata:
    :param n_components:
    :param normalize_components:
    :param n_neighbors:
    :return:
    """
    n_obs = adata.shape[0]

    X = adata.layers['spliced']

    if(not 'X_pca' in adata.obsm.keys()):
        transformer = TruncatedSVD(n_components=n_components, random_state=0)
        X_pca = transformer.fit(X.T).components_.T
        adata.obsm['X_pca'] = X_pca
    else:
        X_pca = adata.obsm['X_pca']

    if reduction_method is 'tSNE':
        bh_tsne = TSNE()
        X_dim = bh_tsne.fit_transform(X_pca)
        adata.obsm['X_tSNE'] = X_dim
    elif reduction_method is 'UMAP':
        X_umap = umap.UMAP().fit(X_pca)
        X_dim = X_umap.embedding_
        adata.obsm['X_umap'] = X_dim
        adj_mat = X_umap.graph_
        adata.obsm['adj_mat'] = adj_mat
    elif reduction_method is 'PSL':
        adj_mat, X_dim = psl_py(X_pca) # this need to be updated
        adata.obsm['X_psl'] = X_dim
        adata.obsm['adj_mat'] = adj_mat

    # use both existing data and predicted future states in dimension reduction to get the velocity plot in 2D
    # use only the existing data for dimension reduction and then project new data in this reduced dimension
    if reduction_method is not 'UMAP':
        if n_neighbors is None: n_neighbors = int(n_obs / 50)
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1) # I could also use UMAP's nearest neighbor graph
        nn.fit(X)
        dists, neighs = nn.kneighbors(X + adata.layers["velocity_u"])
    elif reduction_method is 'UMAP':
        dists, neighs = X_umap.graph_, X_umap.graph_

    scale = np.median(dists, axis=1)
    weight = norm.pdf(x = dists, scale=scale[:, None])
    p_mass = weight.sum(1)
    weight = weight / p_mass[:, None]

    # calculate the embedding for the predicted future states of cells using a Gaussian kernel
    Y_dim = (X_dim[neighs] * (weight[:, :, None])).sum(1)

    adata.obsm['Y_dim'] = Y_dim

    return adata

#
import seaborn as sns
import pandas as pd
import scanpy as sc
import scvelo as scv
from anndata import AnnData

new_RNA = pd.read_csv('/Volumes/xqiu/proj/Aristotle/scSLAM_seq_data/NASC_seq/GSE128273_exp4_newcounts.csv', index_col=0, delimiter=',')
old_RNA = pd.read_csv('/Volumes/xqiu/proj/Aristotle/scSLAM_seq_data/NASC_seq/GSE128273_exp4_oldcounts.csv', index_col=0, delimiter=',')
tot_RNA = pd.read_csv('/Volumes/xqiu/proj/Aristotle/scSLAM_seq_data/NASC_seq/GSE128273_exp4_readcounts.csv', index_col=0, delimiter=',')

split_array = [new_RNA.columns.str.split('_', n=6)[i] for i in range(len(new_RNA.columns.str.split('_', n=6)))]

new_RNA.fillna(0, inplace=True)
old_RNA.fillna(0, inplace=True)
tot_RNA.fillna(0, inplace=True)

# P1_A1_exp2_Jurkat_unlabelled_Unstimulated
split_df = pd.DataFrame(split_array, columns=['Plate', 'Well', 'Exp', 'CellType', '4sU', 'drug_treatment'])
from anndata import AnnData
adata_NASC_seq_4 = AnnData(tot_RNA.values.T,
    obs = split_df,
    layers=dict(
        unspliced=new_RNA.values.T,
        spliced = tot_RNA.values.T))

scv.pp.filter_and_normalize(adata_NASC_seq_4, n_top_genes=500) #, min_counts=15, min_counts_u=10, n_top_genes=2500 n_top_genes can be tuned
scv.pp.moments(adata_NASC_seq_4, n_pcs=10, n_neighbors = 15, mode = 'distances') #
scv.tl.velocity(adata_NASC_seq_4)
scv.tl.velocity_graph(adata_NASC_seq_4)

scv.tl.umap(adata_NASC_seq_4)
sc.pl.umap(adata_NASC_seq_4, color=['4sU'])

scv.tl.velocity_graph(adata_NASC_seq_4)
scv.pl.velocity_embedding(adata_NASC_seq_4, color=['4sU'], basis='umap', figsize=[6, 6])

scv.pl.velocity_embedding_grid(adata_NASC_seq_4, color=['4sU'], basis='umap', figsize=[6, 6])

adata_NASC_seq_4_res = scv.pl.velocity_embedding_stream(adata_NASC_seq_4, color=['4sU'], basis='umap', legend_loc=None, figsize=[12, 12], show=False, linewidth=2)

adata_NASC_seq_4 = reduceDimension(adata_NASC_seq_4)
