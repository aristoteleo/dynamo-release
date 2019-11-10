# """ plotting utilities that are built based on scSLAM-seq paper
from .scatters import scatters

def pca(adata, **kwargs):
    scatters(adata, genes, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='pca', mode='expression', label_on_embedding=True, cmap=None, gs=None, **kwargs)

def UMAP(adata, **kwargs):
    scatters(adata, genes, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='umap', mode='expression', label_on_embedding=True, cmap=None, gs=None, **kwargs)

def trimap(adata, **kwargs):
    scatters(adata, genes, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='trimap', mode='expression', label_on_embedding=True, cmap=None, gs=None, **kwargs)

def tSNE(adata, **kwargs):
    scatters(adata, genes, color, dims=[0, 1], current_layer='spliced', use_raw=False, Vkey='S', Ekey='spliced',
             basis='tSNE', mode='expression', label_on_embedding=True, cmap=None, gs=None, **kwargs)


