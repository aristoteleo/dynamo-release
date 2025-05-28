import pandas as pd
import numpy as np
from anndata import AnnData

from typing import Dict

from ..tl import dynamics,reduceDimension,cell_velocities
from .MultiPreprocessor import knn_smooth_chrom


def multi_velocities(
        mdata                     ,
        model:              str='stochastic',
        method:             str='pearson',
        other_kernels_dict: Dict={'transform': 'sqrt'},
        core:               int=3,
        device:             str='cpu',
        extra_color_key:    str=None,
        max_iter:           int=5,
        velo_arg:           Dict ={},
        vkey:               str='velo_s',
        **kwargs
):
    from mudata import MuData
    """
    Calculate the velocites using the scRNA-seq and scATAC-seq data.

    Args:
        mdata:             MuData object containing the RNA and ATAC data.
        model:             The model used to calculate the dynamics. Default is 'stochastic'.
        method:            The method used to calculate the velocity. Default is 'pearson'.
        other_kernels_dict: The dictionary containing the parameters for the other kernels. Default is {'transform': 'sqrt'}.
        core:              The number of cores used for the calculation. Default is 3.
        device:            The device used for the calculation. Default is 'cpu'.
        extra_color_key:   The extra color key used for the calculation. Default is None.
        max_iter:          The maximum number of iterations used for the calculation. Default is 5.
        velo_arg:          The dictionary containing the parameters for the velocity calculation. Default is {}.
        vkey:              The key used for the velocity calculation. Default is 'velo_s'.
        **kwargs:          The other parameters used for the calculation.

    Returns:
        An updated AnnData object with the velocities calculated.
    
    """
    from .dynamical_chrom_func import recover_dynamics_chrom
    # We need to calculate the dynamics of the RNA data first and reduce the dimensionality
    dynamics(mdata['rna'], model=model, cores=core)
    reduceDimension(mdata['rna'])
    cell_velocities(mdata['rna'], method=method, 
                       other_kernels_dict=other_kernels_dict,
                       **velo_arg
                       )
    
    # And we use the connectivity matrix from the RNA data to smooth the ATAC data and calculate the Mc
    knn_smooth_chrom(mdata['aggr'], conn= mdata['rna'].obsp['connectivities'])

    # We then select the genes that are present in both datasets
    shared_cells = pd.Index(np.intersect1d(mdata['rna'].obs_names, mdata['aggr'].obs_names))
    shared_genes = pd.Index(np.intersect1d(
        [i.split('rna:')[-1] for i in mdata['rna'][:,mdata['rna'].var['use_for_dynamics']].var_names], 
        [i.split('aggr:')[-1] for i in mdata['aggr'].var_names]
                                        ))
    
    # We then create the AnnData objects for the RNA and ATAC data
    adata_rna = mdata['rna'][shared_cells, [f'rna:{i}' for i in shared_genes]].copy()
    adata_atac = mdata['aggr'][shared_cells, [f'aggr:{i}' for i in shared_genes]].copy()
    adata_rna.var.index=[i.split('rna:')[-1] for i in adata_rna.var.index]
    adata_atac.var.index=[i.split('aggr:')[-1] for i in adata_atac.var.index]

    adata_rna.layers['Ms']=adata_rna.layers['M_s']
    adata_rna.layers['Mu']=adata_rna.layers['M_u']

    # Now we use MultiVelo's recover_dynamics_chrom function to calculate the dynamics of the RNA and ATAC data
    adata_result = recover_dynamics_chrom(adata_rna,
                                        adata_atac,
                                        max_iter=max_iter,
                                        init_mode="invert",
                                        parallel=True,
                                        n_jobs = core,
                                        save_plot=False,
                                        rna_only=False,
                                        fit=True,
                                        n_anchors=500,
                                        extra_color_key=extra_color_key,
                                         device=device,
                                         **kwargs
                                        )
    
    # We need to add some information of new RNA velocity to the ATAC data
    if vkey not in adata_result.layers.keys():
        raise ValueError('Velocity matrix is not found. Please run multivelo'
                         '.recover_dynamics_chrom function first.')
    if vkey+'_norm' not in adata_result.layers.keys():
        adata_result.layers[vkey+'_norm'] = adata_result.layers[vkey] / np.sum(
            np.abs(adata_result.layers[vkey]), 0)
        adata_result.layers[vkey+'_norm'] /= np.mean(adata_result.layers[vkey+'_norm'])
        adata_result.uns[vkey+'_norm_params'] = adata_result.uns[vkey+'_params']
    if vkey+'_norm_genes' not in adata_result.var.columns:
        adata_result.var[vkey+'_norm_genes'] = adata_result.var[vkey+'_genes']

    # Transition genes identification and velocity calculation
    transition_genes=adata_result.var.loc[adata_result.var['velo_s_norm_genes']==True].index.tolist()
    if 'pearson_transition_matrix' in adata_result.obsp.keys():
        del adata_result.obsp['pearson_transition_matrix']
    if 'velocity_umap' in adata_result.obsm.keys():
        del adata_result.obsm['velocity_umap']
    cell_velocities(adata_result, vkey='velo_s',#layer='Ms',
                       X=adata_result[:,transition_genes].layers['Ms'],
                       V=adata_result[:,transition_genes].layers['velo_s'],
                       transition_genes=adata_result.var.loc[adata_result.var['velo_s_norm_genes']==True].index.tolist(),
                       method=method, 
                       other_kernels_dict=other_kernels_dict,
                       **velo_arg
                      )
    return adata_result


def get_transition_genes(
        adata:              AnnData,
):
    if 'pearson_transition_matrix' in adata.obsp.keys():
        del adata.obsp['pearson_transition_matrix']
    if 'velocity_umap' in adata.obsm.keys():
        del adata.obsm['velocity_umap']
    transition_genes=adata.var.loc[adata.var['velo_s_norm_genes']==True].index.tolist()
    adata.uns['transition_genes']=transition_genes
    return transition_genes
