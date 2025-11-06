from anndata import AnnData
from typing import Literal, List
import os


def extvelo(
    adata: AnnData,
    method: Literal["latentvelo", "celldancer"] = "celldancer",
    celltype_key: str = "clusters",
    batch_key: str = None,
    basis: str = "X_umap",
    Ms_key: str = "M_s",
    Mu_key: str = "M_u",
    gene_list: List[str] = None,
    latentvelo_VAE_kwargs: dict = {},
    param_name_key: str = 'tmp/latentvelo_params',
    **kwargs,
) -> AnnData:
    if method == "celldancer":
        #tested successfully
        from ..external.celldancer.utilities import adata_to_df_with_embed 
        from ..external.celldancer import velocity
        from ..external.celldancer.utilities import export_velocity_to_dynamo
        if gene_list is None:
            gene_list = adata.var.query("use_for_pca==True").index.tolist()
        if not os.path.exists('temp'):
            os.makedirs('temp')
        cell_type_u_s=adata_to_df_with_embed(adata,
                      us_para=[Mu_key,Ms_key],
                      cell_type_para=celltype_key,
                      embed_para=basis,
                      save_path='temp/test_cell_type_u_s.csv',
                      gene_list=gene_list)
        
        loss_df, cellDancer_df = velocity(cell_type_u_s, gene_list=gene_list, **kwargs)
        adata = export_velocity_to_dynamo(cellDancer_df,adata)
        adata.uns['dynamics']={
            'filter_gene_mode': 'final','t': None,'group': None,
            'X_data': None,'X_fit_data': None,'asspt_mRNA': 'ss',
            'experiment_type': 'conventional','normalized': True,
            'model': 'stochastic','est_method': 'gmm','has_splicing': True,
            'has_labeling': False,'splicing_labeling': False,
            'has_protein': False,'use_smoothed': True,'NTR_vel': False,
            'log_unnormalized': True,'fraction_for_deg': False
        }
        return cellDancer_df,adata
    elif method == "latentvelo":
        #tested successfully
        latent_data, adata = _latentvelo_cal(
                adata=adata,
                velocity_key='velocity_S',
                celltype_key=celltype_key,
                batch_key=batch_key,
                latentvelo_VAE_kwargs=latentvelo_VAE_kwargs,
                param_name_key=param_name_key,
                **kwargs
            )
        
        return latent_data, adata
    elif method == "deepvelo":
        #tested successfully
        from ..external.deepvelo import train
        from ..external.deepvelo.train import Constants
        adata.layers['Ms']=adata.layers[Ms_key]
        adata.layers['Mu']=adata.layers[Mu_key]
        trainer = train(adata, Constants.default_configs)
        adata.layers['velocity_S']=adata.layers['velocity']
        adata.var[f'use_for_dynamics'] = adata.var['use_for_pca']
        adata.var[f'use_for_transition'] = adata.var['use_for_pca']
        adata.uns['dynamics']={
            'filter_gene_mode': 'final','t': None,'group': None,
            'X_data': None,'X_fit_data': None,'asspt_mRNA': 'ss',
            'experiment_type': 'conventional','normalized': True,
            'model': 'stochastic','est_method': 'gmm','has_splicing': True,
            'has_labeling': False,'splicing_labeling': False,
            'has_protein': False,'use_smoothed': True,'NTR_vel': False,
            'log_unnormalized': True,'fraction_for_deg': False
        }
        return adata
    elif method == "velovi":
        #Need to be tested
        from velovi import VELOVI
        import torch
        import numpy as np
        import scipy as sp
        VELOVI.setup_anndata(adata, spliced_layer=Ms_key, unspliced_layer=Mu_key)
        vae = VELOVI(adata)
        vae.train(**kwargs)
        latent_time = vae.get_latent_time(n_samples=25)
        velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

        t = latent_time
        scaling = 20 / t.max(0)

        adata.layers["velocity_S"] = velocities / scaling
        adata.layers["latent_time_velovi"] = latent_time

        adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
        adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
        adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
        adata.var["fit_t_"] = (
            torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
        ) * scaling
        adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
        adata.var["fit_scaling"] = 1.0
        adata.var[f'use_for_dynamics'] = adata.var['use_for_pca']
        adata.var[f'use_for_transition'] = adata.var['use_for_pca']
        adata.uns['dynamics']={
            'filter_gene_mode': 'final','t': None,'group': None,
            'X_data': None,'X_fit_data': None,'asspt_mRNA': 'ss',
            'experiment_type': 'conventional','normalized': True,
            'model': 'stochastic','est_method': 'gmm','has_splicing': True,
            'has_labeling': False,'splicing_labeling': False,
            'has_protein': False,'use_smoothed': True,'NTR_vel': False,
            'log_unnormalized': True,'fraction_for_deg': False
        }
        return adata


    else:
        raise ValueError(f"Method {method} not supported")



def _latentvelo_cal(
    adata: AnnData,
    param_name_key='tmp/latentvelo_params',
    velocity_key='velocity_S',
    celltype_key=None,
    batch_key=None,
    latentvelo_VAE_kwargs={},
    use_rep=None,
    **kwargs):
    try:
        import torchdiffeq
    except:
        raise ValueError("torchdiffeq not installed")
    import os
    os.makedirs(param_name_key, exist_ok=True)
    # latentvelo
    from ..external.latentvelo.models.vae_model import VAE
    from ..external.latentvelo.models.annot_vae_model import AnnotVAE
    from ..external.latentvelo.train import train
    from ..external.latentvelo.utils import standard_clean_recipe, anvi_clean_recipe
    # Optional device override for latentvelo stack
    device_override = kwargs.pop('device', None)
    if device_override is not None:
        from ..external.latentvelo import trainer as lv_trainer
        from ..external.latentvelo import trainer_anvi as lv_trainer_anvi
        from ..external.latentvelo import trainer_atac as lv_trainer_atac
        from ..external.latentvelo import output_results as lv_out_mod
        from ..external.latentvelo import utils as lv_utils
        for m in (lv_trainer, lv_trainer_anvi, lv_trainer_atac, lv_out_mod, lv_utils):
            if hasattr(m, 'set_device'):
                m.set_device(device_override)

    # Shared preprocessing
    if celltype_key == None:
        adata = standard_clean_recipe(adata, batch_key=batch_key, 
                    celltype_key=celltype_key, r2_adjust=True,use_rep=use_rep)

        model = VAE(**latentvelo_VAE_kwargs)
        epochs, vae, val_traj = train(model,adata,name=param_name_key,**kwargs)
    else:
        adata=anvi_clean_recipe(adata, celltype_key=celltype_key,
                    batch_key=batch_key,r2_adjust=True,use_rep=use_rep)
        # Get required parameters from adata
        observed = adata.n_vars
        celltypes = len(adata.obs[celltype_key].unique())
        model = AnnotVAE(observed=observed, celltypes=celltypes, **latentvelo_VAE_kwargs)
        epochs, vae, val_traj = train(model,adata,name=param_name_key,**kwargs)
    adata.uns['latentvelo_train_params'] = {
                'epochs': epochs,
                'vae': vae,
                'val_traj': val_traj
            }
    from ..external.latentvelo.output_results import output_results as lv_output
    latent_data, adta = lv_output(model,adata,gene_velocity=True,)
    adata.var[f'{velocity_key}_genes'] = adata.var['velocity_genes']
    #covert to csr
    import scipy as sp
    from scipy.sparse import issparse
    if not issparse(adta.layers['velo_s']):
        adata.layers['velocity_S'] = sp.sparse.csr_matrix(adta.layers['velo_s'])
    else:
        adata.layers['velocity_S'] = adta.layers['velo_s']
    if not issparse(adta.layers['velo_u']):
        adata.layers['velocity_U'] = sp.sparse.csr_matrix(adta.layers['velo_u'])
    else:
        adata.layers['velocity_U'] = adta.layers['velo_u']
    adata.obsm['X_latentvelo'] = latent_data.X
    adata.obsm['X_latentvelo_velo_s'] = latent_data.layers['spliced_velocity']
    adata.obsm['X_latentvelo_velo_u'] = latent_data.layers['unspliced_velocity']

    adata.var[f'use_for_dynamics'] = adata.var['velocity_genes']
    adata.var[f'use_for_transition'] = adata.var['velocity_genes']

    adata.uns['dynamics']={'filter_gene_mode': 'final',
                    't': None,
                    'group': None,
                    'X_data': None,
                    'X_fit_data': None,
                    'asspt_mRNA': 'ss',
                    'experiment_type': 'conventional',
                    'normalized': True,
                    'model': 'stochastic',
                    'est_method': 'gmm',
                    'has_splicing': True,
                    'has_labeling': False,
                    'splicing_labeling': False,
                    'has_protein': False,
                    'use_smoothed': True,
                    'NTR_vel': False,
                    'log_unnormalized': True,
                    'fraction_for_deg': False}
    return latent_data, adata

