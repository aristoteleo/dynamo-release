import anndata as ad
from anndata import AnnData
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

# Imports from MultiDynamo
from .MultiConfiguration import MDKM
from .old_MultiVelocity import MultiVelocity

from ..pl import cell_wise_vectors, streamline_plot, topography
from ..pd import fate, perturbation
from ..mv import animate_fates
from ..pp import pca
from ..tl import reduceDimension, cell_velocities
from ..vf import VectorField


# Helper functions
def compute_animations(adata,
                       cell_type_key:   str,
                       cores:           int = 6,
                       delta_epsilon:   float = 0.25,
                       epsilon:         float = 1.0,
                       max_tries:       int = 10,
                       n_cells:         int = 100,
                       n_earliest:      int = 30,
                       prefix:          str = None,
                       skip_cell_types: List = []
                       ) -> None:
    # Extract cell metadata
    cell_metadata = adata.obs.copy()

    # Add UMAP
    cell_metadata['umap_1'] = adata.obsm['X_umap'][:, 0]
    cell_metadata['umap_2'] = adata.obsm['X_umap'][:, 1]

    # Group by cell_type_key and find the rows with the maximal 'rotated_umap_1'
    grouped = cell_metadata.groupby(cell_type_key)

    # Find the mean locations of cell types
    top_indices_1, top_indices_2 = {}, {}
    for cell_type, celltype_data in grouped:
        subset_df = celltype_data.nsmallest(n_cells, 'umap_1')
        top_indices_1[cell_type] = subset_df['umap_1'].mean()
        subset_df = celltype_data.nlargest(n_cells, 'umap_2')
        top_indices_2[cell_type] = subset_df['umap_2'].mean()

    cell_types = cell_metadata[cell_type_key].cat.categories.tolist()
    progenitor_list = []

    for cell_type in cell_types:
        if (skip_cell_types is not None) and (cell_type in skip_cell_types):
            continue

        print(f'Computing animation for cell type {cell_type}')

        # Find the progenitors
        n_tries, progenitors = 1, []
        while len(progenitors) < n_cells and n_tries < max_tries + 1:
            progenitors = adata.obs_names[adata.obs.celltype.isin([cell_type]) &
                                          (abs(cell_metadata['umap_1'] - top_indices_1[cell_type]) < (
                                                  epsilon + n_tries * delta_epsilon)) &
                                          (abs(cell_metadata['umap_2'] - top_indices_2[cell_type]) < (
                                                  epsilon + n_tries * delta_epsilon))]
            n_tries += 1

        if len(progenitors) >= n_earliest:
            # Progenitors for all subset simulation
            print(f'Adding {n_earliest} cells of type {cell_type}.')
            progenitor_list.extend(progenitors[0:min(len(progenitors), n_earliest)])

        # Progenitors for this animation
        # progenitors = progenitors[0:min(len(progenitors), n_cells)]

        # Determine their fate
        # dyn.pd.fate(adata, basis='umap_perturbation', init_cells=progenitors, interpolation_num=100,
        #             direction='forward', inverse_transform=False, average=False, cores=6)

        # Compute the animation
        # animation_fn = cell_type + '_perturbed_fate_ani.mp4'
        # animation_fn = animation_fn.replace('/', '-')
        # dyn.mv.animate_fates(adata, basis='umap_perturbation', color='celltype', n_steps=100,
        #                      interval=100, save_show_or_return='save',
        #                      save_kwargs={'filename': animation_fn,
        #                                   'writer': 'ffmpeg'})

    # Determine fate of progenitor_list
    fate(adata, basis='umap_perturbation', init_cells=progenitor_list, interpolation_num=100,
                direction='forward', inverse_transform=False, average=False, cores=cores)

    # Compute the animation
    file_name = prefix + '_perturbation.mpeg'
    file_name = file_name.replace(':', '-')
    file_name = file_name.replace('/', '-')
    animate_fates(adata, basis='umap_perturbation', color='celltype', n_steps=100,
                         interval=100, save_show_or_return='save',
                         save_kwargs={'filename': file_name,
                                      'writer': 'ffmpeg'})

def genes_and_elements_for_dynamics(atac_adata:    AnnData,
                                    rna_adata:     AnnData,
                                    cre_dict:      Dict[str, List[str]],
                                    promoter_dict: Dict[str, List[str]],
                                    min_r2:        float = 0.01) -> List[bool]:
    # Get fit parameters
    vel_params_array = rna_adata.varm['vel_params']

    # Extract 'gamma_r2'
    gamma_r2_index = np.where(np.array(rna_adata.uns['vel_params_names']) == 'gamma_r2')[0][0]
    r2 = vel_params_array[:, gamma_r2_index]

    # Set genes for dynamics
    genes_for_dynamics = rna_adata.var_names[r2 > min_r2].to_list()
    use_for_dynamics = [gene in genes_for_dynamics for gene in rna_adata.var_names.to_list()]

    # Compute elements for dynamics
    cre_for_dynamics = []
    for gene, cre_list in cre_dict.items():
        if gene in genes_for_dynamics:
            cre_for_dynamics += cre_list

    for gene, promoter_list in promoter_dict.items():
        if gene in genes_for_dynamics:
            cre_for_dynamics += promoter_list

    use_for_dynamics += [element in cre_for_dynamics for element in atac_adata.var_names]

    return use_for_dynamics


class MultiomicVectorField:
    def __init__(self,
                 multi_velocity:        Union[MultiVelocity],
                 min_gamma:             float = None,
                 min_r2:                float = 0.01,
                 rescale_velo_c:        float = 1.0):
        # This is basically an adapter from multiomic data to format where we can borrow tools previously developed
        # in dynamo.
        if isinstance(multi_velocity, MultiVelocity):
            multi_velocity = MultiVelocity.from_mdata(multi_velocity)

        # ... mdata
        mdata = multi_velocity.get_mdata()
        atac_adata, rna_adata = mdata.mod['atac'], mdata.mod['rna']

        # ... CRE dictionary
        cre_dict = multi_velocity.get_cre_dict()

        # ... promoter dictionary
        promoter_dict = multi_velocity.get_promoter_dict()

        # To estimate the multi-omic velocity field, we assemble a single AnnData object from the following components
        # NOTE: In our descriptions below *+* signifies the directo sum of two vector spaces
        # ... .layers
        # ... ... counts: counts => rna counts *+* atac counts
        rna_counts = rna_adata.layers[MDKM.RNA_COUNTS_LAYER].toarray().copy()
        atac_counts = atac_adata.layers[MDKM.ATAC_COUNTS_LAYER].toarray().copy()
        counts = np.concatenate((rna_counts, atac_counts), axis=1)

        # ... ... raw: spliced, unspliced ==> spliced *+* chromatin, unspliced *+* 0
        chromatin_state = atac_adata.layers[MDKM.ATAC_COUNTS_LAYER].toarray().copy()
        spliced = rna_adata.layers[MDKM.RNA_SPLICED_LAYER].toarray().copy()
        unspliced = rna_adata.layers[MDKM.RNA_UNSPLICED_LAYER].toarray().copy()

        spliced = np.concatenate((spliced, chromatin_state), axis=1)
        unspliced = np.concatenate((unspliced, np.zeros(chromatin_state.shape)), axis=1)
        del chromatin_state

        # ... ... first moments: M_s, M_u => M_s *+* Mc, M_u *+* 0
        Mc = atac_adata.layers[MDKM.RNA_FIRST_MOMENT_CHROM_LAYER].toarray().copy()
        Ms = rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER].copy()
        Mu = rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER].copy()

        Ms = np.concatenate((Ms, Mc), axis=1)
        Mu = np.concatenate((Mu, np.zeros(Mc.shape)), axis=1)
        del Mc

        # ... ... velocity_S ==> velocity_S + lifted_velo_c
        velocity_C = atac_adata.layers[MDKM.ATAC_CHROMATIN_VELOCITY_LAYER].copy()
        velocity_S = rna_adata.layers[MDKM.RNA_SPLICED_VELOCITY_LAYER].toarray().copy()

        velocity_S = np.concatenate((velocity_S, rescale_velo_c * velocity_C), axis=1)
        del velocity_C

        # ... .obs
        # ... ... carry over entire obs for now
        obs_df = rna_adata.obs.copy()

        # ... .obsp
        # ... ... connectivities ==> connectivities
        connectivities = rna_adata.obsp['connectivities'].copy()

        # ... ... distances ==> distances
        distances = rna_adata.obsp['distances'].copy()

        # ... .uns
        # ... ... dynamics ==> dynamics
        dynamics = rna_adata.uns['dynamics'].copy()

        # ... ... neighbors ==> neighbors
        neighbors = rna_adata.uns['neighbors'].copy()

        # ... ... pp ==> pp
        pp = rna_adata.uns['pp'].copy()

        # ... ... vel_params_names ==> vel_params_names
        vel_params_names = rna_adata.uns['vel_params_names'].copy()

        # ... .var
        # ... ... var_names ==> (rna) var_names + (atac) var_names
        var_names = rna_adata.var_names.tolist() + atac_adata.var_names.tolist()

        # ... ... feature_type ==> n_genes * 'gene', n_elements * 'CRE'
        feature_type = rna_adata.n_vars * ['gene'] + atac_adata.n_vars * ['CRE']

        # ... ... use_for_pca
        use_for_dynamics = genes_and_elements_for_dynamics(atac_adata=atac_adata,
                                                           rna_adata=rna_adata,
                                                           cre_dict=cre_dict,
                                                           promoter_dict=promoter_dict,
                                                           min_r2=min_r2)

        # ... ... use_for_pca
        use_for_pca = genes_and_elements_for_dynamics(atac_adata=atac_adata,
                                                      rna_adata=rna_adata,
                                                      cre_dict=cre_dict,
                                                      promoter_dict=promoter_dict,
                                                      min_r2=min_r2)

        var_df = pd.DataFrame(data={'feature_type':     feature_type,
                                    'use_for_dynamics': use_for_dynamics,
                                    'use_for_pca':      use_for_pca},
                              index=var_names)

        # ... .varm
        # ... ... vel_params => vel_params + (1,1)
        vel_params_array = rna_adata.varm['vel_params']

        chrom_vel_params_array = np.full((atac_adata.n_vars, len(vel_params_names)), np.nan)

        # ... ... create vacuous 'gamma' for chromatin data
        gamma_index = np.where(np.array(vel_params_names) == 'gamma')[0][0]
        chrom_vel_params_array[:, gamma_index] = np.ones(atac_adata.n_vars)

        # ... ... create vacuous 'gamma_r2' for chromatin data
        gamma_r2_index = np.where(np.array(vel_params_names) == 'gamma_r2')[0][0]
        chrom_vel_params_array[:, gamma_r2_index] = np.ones(atac_adata.n_vars)

        # ... ... concatenate the arrays
        vel_params_array = np.concatenate((vel_params_array, chrom_vel_params_array), axis=0)

        # X ==> X + X
        X = np.concatenate((rna_adata.X.toarray().copy(), atac_adata.X.toarray().copy()), axis=1)

        # Instantiate the multiomic AnnData object
        adata_multi = AnnData(obs=obs_df,
                              var=var_df,
                              X=X)
        # ... add .layers
        # ... ... counts
        adata_multi.layers[MDKM.RNA_COUNTS_LAYER] = counts

        # ... ... raw
        adata_multi.layers[MDKM.RNA_SPLICED_LAYER] = spliced
        adata_multi.layers[MDKM.RNA_UNSPLICED_LAYER] = unspliced

        # ... ... first moments
        adata_multi.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER] = Ms
        adata_multi.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER] = Mu

        # ... ... rna velocity
        adata_multi.layers[MDKM.RNA_SPLICED_VELOCITY_LAYER] = velocity_S

        # ... add .obsp
        adata_multi.obsp['connectivities'] = connectivities
        adata_multi.obsp['distances'] = distances

        # ... add .uns
        adata_multi.uns['dynamics'] = dynamics
        adata_multi.uns['neighbors'] = neighbors
        adata_multi.uns['pp'] = pp
        adata_multi.uns['vel_params_names'] = vel_params_names

        # ... add varm
        adata_multi.varm['vel_params'] = vel_params_array

        # Set instance variables

        self.multi_adata = adata_multi.copy()

    def cell_velocities(self,
                        cores:               int = 6,
                        min_r2:              float = 0.5,
                        n_neighbors:         int = 30,
                        n_pcs:               int = 30,
                        random_seed:         int = 42,
                        trans_matrix_method: Literal["kmc", "fp", "cosine", "pearson", "transform"] = "pearson",
                        ) -> AnnData:
        # We'll save ourselves some grief and just compute both the PCA and UMAP representations
        # of the vector field up front
        # ... extract the multiomic AnnData object
        adata_multi = self.multi_adata.copy()

        # ... compute PCA
        adata_multi = pca(adata=adata_multi,
                                 n_pca_components=n_pcs,
                                 random_state=random_seed)

        # ... compute the appropriate dimensional reduction
        reduceDimension(adata_multi,
                               basis='pca',
                               cores=cores,
                               n_pca_components=n_pcs,
                               n_components=2,
                               n_neighbors=n_neighbors,
                               reduction_method='umap')

        # ... project high dimensional velocities onto PCA embeddings and compute cell transitions
        cell_velocities(adata_multi,
                               basis='pca',
                               method=trans_matrix_method,
                               min_r2=min_r2,
                               other_kernels_dict={'transform': 'sqrt'})

        # ... project high dimensional velocities onto PCA embeddings and compute cell transitions
        cell_velocities(adata_multi,
                               basis='umap',
                               method=trans_matrix_method,
                               min_r2=min_r2,
                               other_kernels_dict={'transform': 'sqrt'})

        self.multi_adata = adata_multi.copy()

        return self.multi_adata

    def compute_vector_field(self,
                             cores:       int = 6,
                             restart_num: int = 5
                             ):
        VectorField(self.multi_adata,
                           basis='pca',
                           cores=cores,
                           grid_num=100,
                           M=1000,
                           pot_curl_div=True,
                           restart_num=restart_num,
                           restart_seed=[i * 888888888 for i in range(1, restart_num + 1)])
        '''
        dyn.vf.VectorField(self.multi_adata,
                           basis='umap',
                           cores=cores,
                           grid_num=100,
                           M=1000,
                           pot_curl_div=True,
                           restart_num=restart_num,
                           restart_seed=[i * 888888888 for i in range(1, restart_num + 1)])
        '''

    def plot_cell_wise_vectors(self,
                               color:        str = 'cell_type',
                               figsize:      Tuple[float, float] = (9, 6),
                               **save_kwargs
                               ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        cell_wise_vectors(self.multi_adata,
                                 basis='umap',
                                 color=[color],
                                 pointsize=0.1,
                                 quiver_length=6,
                                 quiver_size=6,
                                 save_kwargs=save_kwargs,
                                 save_show_or_return='show',
                                 show_arrowed_spines=False,
                                 show_legend='on_data',
                                 ax = ax)
        plt.show()

    def plot_streamline_plot(self,
                             color:   str = 'cell_type',
                             figsize: Tuple[float, float] = (9, 6),
                             **save_kwargs
                             ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        streamline_plot(self.multi_adata,
                               basis='umap',
                               color=[color],
                               show_arrowed_spines=True,
                               show_legend='on_data',
                               ax = ax)
        plt.show()

    def plot_topography(self,
                        color:        str = 'cell_type',
                        figsize:      Tuple[float, float] = (9, 6),
                        **save_kwargs
                        ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        topography(self.multi_adata,
                          basis='pca',
                          background='white',
                          color=color,
                          frontier=True,
                          n = 200,
                          show_legend='on data',
                          streamline_color='black',
                          ax = ax)

    def predict_perturbation(self,
                             gene:              str,
                             expression:        float,
                             cell_type_key:     str = 'cell_type',
                             compute_animation: bool = False,
                             emb_basis:         str = 'umap',
                             skip_cell_types:   List = None
                             ) -> AnnData:

        perturbed_multi_adata = perturbation(self.multi_adata,
                                                    genes=gene,
                                                    expression=expression,
                                                    emb_basis='umap')
        streamline_plot(self.multi_adata, color=["cell_type", gene],
                               basis="umap_perturbation")

        if compute_animation:
            # Fit analytic vector field
            VectorField(self.multi_adata,
                               basis='umap_perturbation')

            compute_animations(adata=self.multi_adata,
                               cell_type_key=cell_type_key,
                               prefix=gene,
                               skip_cell_types=skip_cell_types)

        return perturbed_multi_adata
