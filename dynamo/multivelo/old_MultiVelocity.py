from anndata import AnnData
import matplotlib.pyplot as plt
from multiprocessing import Pool

import numpy as np
import os
from os import PathLike
import pandas as pd

from scipy.sparse import coo_matrix, csr_matrix, hstack, issparse
from scipy.sparse.linalg import svds

from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union
)

import warnings

# Import from dynamo
from ..dynamo_logger import (
    LoggerManager,
    main_exception,
    main_info,
)

# Imports from MultiDynamo
from .ChromatinVelocity import ChromatinVelocity
from .MultiConfiguration import MDKM
from .pyWNN import pyWNN


# Static function
# direction_cosine
def direction_cosine(args):
    i, j, expression_mtx, velocity_mtx = args

    if i == j:
        return i, j, -1

    delta_ij = None
    if isinstance(expression_mtx, csr_matrix):
        delta_ij = (expression_mtx.getrow(j) - expression_mtx.getrow(i)).toarray().flatten()
    elif isinstance(expression_mtx, np.ndarray):
        delta_ij = (expression_mtx[j, :] - expression_mtx[i, :]).flatten()
    else:
        main_exception(f'Expression matrix is instance of class {type(expression_mtx)}')

    vel_i = velocity_mtx.getrow(i).toarray().flatten()

    dot_product = np.dot(delta_ij, vel_i)  # vel_i.dot(delta_ij)
    magnitude_vel_i = np.linalg.norm(vel_i)
    magnitude_delta_ij = np.linalg.norm(delta_ij)

    if magnitude_vel_i != 0 and magnitude_delta_ij != 0:
        cosine_similarity = dot_product / (magnitude_vel_i * magnitude_delta_ij)
    else:
        # One of velocity or delta_ij is zero, so can't compute a cosine, we'll just set to
        # lowest possible value (-1)
        cosine_similarity = -1

    return i, j, cosine_similarity


# get_connectivities - patterned after function in scVelo
def get_connectivities(adata:             AnnData,
                       mode:              str = 'connectivities',
                       n_neighbors:       int = None,
                       recurse_neighbors: bool = False
                       ) -> Union[csr_matrix, None]:
    if 'neighbors' in adata.uns.keys():
        C = get_neighbors(adata=adata, mode=mode)
        if n_neighbors is not None and n_neighbors < get_n_neighbors(adata=adata):
            if mode == 'connectivities':
                C = select_connectivities(C, n_neighbors)
            else:
                C = select_distances(C, n_neighbors)
        connectivities = C > 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            connectivities.setdiag(1)
            if recurse_neighbors:
                connectivities += connectivities.dot(connectivities * 0.5)
                connectivities.data = np.clip(connectivities.data, 0, 1)
            connectivities = connectivities.multiply(1.0 / connectivities.sum(1))
        return connectivities.tocsr().astype(np.float32)
    else:
        return None


# get_n_neighbors - lifted from scVelo
def get_n_neighbors(adata: AnnData) -> int:
    return adata.uns.get('neighbors', {}).get('params', {}).get('n_neighbors', 0)


def get_neighbors(adata: AnnData,
                  mode:  str = 'distances'):
    if hasattr(adata, 'obsp') and mode in adata.obsp:
        return adata.obsp[mode]
    elif 'neighbors' in adata.uns.keys() and mode in adata.uns['neighbors']:
        return adata.uns['neighbors'][mode]
    else:
        main_exception(f'The selected mode {mode} is not valid.')


def lifted_chromatin_velocity(arg):
    i, j, chromatin_state, cosines, expression_mtx, rna_velocity = arg

    if i == j:
        main_exception('A cell should never be its own integral neighbor.')

    # Compute change in chromatin state
    delta_c_ij = None
    if isinstance(chromatin_state, csr_matrix):
        delta_c_ij = (chromatin_state.getrow(j) - chromatin_state.getrow(i)).toarray().flatten()
    elif isinstance(chromatin_state, np.ndarray):
        delta_c_ij = (chromatin_state[j, :] - chromatin_state[i, :]).flatten()
    else:
        main_exception(f'Chromatin state matrix is instance of class {type(chromatin_state)}')

    # Retrieve cosine
    cosine = cosines[i, j]

    # Compute change in RNA expression
    delta_s_ij = None
    if isinstance(expression_mtx, csr_matrix):
        delta_s_ij = (expression_mtx.getrow(j) - expression_mtx.getrow(i)).toarray().flatten()
    elif isinstance(expression_mtx, np.ndarray):
        delta_s_ij = (expression_mtx[j, :] - expression_mtx[i, :]).flatten()
    else:
        main_exception(f'RNA expression matrix is instance of class {type(expression_mtx)}')

    # Compute norms
    norm_delta_s_ij = np.linalg.norm(delta_s_ij)
    norm_rna_velocity = np.linalg.norm(rna_velocity.toarray())

    if norm_delta_s_ij != 0:
        chromatin_velocity = (norm_rna_velocity * cosine / norm_delta_s_ij) * delta_c_ij
    else:
        chromatin_velocity = np.zeros(chromatin_state.shape[1])

    return i, chromatin_velocity


def regression(c,
               u,
               s,
               ss,
               us,
               uu,
               fit_args,
               mode,
               gene):
    c_90 = np.percentile(c, 90)
    u_90 = np.percentile(u, 90)
    s_90 = np.percentile(s, 90)

    low_quality = (c_90 == 0 or s_90 == 0 or u_90 == 0)

    if low_quality:
        # main_info(f'Skipping low quality gene {gene}.')
        return np.zeros(len(u)), np.zeros(len(u)), 0, 0, np.inf

    cvc = ChromatinVelocity(c,
                            u,
                            s,
                            ss,
                            us,
                            uu,
                            fit_args,
                            gene=gene)

    if cvc.low_quality:
        return np.zeros(len(u)), np.zeros(len(u)), 0, 0, np.inf

    if mode == 'deterministic':
        cvc.compute_deterministic()
    elif mode == 'stochastic':
        cvc.compute_stochastic()
    velocity = cvc.get_velocity(mode=mode)
    gamma = cvc.get_gamma(mode=mode)
    r2 = cvc.get_r2(mode=mode)
    loss = cvc.get_loss(mode=mode)
    variance_velocity = (None if mode == 'deterministic'
                         else cvc.get_variance_velocity())
    return velocity, variance_velocity, gamma, r2, loss


def select_connectivities(connectivities,
                          n_neighbors=None):
    C = connectivities.copy()
    n_counts = (C > 0).sum(1).A1 if issparse(C) else (C > 0).sum(1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(),
                                                       n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = C.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[::-1][n_neighbors:]
        dat[rm_idx] = 0

    C.eliminate_zeros()
    return C


def select_distances(dist,
                     n_neighbors: int = None):
    D = dist.copy()
    n_counts = (D > 0).sum(1).A1 if issparse(D) else (D > 0).sum(1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0

    D.eliminate_zeros()
    return D


# smooth_scale - lifted from MultiVelo
def smooth_scale(conn,
                 vector):
    max_to = np.max(vector)
    min_to = np.min(vector)
    v = conn.dot(vector.T).T
    max_from = np.max(v)
    min_from = np.min(v)
    res = ((v - min_from) * (max_to - min_to) / (max_from - min_from)) + min_to
    return res


# top_n_sparse - lifted from MultiVelo
def top_n_sparse(conn, n):
    conn_ll = conn.tolil()
    for i in range(conn_ll.shape[0]):
        row_data = np.array(conn_ll.data[i])
        row_idx = np.array(conn_ll.rows[i])
        new_idx = row_data.argsort()[-n:]
        top_val = row_data[new_idx]
        top_idx = row_idx[new_idx]
        conn_ll.data[i] = top_val.tolist()
        conn_ll.rows[i] = top_idx.tolist()
    conn = conn_ll.tocsr()
    idx1 = conn > 0
    idx2 = conn > 0.25
    idx3 = conn > 0.5
    conn[idx1] = 0.25
    conn[idx2] = 0.5
    conn[idx3] = 1
    conn.eliminate_zeros()
    return conn


class MultiVelocity:
    def __init__(self,
                 mdata                     ,
                 cosine_similarities: csr_matrix = None,
                 cre_dict:            Dict = None,
                 include_gene_body:   bool = False,
                 integral_neighbors:  Dict = None,
                 linkage_fn:          str = 'feature_linkage.bedpe', # in 'outs/analysis/feature_linkage' directory
                 linkage_method:      Literal['cellranger', 'cicero', 'scenic+'] = 'cellranger',
                 max_peak_dist:       int = 10000,
                 min_corr:            float = 0.5,
                 neighbor_method:     Literal['multivi', 'wnn'] = 'multivi',
                 nn_dist:             csr_matrix = None,
                 nn_idx:              csr_matrix = None,
                 peak_annot_fn:       str = 'peak_annotation.tsv', # in 'outs' directory
                 promoter_dict:       Dict = None
                 ):
        # Initialize instance variables
        self.mdata = mdata.copy() if mdata is not None else None

        self._cre_dict = cre_dict.copy() if cre_dict is not None else None

        self.cosine_similarities = cosine_similarities.copy() if cosine_similarities is not None else None

        self.include_gene_body = include_gene_body

        self.integral_neighbors = integral_neighbors.copy() if integral_neighbors is not None else None

        self.linkage_fn = linkage_fn

        self.linkage_method = linkage_method

        self.max_peak_dist = max_peak_dist

        self.min_corr = min_corr

        self.neighbor_method = neighbor_method

        self.nn_dist = nn_dist.copy() if nn_dist is not None else None

        self.nn_idx = nn_idx.copy() if nn_idx is not None else None

        self.peak_annot_fn = peak_annot_fn

        self._promoter_dict = promoter_dict.copy() if promoter_dict is not None else None

    def atac_elements(self):
        return self.mdata['atac'].var_names.tolist()

    def compute_linkages(self) -> None:
        if self.linkage_method == 'cellranger':
            self.compute_linkages_via_cellranger()
        elif self.linkage_method == 'cicero':
            self.compute_linkages_via_cicero()
        elif self.linkage_method == 'scenic+':
            self.compute_linkages_via_scenicplus()
        else:
            main_exception(f'Unrecognized method to compute linkages ({self.linkage_method}) requested.')

    def compute_linkages_via_cellranger(self) -> None:
        # This reads the cellranger-arc 'feature_linkage.bedpe' and 'peak_annotation.tsv' files
        # to extract dictionaries attributing cis-regulatory elements with specific genes
        main_info('Computing linkages via cellranger ...')
        linkage_logger = LoggerManager.gen_logger('compute_linkages_via_cellranger')
        linkage_logger.log_time()

        # Confirm that this is matched ATAC- and RNA-seq data
        if not self.mdata.mod['atac'].uns[MDKM.MATCHED_ATAC_RNA_DATA_KEY]:
            main_exception('Cannot use cellranger to compute CRE linkages for UNMATCHED data')

        outs_data_path = os.path.join(self.mdata.mod['atac'].uns['base_data_path'], 'outs')
        # Confirm that the base path to the 'outs' directory exists
        if not os.path.exists(outs_data_path):
            main_exception(f'The path to the 10X outs directory ({outs_data_path}) does not exist.')

        # Read annotations
        peak_annot_path = os.path.join(outs_data_path, self.peak_annot_fn)
        if not os.path.exists(peak_annot_path):
            main_exception(f'The path to the peak annotation file ({peak_annot_path}) does not exist.')

        corr_dict, distal_dict, gene_body_dict, promoter_dict = {}, {}, {}, {}
        with open(peak_annot_path) as f:
            # Scan the header to determine version of CellRanger used in making the peak annotation file
            header = next(f)
            fields = header.split('\t')

            # Peak annotation should contain 4 columns for version 1.X of CellRanger and 6 columns for
            # version 2.X
            if len(fields) not in [4, 6]:
                main_exception('Peak annotation file should contain 4 columns (CellRanger ARC 1.0.0) ' +
                               'or 6 columns (CellRanger ARC 2.0.0)')
            else:
                offset = 0 if len(fields) == 4 else 2

            for line in f:
                fields = line.rstrip().split('\t')

                peak = f'{fields[0]}:{fields[1]}-{fields[2]}' if offset else \
                    f"{fields[0].split('_')[0]}:{fields[0].split('_')[1]}-{fields[0].split('_')[2]}"

                if fields[1 + offset] == '':
                    continue

                genes, dists, types = \
                    fields[1 + offset].split(';'), fields[2 + offset].split(';'), fields[3 + offset].split(';')

                for gene, dist, annot in zip(genes, dists, types):
                    if annot == 'promoter':
                        promoter_dict.setdefault(gene, []).append(peak)
                    elif annot == 'distal':
                        if dist == '0':
                            gene_body_dict.setdefault(gene, []).append(peak)
                        else:
                            distal_dict.setdefault(gene, []).append(peak)

        # Read linkages
        linkage_path = os.path.join(outs_data_path, 'analysis', 'feature_linkage', self.linkage_fn)
        if not os.path.exists(linkage_path):
            main_exception(f'The path to the linkage file ({linkage_path}) does not exist.')
        with open(linkage_path) as f:
            for line in f:
                fields = line.rstrip().split('\t')

                # Form proper peak coordinates
                peak_1, peak_2 = f'{fields[0]}:{fields[1]}-{fields[2]}', f'{fields[3]}:{fields[4]}-{fields[5]}'

                # Split the gene pairs
                genes_annots_1, genes_annots_2 = \
                    fields[6].split('><')[0][1:].split(';'), fields[6].split('><')[1][:-1].split(';')

                # Extract correlation
                correlation = float(fields[7])

                # Extract distance between peaks
                dist = float(fields[11])

                if fields[12] == 'peak-peak':
                    for gene_annot_1 in genes_annots_1:
                        gene_1, annot_1 = gene_annot_1.split('_')
                        for gene_annot_2 in genes_annots_2:
                            gene_2, annot_2 = gene_annot_2.split('_')

                            if (((annot_1 == 'promoter') != (annot_2 == 'promoter')) and
                                    ((gene_1 == gene_2) or (dist < self.max_peak_dist))):
                                gene = gene_1 if annot_1 == 'promoter' else gene_2

                                if (peak_2 not in corr_dict.get(gene, []) and annot_1 == 'promoter' and
                                        (gene_2 not in gene_body_dict or peak_2 not in gene_body_dict.get(gene_2, []))):
                                    corr_dict.setdefault(gene, [[], []])[0].append(peak_2)
                                    corr_dict[gene][1].append(correlation)

                                if (peak_1 not in corr_dict.get(gene, []) and annot_2 == 'promoter' and
                                        (gene_1 not in gene_body_dict or peak_1 not in gene_body_dict.get(gene_1, []))):
                                    corr_dict.setdefault(gene, [[], []])[0].append(peak_1)
                                    corr_dict[gene][1].append(correlation)

                elif fields[12] == 'peak-gene':
                    gene_2 = genes_annots_2[0]
                    for gene_annot_1 in genes_annots_1:
                        gene_1, annot_1 = gene_annot_1.split('_')

                        if (gene_1 == gene_2) or (dist < self.max_peak_dist):
                            gene = gene_1

                            if (peak_1 not in corr_dict.get(gene, []) and annot_1 != 'promoter' and
                                    (gene_1 not in gene_body_dict or peak_1 not in gene_body_dict.get(gene_1, []))):
                                corr_dict.setdefault(gene, [[], []])[0].append(peak_1)
                                corr_dict[gene][1].append(correlation)

                elif fields[12] == 'gene-peak':
                    gene_1 = genes_annots_1[0]
                    for gene_annot_2 in genes_annots_2:
                        gene_2, annot_2 = gene_annot_2.split('_')

                        if (gene_1 == gene_2) or (dist < self.max_peak_dist):
                            gene = gene_1

                            if (peak_2 not in corr_dict.get(gene, []) and annot_2 != 'promoter' and
                                    (gene_2 not in gene_body_dict or peak_2 not in gene_body_dict.get(gene_2, []))):
                                corr_dict.setdefault(gene, [[], []])[0].append(peak_2)
                                corr_dict[gene][1].append(correlation)

        cre_dict = {}
        gene_dict = promoter_dict
        promoter_genes = list(promoter_dict.keys())

        for gene in promoter_genes:
            if self.include_gene_body:  # add gene-body peaks
                if gene in gene_body_dict:
                    for peak in gene_body_dict[gene]:
                        if peak not in gene_dict[gene]:
                            gene_dict[gene].append(peak)
            cre_dict[gene] = []
            if gene in corr_dict:  # add enhancer peaks
                for j, peak in enumerate(corr_dict[gene][0]):
                    corr = corr_dict[gene][1][j]
                    if corr > self.min_corr:
                        if peak not in gene_dict[gene]:
                            gene_dict[gene].append(peak)
                            cre_dict[gene].append(peak)

        # Update the enhancer and promoter dictionaries
        self._update_cre_and_promoter_dicts(cre_dict=cre_dict,
                                            promoter_dict=promoter_dict)

        linkage_logger.finish_progress(progress_name='compute_linkages_via_cellranger')

    def compute_linkages_via_cicero(self) -> None:
        # TODO: Use cicero to filter significant linkages
        pass

    def compute_linkages_via_scenicplus(self) -> None:
        # TODO: Use scenicplus to filter significant linkages
        pass

    def compute_neighbors(self,
                          atac_lsi_key:        str = MDKM.ATAC_OBSM_LSI_KEY,
                          lr:                  float = 0.0001,
                          max_epochs:          int = 10, # 10 for debug mode 500 for release,
                          mv_algorithm:        bool = True,
                          n_comps_atac:        int = 20,
                          n_comps_rna:         int = 20,
                          n_neighbors:         int = 20,
                          pc_key:              str = MDKM.ATAC_OBSM_PC_KEY,
                          random_state:        int = 42,
                          rna_pca_key:         str = MDKM.RNA_OBSM_PC_KEY,
                          scale_factor:        float = 1e4,
                          use_highly_variable: bool = False
                          ) -> None:
        if self.neighbor_method == 'multivi':
            self.compute_neighbors_via_multivi(
                lr=lr,
                max_epochs=max_epochs)
        elif self.neighbor_method == 'wnn':
            self.weighted_nearest_neighbors(
                atac_lsi_key=atac_lsi_key,
                n_components_atac=n_comps_atac,
                n_components_rna=n_comps_rna,
                nn=n_neighbors,
                random_state=random_state,
                rna_pca_key=rna_pca_key,
                use_highly_variable=use_highly_variable)
        else:
            main_exception(f'Unrecognized method to compute neighbors ({self.neighbor_method}) requested.')

    def compute_neighbors_via_multivi(
            self,
            lr:          float = 0.0001,
            max_epochs:  int = 500,
            n_comps:     int = 20,
            n_neighbors: int = 20,
    ) -> None:
        import scvi
        import scanpy as sc
        main_info('Computing nearest neighbors in latent representation generated by MULTIVI ...', indent_level=1)
        nn_logger = LoggerManager.gen_logger('compute_nn_via_mvi')
        nn_logger.log_time()

        # Extract the ATAC-seq and RNA-seq portions
        atac_adata, rna_adata = self.mdata.mod['atac'], self.mdata.mod['rna']
        n_peaks, n_genes = atac_adata.n_vars, rna_adata.n_vars

        # Ensure that the ATAC- and RNA-seq portions have same number of cells
        assert (atac_adata.n_obs == rna_adata.n_obs)

        # Restructure the data into MULTIVI format - we do not perform TF-IDF transformation
        # ... X - counts or normalized counts???
        tmp_adata_X = hstack([rna_adata.layers[MDKM.RNA_COUNTS_LAYER], atac_adata.layers[MDKM.ATAC_COUNTS_LAYER]])

        # ... obs
        tmp_adata_obs = rna_adata.obs.copy()

        # ... var
        tmp_adata_var = pd.concat([rna_adata.var.copy(), atac_adata.var.copy()], join='inner', axis=0)

        tmp_adata = AnnData(X=tmp_adata_X.copy(), obs=tmp_adata_obs, var=tmp_adata_var)
        tmp_adata.layers['counts'] = tmp_adata.X.copy()

        # Get the number of cells
        num_cells = tmp_adata.n_obs

        # Generate a random permutation of cell indices
        cell_indices = np.random.permutation(num_cells)

        # Determine the split point
        split_point = num_cells // 2

        # Split indices into two groups
        cell_indices_1 = cell_indices[:split_point]
        cell_indices_2 = cell_indices[split_point:]

        # Subset the AnnData object into two disjoint AnnData objects
        tmp_adata_1 = tmp_adata[cell_indices_1].copy()
        tmp_adata_1.obs['modality'] = 'first_set'
        tmp_adata_2 = tmp_adata[cell_indices_2].copy()
        tmp_adata_2.obs['modality'] = 'second_set'

        tmp_adata = scvi.data.organize_multiome_anndatas(tmp_adata_1, tmp_adata_2)

        # Run MULTIVI
        # ... setup AnnData object for scvi-tools
        main_info('Setting up combined data for MULTIVI', indent_level=2)
        scvi.model.MULTIVI.setup_anndata(tmp_adata, batch_key='modality')

        # ... instantiate the SCVI model
        main_info('Instantiating MULTIVI model', indent_level=2)
        multivi_model = scvi.model.MULTIVI(adata=tmp_adata, n_genes=n_genes, n_regions=n_peaks, n_latent=n_comps)
        multivi_model.view_anndata_setup()

        # ... train the model
        main_info('Training MULTIVI model', indent_level=2)
        multivi_model.train(max_epochs=max_epochs, lr=lr)

        # Extract latent representation
        main_info('extracting latent representation for ATAC-seq', indent_level=3)
        atac_adata.obsm['X_mvi_latent'] = multivi_model.get_latent_representation().copy()
        rna_adata.obsm['X_mvi_latent'] = multivi_model.get_latent_representation().copy()

        # Compute nearest neighbors
        main_info('Computing nearest neighbors in MVI latent representation', indent_level=2)
        sc.pp.neighbors(rna_adata, n_neighbors=n_neighbors, n_pcs=n_comps, use_rep='X_mvi_latent')

        # Redundantly copy over to atac-seq modality
        atac_adata.obsp['distances'] = rna_adata.obsp['distances'].copy()
        atac_adata.obsp['connectivities'] = rna_adata.obsp['connectivities'].copy()
        atac_adata.uns['neighbors'] = rna_adata.uns['neighbors'].copy()

        # Extract the matrix storing the distances between each cell and its neighbors
        cx = coo_matrix(rna_adata.obsp['distances'].copy())

        # the number of cells
        cells = rna_adata.obsp['distances'].shape[0]

        # define the shape of our final results
        # and make the arrays that will hold the results
        new_shape = (cells, n_neighbors)
        nn_dist = np.zeros(shape=new_shape)
        nn_idx = np.zeros(shape=new_shape)

        # new_col defines what column we store data in our result arrays
        new_col = 0

        # loop through the distance matrices
        for i, j, v in zip(cx.row, cx.col, cx.data):
            # store the distances between neighbor cells
            nn_dist[i][new_col % n_neighbors] = v

            # for each cell's row, store the row numbers of its neighbor cells
            # (1-indexing instead of 0- is a holdover from R multimodalneighbors())
            nn_idx[i][new_col % n_neighbors] = int(j) + 1

            new_col += 1

        # Add index and distance to the MultiomeVelocity object
        self.nn_idx = nn_idx
        self.nn_dist = nn_dist

        # Copy the subset AnnData scRNA-seq and scATAC-seq objects back into the MultiomeVelocity object
        self.mdata.mod['atac'] = atac_adata.copy()
        self.mdata.mod['rna'] = rna_adata.copy()

        nn_logger.finish_progress(progress_name='compute_nn_via_mvi')

    def compute_second_moments(
            self,
            adjusted: bool = False
    ) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
        # Extract transcriptome
        rna_adata = self.mdata.mod['rna']

        # Obtain connectivities matrix
        connectivities = get_connectivities(rna_adata)

        s, u = (csr_matrix(rna_adata.layers[MDKM.RNA_SPLICED_LAYER]),
                csr_matrix(rna_adata.layers[MDKM.RNA_UNSPLICED_LAYER]))
        if s.shape[0] == 1:
            s, u = s.T, u.T
        Mss = csr_matrix.dot(connectivities, s.multiply(s)).astype(np.float32).A
        Mus = csr_matrix.dot(connectivities, s.multiply(u)).astype(np.float32).A
        Muu = csr_matrix.dot(connectivities, u.multiply(u)).astype(np.float32).A
        if adjusted:
            Mss = 2 * Mss - rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER].reshape(Mss.shape)
            Mus = 2 * Mus - rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER].reshape(Mus.shape)
            Muu = 2 * Muu - rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER].reshape(Muu.shape)
        return Mss, Mus, Muu

    def compute_velocities(self,
                           linkage_method:  Optional[Literal['cellranger', 'cicero', 'scenic+']] = 'cellranger',
                           mode:            Literal['deterministic', 'stochastic'] = 'deterministic',
                           neighbor_method: Literal['multivi', 'wnn'] = 'wnn',
                           num_processes:   int = 6) -> None:
        if linkage_method is not None:
            self.linkage_method = linkage_method

        if neighbor_method is not None:
            self.neighbor_method = neighbor_method

        if (self.linkage_method is None) or (self.neighbor_method is None):
            main_exception('linkage_method and neighbor_method mus be specified.')

        # Compute linkages
        self.compute_linkages()

        # Compute neighbors
        self.compute_neighbors()

        # Compute smoother accessibility
        self.knn_smoothed_chrom()

        # Compute transcriptomic velocity
        self.transcriptomic_velocity(mode=mode, num_processes=num_processes)

        # Compute lift of transcriptomic velocity
        self.lift_transcriptomic_velocity(num_processes=num_processes)

    def find_cell_along_integral_curve(self,
                                       num_processes:    int = 6,
                                       plot_dir_cosines: bool = False):
        # Extract the ATAC- and RNA-seq portions
        atac_adata, rna_adata = self.mdata.mod['atac'], self.mdata.mod['rna']

        expression_mtx = rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER]
        velocity_mtx = rna_adata.layers[MDKM.RNA_SPLICED_VELOCITY_LAYER]

        # Extract connectivities
        connectivities = get_connectivities(rna_adata)

        # Get non-zero indices from connectivities
        nonzero_idx = connectivities.nonzero()

        # Prepare argument list for parallel processing
        args_list = [(i, j, expression_mtx, velocity_mtx)
                     for i, j in zip(nonzero_idx[0], nonzero_idx[1])]

        # Use multiprocessing to compute the results
        with Pool(processes=num_processes) as pool:
            results = pool.map(direction_cosine, args_list)

        # Convert results to sparse matrix
        data = [cosines for _, _, cosines in results]
        i_indices = [i_idx for i_idx, _, _ in results]
        j_indices = [j_idx for _, j_idx, _ in results]
        direction_cosines = csr_matrix((data, (i_indices, j_indices)), shape=connectivities.shape)

        # Find nearest neighbor along integral curve
        integral_neighbors = direction_cosines.argmax(axis=1).A.flatten()

        if plot_dir_cosines:
            # Summarize statistics about the best direction cosines
            max_dir_cosines = direction_cosines.max(axis=1).A.flatten()
            plt.hist(max_dir_cosines, bins=25)
            plt.title('Frequencies of direction cosines')
            plt.xlabel('Direction Cosines')
            plt.ylabel('Frequency')
            plt.show()

        # Save the results in this class
        # TODO: Consider whether to add to AnnData objects
        self.cosine_similarities = direction_cosines
        self.integral_neighbors = {int(idx): int(integral_neighbor)
                                   for idx, integral_neighbor in enumerate(integral_neighbors)}

    @classmethod
    def from_mdata(cls,
                   mdata):
        from mudata import MuData
        # Deep copy MuData object for export
        atac_adata, rna_adata = mdata.mod['atac'].copy(), mdata.mod['rna'].copy()

        # ... from atac
        # ... bit of kludge: dictionaries appear to require type casting after deserialization
        deser_cre_dict = atac_adata.uns['cre_dict'].copy()
        cre_dict = {}
        for gene, cre_list in deser_cre_dict.items():
            cre_dict[str(gene)] = [str(cre) for cre in cre_list]
        # ... bit of kludge: dictionaries appear to require type casting after deserialization
        deser_promoter_dict = atac_adata.uns['promoter_dict']
        promoter_dict = {}
        for gene, promoter_list in deser_promoter_dict.items():
            promoter_dict[str(gene)] = [str(promoter) for promoter in promoter_list]

        multi_dynamo_kwargs = atac_adata.uns['multi_dynamo_kwargs']
        include_gene_body = multi_dynamo_kwargs.get('include_gene_body', False)
        linkage_fn = multi_dynamo_kwargs.get('linkage_fn', 'feature_linkage.bedpe')
        linkage_method = multi_dynamo_kwargs.get('linkage_method', 'cellranger')
        max_peak_dist = multi_dynamo_kwargs.get('max_peak_dist', 10000)
        min_corr = multi_dynamo_kwargs.get('min_corr', 0.5)
        peak_annot_fn = multi_dynamo_kwargs.get('min_corr', 'peak_annotation.tsv')

        # ... from rna
        nn_dist = rna_adata.obsm['multi_dynamo_nn_dist']
        nn_idx = rna_adata.obsm['multi_dynamo_nn_idx']

        cosine_similarities = rna_adata.obsp['cosine_similarities']
        # ... bit of kludge: dictionaries appear to require type casting after deserialization
        integral_neighbors = {int(k): int(v) for k,v in rna_adata.uns['integral_neighbors'].items()}

        multi_dynamo_kwargs = rna_adata.uns['multi_dynamo_kwargs']
        neighbor_method = multi_dynamo_kwargs.get('neighbor_method', 'multivi')

        multi_velocity = cls(mdata=mdata,
                             cre_dict=cre_dict,
                             cosine_similarities=cosine_similarities,
                             include_gene_body=include_gene_body,
                             integral_neighbors=integral_neighbors,
                             linkage_fn=linkage_fn,
                             linkage_method=linkage_method,
                             max_peak_dist=max_peak_dist,
                             min_corr=min_corr,
                             nn_dist=nn_dist,
                             nn_idx=nn_idx,
                             neighbor_method=neighbor_method,
                             peak_annot_fn=peak_annot_fn,
                             promoter_dict=promoter_dict)

        return multi_velocity

    def get_cre_dict(self):
        return self._cre_dict

    def get_mdata(self):
        return self.mdata

    def get_nn_dist(self):
        return self.nn_dist

    def get_nn_idx(self):
        return self.nn_idx

    def get_promoter_dict(self):
        return self._promoter_dict

    # knn_smoothed_chrom - method adapted from MultiVelo
    def knn_smoothed_chrom(self,
                           nn: int = 20
                           ) -> None:
        # Consistency checks
        nn_idx = None
        if self.nn_idx is None:
            main_exception('Missing KNN index matrix.  Try calling compute_neighbors first.')
        else:
            nn_idx = self.nn_idx

        nn_dist = None
        if self.nn_dist is None:
            main_exception('Missing KNN distance matrix.  Try calling compute_neighbors first.')
        else:
            nn_dist = self.nn_dist

        atac_adata, rna_adata = self.mdata.mod['atac'], self.mdata.mod['rna']
        n_cells = atac_adata.n_obs

        if (nn_idx.shape[0] != n_cells) or (nn_dist.shape[0] != n_cells):
            main_exception('Number of rows of KNN indices does not equal to number of cells.')

        X = coo_matrix(([], ([], [])), shape=(n_cells, 1))
        from umap.umap_ import fuzzy_simplicial_set
        conn, sigma, rho, dists = fuzzy_simplicial_set(X=X,
                                                       n_neighbors=nn,
                                                       random_state=None,
                                                       metric=None,
                                                       knn_indices=nn_idx-1,
                                                       knn_dists=nn_dist,
                                                       return_dists=True)

        conn = conn.tocsr().copy()
        n_counts = (conn > 0).sum(1).A1
        if nn is not None and nn < n_counts.min():
            conn = top_n_sparse(conn, nn)
        conn.setdiag(1)
        conn_norm = conn.multiply(1.0 / conn.sum(1)).tocsr()

        # Compute first moment of chromatin accessibility
        atac_adata.layers[MDKM.RNA_FIRST_MOMENT_CHROM_LAYER] = \
            csr_matrix.dot(conn_norm, atac_adata.layers['counts']).copy()

        # Overwrite ATAC- and RNA-seq connectivities
        atac_adata.obsp['connectivities'] = conn.copy()
        rna_adata.obsp['connectivities'] = conn.copy()

        self.mdata.mod['atac'] = atac_adata.copy()
        self.mdata.mod['rna'] = rna_adata.copy()

    def lift_transcriptomic_velocity(self,
                                     num_processes:      int = 6):
        # Compute integral neighbors
        main_info('Starting computation of integral neighbors ...')
        self.find_cell_along_integral_curve(num_processes=num_processes)

        # Extract the ATAC- and RNA-seq data
        atac_adata, rna_adata = self.mdata.mod['atac'], self.mdata.mod['rna']

        # Retrieve specified layer for chromatin state
        chromatin_state = atac_adata.layers[MDKM.ATAC_TFIDF_LAYER]

        cosine_similarities = None
        if self.cosine_similarities is None:
            main_exception('Please compute integral neighbors before calling lift_transcriptomic_velocity.')
        else:
            cosine_similarities = self.cosine_similarities

        # Retrieve specified layer for expression matrix
        expression_mtx = rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER]

        integral_neighbors = None
        if self.integral_neighbors is None:
            main_exception('Please compute integral neighbors before calling lift_transcriptomic_velocity.')
        else:
            integral_neighbors = self.integral_neighbors

        # Retrieve specified layer for the velocity matrix
        velocity_mtx = rna_adata.layers[MDKM.RNA_SPLICED_VELOCITY_LAYER]

        # Prepare argument list for parallel processing
        args_list = [(i, j, chromatin_state, cosine_similarities, expression_mtx, velocity_mtx[i, :])
                     for i, j in integral_neighbors.items()]

        # Use multiprocessing to compute the results
        with Pool(processes=num_processes) as pool:
            results = pool.map(lifted_chromatin_velocity, args_list)

        # Convert results to sparse matrix
        chromatin_velocity_mtx = np.zeros(chromatin_state.shape)
        for i, chromatin_velocity in results:
            chromatin_velocity_mtx[i, :] = chromatin_velocity

        atac_adata.layers[MDKM.ATAC_CHROMATIN_VELOCITY_LAYER] = chromatin_velocity_mtx

        # Copy the scATAC-seq AnnData object into the MultiomeVelocity object
        self.mdata.mod['atac'] = atac_adata.copy()

    def _restrict_dicts_to_gene_list(self,
                                     gene_list:     List[str],
                                     cre_dict:      Dict[str, List[str]] = None,
                                     promoter_dict: Dict[str, List[str]] = None
                                     ) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
        # Elements present in scATAC-seq data
        present_elements = self.atac_elements()

        if len(gene_list) == 0:
            main_exception('Require non-trivial gene_list for _restrict_to_gene_list.')

        if len(cre_dict) == 0 or len(promoter_dict) == 0:
            main_exception('Require non-trivial enhancer and promoter dicts for _restrict_to_gene_list.')

        # Elements associated to genes in gene_list and present in scATAC-seq data
        shared_elements = []

        # Dictionary from gene to element list for all genes present in gene_list and with
        # corresponding elements in enhancer dicts
        shared_cre_dict = {}
        for gene, element_list in cre_dict.items():
            if gene in gene_list:
                shared_elements_for_gene =\
                    [element for element in element_list if element in present_elements]
                shared_elements_for_gene = list(set(shared_elements_for_gene))

                shared_elements += shared_elements_for_gene
                shared_cre_dict[gene] = shared_elements_for_gene

        # Add all promoters for genes in gene_list
        shared_promoter_dict = {}
        for gene, element_list in promoter_dict.items():
            if gene in gene_list:
                shared_elements_for_gene = \
                    [element for element in element_list if element in present_elements]
                shared_elements_for_gene = list(set(shared_elements_for_gene))  # Bit pedantic ...

                shared_elements += shared_elements_for_gene
                shared_promoter_dict[gene] = shared_elements_for_gene

        # Make elements into unique list
        shared_elements = list(set(shared_elements))

        # Determine which genes actually have elements present in the scATAC-seq data
        all_dict_genes = list(set(list(shared_cre_dict.keys()) + list(shared_promoter_dict.keys())))
        shared_genes = []
        for gene in all_dict_genes:
            enhancers_for_gene = len(shared_cre_dict.get(gene, [])) > 0

            promoters_for_gene = len(shared_promoter_dict.get(gene, [])) > 0

            if enhancers_for_gene or promoters_for_gene:
                shared_genes.append(gene)

            # Clean up trivial entries in dicts
            if not enhancers_for_gene and gene in shared_cre_dict:
                del shared_cre_dict[gene]

            if not promoters_for_gene and gene in shared_promoter_dict:
                del shared_promoter_dict[gene]

        shared_genes = list(set(shared_genes))

        return shared_elements, shared_genes, shared_cre_dict, shared_promoter_dict

    def restrict_to_gene_list(self,
                              gene_list: List[str] = None,
                              subset:    bool = False) -> Tuple[List[str], List[str]]:
        # Extract genes from scRNA-seq data
        rna_genes = self.rna_genes()

        if gene_list is None:
            # If no gene_list offered, then use the genes found in scRNA-seq dataset
            gene_list = rna_genes
        else:
            # Otherwise ensure gene is contained within the shared list
            if not set(gene_list).issubset(set(rna_genes)):
                main_exception('gene_list is not a subset of genes found in scRNA-seq dataset.')

        shared_elements, shared_genes, shared_enhancer_dict, shared_promoter_dict = \
            self._restrict_dicts_to_gene_list(gene_list=gene_list,
                                              cre_dict=self._cre_dict,
                                              promoter_dict=self._promoter_dict)

        if subset:
            # Subset the scATAC-seq data to shared elements
            self.mdata.mod['atac'] = self.mdata.mod['atac'][:, shared_elements].copy()

            # Subset the scRNA_seq data to shared genes
            self.mdata.mod['rna'] = self.mdata.mod['rna'][:, shared_genes].copy()

        return shared_elements, shared_genes

    def rna_genes(self):
        return self.mdata.mod['rna'].var_names.tolist()

    def to_mdata(self):
        from mudata import MuData
        # Deep copy MuData object for export
        atac_adata, rna_adata = self.mdata.mod['atac'].copy(), self.mdata.mod['rna'].copy()

        # ... embellish atac
        atac_adata.uns['cre_dict'] = self._cre_dict.copy()
        atac_adata.uns['promoter_dict'] = self._promoter_dict.copy()
        atac_adata.uns['multi_dynamo_kwargs'] = {'include_gene_body': self.include_gene_body,
                                                 'linkage_fn':        self.linkage_fn,
                                                 'linkage_method':    self.linkage_method,
                                                 'max_peak_dist':     self.max_peak_dist,
                                                 'min_corr':          self.min_corr,
                                                 'peak_annot_fn':     self.peak_annot_fn}

        # ... embellish rna
        rna_adata.obsm['multi_dynamo_nn_dist'] = self.nn_dist.copy()
        rna_adata.obsm['multi_dynamo_nn_idx'] = self.nn_idx.copy()

        rna_adata.obsp['cosine_similarities'] = self.cosine_similarities.copy()
        rna_adata.uns['integral_neighbors'] = {str(k): str(v) for k,v in self.integral_neighbors.items()}.copy()
        rna_adata.uns['multi_dynamo_kwargs'] = {'neighbor_method': self.neighbor_method}

        return MuData({'atac': atac_adata, 'rna': rna_adata})

    # transcriptomic_velocity: this could really be any of the many methods that already exist, including those in
    # dynamo and we plan to add this capability later.
    def transcriptomic_velocity(self,
                                adjusted:      bool = False,
                                min_r2:        float = 1e-2,
                                mode:          Literal['deterministic', 'stochastic'] = 'deterministic',
                                n_neighbors:   int = 20,
                                n_pcs:         int = 20,
                                num_processes: int = 6,
                                outlier:       float = 99.8):
        # Extract transcriptome and chromatin accessibility
        atac_adata, rna_adata = self.mdata.mod['atac'], self.mdata.mod['rna']

        # Assemble dictionary of arguments for fits
        fit_args = {'min_r2':      min_r2,
                    'mode':        mode,
                    'n_pcs':       n_pcs,
                    'n_neighbors': n_neighbors,
                    'outlier':     outlier}

        # Obtain connectivities from the scRNA-seq object
        rna_conn = rna_adata.obsp['connectivities']

        # Compute moments for transcriptome data
        main_info('computing moments for transcriptomic data ...')
        rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER] = (
            csr_matrix.dot(rna_conn, csr_matrix(rna_adata.layers[MDKM.RNA_SPLICED_LAYER]))
            .astype(np.float32)
            .toarray()
        )
        rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER] = (
            csr_matrix.dot(rna_conn, csr_matrix(rna_adata.layers[MDKM.RNA_UNSPLICED_LAYER]))
            .astype(np.float32)
            .toarray()
        )

        # Initialize select second moments for the transcriptomic data
        Mss, Mus, Muu = None, None, None
        if mode == 'stochastic':
            main_info('computing second moments', indent_level=2)
            Mss, Mus, Muu = self.compute_second_moments(adjusted=adjusted)

            rna_adata.layers[MDKM.RNA_SECOND_MOMENT_SS_LAYER] = Mss.copy()
            rna_adata.layers[MDKM.RNA_SECOND_MOMENT_US_LAYER] = Mus.copy()
            rna_adata.layers[MDKM.RNA_SECOND_MOMENT_UU_LAYER] = Muu.copy()

        if 'highly_variable' in rna_adata.var:
            main_info('using highly variable genes', indent_level=2)
            rna_gene_list = rna_adata.var_names[rna_adata.var['highly_variable']].values
        else:
            rna_gene_list = rna_adata.var_names.values[
                (~np.isnan(np.asarray(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER].sum(0))
                           .reshape(-1)
                           if issparse(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER])
                           else np.sum(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER], axis=0)))
                & (~np.isnan(np.asarray(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER].sum(0))
                             .reshape(-1)
                             if issparse(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER])
                             else np.sum(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER], axis=0)))]

        # Restrict to genes with corresponding peaks in scATAC-seq data
        shared_elements, shared_genes = self.restrict_to_gene_list(gene_list=rna_gene_list,
                                                                   subset=True)

        n_fitted_genes = len(shared_genes)
        if n_fitted_genes:
            main_info(f'{n_fitted_genes} genes will be fitted')
        else:
            main_exception('None of the genes specified are in the adata object')

        velo_s = np.zeros((rna_adata.n_obs, n_fitted_genes))
        variance_velo_s = np.zeros((rna_adata.n_obs, n_fitted_genes))
        gammas = np.zeros(n_fitted_genes)
        r2s = np.zeros(n_fitted_genes)
        losses = np.zeros(n_fitted_genes)

        u_mat = (rna_adata[:, shared_genes].layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER].A
                 if issparse(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER])
                 else rna_adata[:, shared_genes].layers[MDKM.RNA_FIRST_MOMENT_UNSPLICED_LAYER])
        s_mat = (rna_adata[:, shared_genes].layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER].A
                 if issparse(rna_adata.layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER])
                 else rna_adata[:, shared_genes].layers[MDKM.RNA_FIRST_MOMENT_SPLICED_LAYER])

        M_c = csr_matrix(atac_adata[:, shared_elements].layers[MDKM.RNA_FIRST_MOMENT_CHROM_LAYER]) \
            if issparse(atac_adata.layers[MDKM.RNA_FIRST_MOMENT_CHROM_LAYER]) else \
            atac_adata[:, shared_elements].layers[MDKM.RNA_FIRST_MOMENT_CHROM_LAYER]
        c_mat = M_c.toarray() if issparse(M_c) else M_c

        # Create dictionary from gene to index
        gene_to_idx_dict = {gene: idx for idx, gene in enumerate(shared_genes)}

        # Create dictionary from peak to index
        peak_to_idx_dict = {element: idx for idx, element in enumerate(shared_elements)}

        # Create unified gene to list of elements dict
        tmp_elements_for_gene_dict = {}
        for gene, element_list in self._cre_dict.items():
            tmp_elements_for_gene_dict[gene] = tmp_elements_for_gene_dict.setdefault(gene, []) + element_list

        for gene, element_list in self._promoter_dict.items():
            tmp_elements_for_gene_dict[gene] = tmp_elements_for_gene_dict.setdefault(gene, []) + element_list

        elements_for_gene_dict = {}
        for gene, element_list in tmp_elements_for_gene_dict.items():
            elements_for_gene_dict[gene] = list(set(element_list))

        # Create dictionary from gene indices to list of peaks by indices
        gene_idx_to_peak_idx = {gene_to_idx_dict[gene]: [peak_to_idx_dict[peak] for peak in peak_list]
                                for gene, peak_list in elements_for_gene_dict.items()}

        # Define batch arguments
        batches_of_arguments = []
        for i in range(n_fitted_genes):
            gene = shared_genes[i]
            peak_idx = gene_idx_to_peak_idx[i]

            batches_of_arguments.append(
                (c_mat[:, peak_idx],
                 u_mat[:, i],
                 s_mat[:, i],
                 None if mode == 'deterministic' else Mss[:, i],
                 None if mode == 'deterministic' else Mus[:, i],
                 None if mode == 'deterministic' else Muu[:, i],
                 fit_args,
                 mode,
                 gene))

        # Carry out fits in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(regression, batches_of_arguments)

        # Reformat the results
        for idx, (velocity, velocity_variance, gamma, r2, loss) in enumerate(results):
            gammas[idx] = gamma
            r2s[idx] = r2
            losses[idx] = loss
            velo_s[:, idx] = smooth_scale(rna_conn, velocity)

            if mode == 'stochastic':
                variance_velo_s[:, idx] = smooth_scale(rna_conn,
                                                       velocity_variance)

        # Determine which fits failed
        kept_genes = [gene for gene, loss in zip(shared_genes, losses) if loss != np.inf]
        if len(kept_genes) == 0:
            main_exception('None of the genes were fit due to low quality.')

        # Subset the transcriptome to the genes for which the fits were successful
        rna_copy = rna_adata[:, kept_genes].copy()

        # Add the fit results
        keep = [loss != np.inf for loss in losses]

        # ... layers
        rna_copy.layers[MDKM.RNA_SPLICED_VELOCITY_LAYER] = csr_matrix(velo_s[:, keep])
        if mode == 'stochastic':
            rna_copy.layers['variance_velo_s'] = csr_matrix(variance_velo_s[:, keep])

        # ... .obsp
        rna_copy.obsp['_RNA_conn'] = rna_conn

        # ... .uns
        # ... ... augment the dynamical and normalization information
        dyn_and_norm_info = rna_copy.uns['pp'].copy()
        dyn_and_norm_info['experiment_total_layers'] = None
        dyn_and_norm_info['layers_norm_method'] = None
        dyn_and_norm_info['tkey'] = None
        rna_copy.uns['pp'] = dyn_and_norm_info.copy()

        dynamics = {'filter_gene_mode': 'final',
                    't': None,
                    'group': None,
                    'X_data': None,
                    'X_fit_data': None,
                    'asspt_mRNA': 'ss',
                    'experiment_type': dyn_and_norm_info.get('experiment_type', 'conventional'),
                    'normalized': True,
                    'model': mode,
                    'est_method': 'gmm',  # Consider altering
                    'has_splicing': dyn_and_norm_info.get('has_splicing', True),
                    'has_labeling': dyn_and_norm_info.get('has_labeling', False),
                    'splicing_labeling': dyn_and_norm_info.get('splicing_labeling', False),
                    'has_protein': dyn_and_norm_info.get('has_protein', False),
                    'use_smoothed': True,
                    'NTR_vel': False,
                    'log_unnormalized': True,
                    # Ensure X is indeed log normalized (compute exp1m, sum and check rowsums)
                    'fraction_for_deg': False}
        rna_copy.uns['dynamics'] = dynamics.copy()

        rna_copy.uns['velo_s_params'] = {'mode': mode,
                                         'fit_offset': False,
                                         'perc': outlier}
        rna_copy.uns['velo_s_params'].update(fit_args)

        # ... ... These are the column names for the array in .varm['vel_params']
        rna_copy.uns['vel_params_names'] = ['beta', 'gamma', 'half_life', 'alpha_b', 'alpha_r2', 'gamma_b',
                                            'gamma_r2', 'gamma_logLL', 'delta_b', 'delta_r2', 'bs', 'bf',
                                            'uu0', 'ul0', 'su0', 'sl0', 'U0', 'S0', 'total0']

        # ... .var
        rna_copy.var['fit_gamma'] = gammas[keep]
        rna_copy.var['fit_loss'] = losses[keep]
        rna_copy.var['fit_r2'] = r2s[keep]

        # Introduce var['use_for_dynamics'] for dynamo
        v_gene_ind = rna_copy.var['fit_r2'] >= min_r2
        rna_copy.var['use_for_dynamics'] = v_gene_ind
        rna_copy.var['velo_s_genes'] = v_gene_ind

        # ... .varm
        vel_params_array = np.full((rna_copy.shape[1], len(rna_copy.uns['vel_params_names'])), np.nan)

        # ... ... ... transfer 'gamma'
        gamma_index = np.where(np.array(rna_copy.uns['vel_params_names']) == 'gamma')[0][0]
        vel_params_array[:, gamma_index] = rna_copy.var['fit_gamma']

        # ... ... ... transfer 'gamma_r2'
        gamma_r2_index = np.where(np.array(rna_copy.uns['vel_params_names']) == 'gamma_r2')[0][0]
        vel_params_array[:, gamma_r2_index] = rna_copy.var['fit_r2']

        rna_copy.varm['vel_params'] = vel_params_array

        # Copy the subset AnnData scRNA-seq and scATAC-seq objects back into the MultiomeVelocity object
        self.mdata.mod['rna'] = rna_copy.copy()

        # Filter the scATAC-seq peaks to retain only those corresponding to fit genes
        shared_elements, shared_genes = self.restrict_to_gene_list(gene_list=kept_genes,
                                                                   subset=True)

        # Confer same status to element corresponding to genes declared as 'use_for_dynamics'
        v_genes = [gene for gene, v_ind in zip(shared_genes, v_gene_ind) if v_ind]
        # v_elements, v_genes = self.restrict_to_gene_list(gene_list=v_genes, subset=False)
        # v_element_ind = [element in v_elements for element in shared_elements]
        # TODO: Need to special case when no genes rise to significance
        v_element_ind = [True for _ in range(atac_adata.n_vars)]

        # Introduce var['use_for_dynamics'] for dynamo
        # TODO: This does NOT appear to work properly yet - so left permissive
        atac_adata.var['use_for_dynamics'] = v_element_ind

        self.mdata.mod['atac'] = atac_adata.copy()

    def _update_cre_and_promoter_dicts(self,
                                       cre_dict:      Dict[str, List[str]] = None,
                                       promoter_dict: Dict[str, List[str]] = None):
        if cre_dict is not None or promoter_dict is not None:
            # Should only have exogenous enhancer and promoter dicts if none are present in object
            if self._cre_dict is not None or self._promoter_dict is not None:
                main_exception('Should only specify exogenous CRE and promoter dicts if none are present in object.')
        else:
            # Extract the dictionaries
            cre_dict = self._cre_dict
            promoter_dict = self._promoter_dict

        # Extract the RNA genes
        rna_genes = self.rna_genes()

        # ... determine which genes are actually present in the scATAC-seq data and for these
        #     which elements are present
        shared_elements, shared_genes, shared_cre_dict, shared_promoter_dict = \
            self._restrict_dicts_to_gene_list(gene_list=rna_genes,
                                              cre_dict=cre_dict,
                                              promoter_dict=promoter_dict)

        if len(shared_genes) == 0:
            main_exception('scATAC-seq data and scRNA-seq data do NOT share any genes.')

        # Subset the scATAC-seq data to shared elements
        self.mdata.mod['atac'] = self.mdata.mod['atac'][:, shared_elements].copy()

        # Subset the scRNA_seq data to shared genes
        self.mdata.mod['rna'] = self.mdata.mod['rna'][:, shared_genes].copy()

        # Initialize the original enhancer and promoter dicts
        self._cre_dict = shared_cre_dict
        self._promoter_dict = shared_promoter_dict

    def weighted_nearest_neighbors(
            self,
            atac_lsi_key:        str = MDKM.ATAC_OBSM_LSI_KEY,
            n_components_atac:   int = 20,
            n_components_rna:    int = 20,
            nn:                  int = 20,
            random_state:        int = 42,
            rna_pca_key:         str = MDKM.RNA_OBSM_PC_KEY,
            use_highly_variable: bool = False):
        import scanpy as sc
        main_info('Starting computation of weighted nearest neighbors ...', indent_level=1)
        nn_logger = LoggerManager.gen_logger('weighted_nearest_neighbors')
        nn_logger.log_time()

        # Restrict to shared genes and their elements - as tied together by the attribution of CRE to genes
        shared_elements, shared_genes = self.restrict_to_gene_list(subset=True)

        # Extract scATAC-seq and scRNA-seq data
        atac_adata = self.mdata.mod['atac'][:, shared_elements].copy()
        rna_adata = self.mdata.mod['rna'][:, shared_genes].copy()

        if rna_pca_key not in rna_adata.obsm:
            # TODO: Consider normalizing counts here, if needed

            # Carry out PCA on scRNA-seq data
            main_info('computing PCA on normalized and scaled scRNA-seq data', indent_level=2)
            sc.tl.pca(rna_adata,
                      n_comps=n_components_rna,
                      random_state=random_state,
                      use_highly_variable=use_highly_variable)

        if atac_lsi_key not in atac_adata.obsm:
            # Carry out singular value decomposition on the scATAC-seq data
            main_info('computing latent semantic indexing of scATAC-seq data ...')
            lsi = svds(atac_adata.X, k=n_components_atac)

            # get the lsi result
            atac_adata.obsm[atac_lsi_key] = lsi[0]

        # Cross copy the LSI decomposition
        rna_adata.obsm[atac_lsi_key] = atac_adata.obsm[atac_lsi_key]

        # Use Dylan Kotliar's python implementation of
        # TODO: As alternative to PCA could use the latent space from variational autoencoder.
        WNNobj = pyWNN(rna_adata,
                       reps=[rna_pca_key, atac_lsi_key],
                       npcs=[n_components_rna, n_components_atac],
                       n_neighbors=nn,
                       seed=42)

        adata_seurat = WNNobj.compute_wnn(rna_adata)

        # extract the matrix storing the distances between each cell and its neighbors
        cx = coo_matrix(adata_seurat.obsp["WNN_distance"])

        # the number of cells
        cells = adata_seurat.obsp['WNN_distance'].shape[0]

        # define the shape of our final results
        # and make the arrays that will hold the results
        new_shape = (cells, nn)
        nn_dist = np.zeros(shape=new_shape)
        nn_idx = np.zeros(shape=new_shape)

        # new_col defines what column we store data in
        # our result arrays
        new_col = 0

        # loop through the distance matrices
        for i, j, v in zip(cx.row, cx.col, cx.data):

            # store the distances between neighbor cells
            nn_dist[i][new_col % nn] = v

            # for each cell's row, store the row numbers of its neighbor cells
            # (1-indexing instead of 0- is a holdover from R multimodalneighbors())
            nn_idx[i][new_col % nn] = int(j) + 1

            new_col += 1

        # Add index and distance to the MultiomeVelocity object
        self.nn_idx = nn_idx
        self.nn_dist = nn_dist

        # Revert to canonical naming of connectivities and distances
        # ... .uns['neighbors']
        atac_adata.uns['neighbors'] = adata_seurat.uns['WNN'].copy()
        rna_adata.uns['neighbors'] = adata_seurat.uns['WNN'].copy()
        del adata_seurat.uns['WNN']

        # ... .obsp['connectivities']
        atac_adata.obsp['connectivities'] = adata_seurat.obsp['WNN'].copy()
        rna_adata.obsp['connectivities'] = adata_seurat.obsp['WNN'].copy()
        del adata_seurat.obsp['WNN']

        # ... .obsp['distances']
        atac_adata.obsp['distances'] = adata_seurat.obsp['WNN_distance'].copy()
        rna_adata.obsp['distances'] = adata_seurat.obsp['WNN_distance'].copy()
        del adata_seurat.obsp['WNN_distance']

        # Copy the subset AnnData scRNA-seq and scATAC-seq objects back into the MultiomeVelocity object
        self.mdata.mod['atac'] = atac_adata.copy()
        self.mdata.mod['rna'] = rna_adata.copy()

    def write(self,
              filename: Union[PathLike, str]) -> None:
        export_mdata = self.to_mdata()
        export_mdata.write_h5mu(filename)
