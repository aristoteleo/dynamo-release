from typing import Optional, Union

import anndata
from scipy.sparse import csr_matrix

from ..dynamo_logger import LoggerManager, main_exception, main_warning
from ..utils import copy_adata
from .utils import init_r0_pulse


def get_pulse_r0(
    adata: anndata.AnnData,
    genes: Union[list, str] = "use_for_dynamics",
    tkey: str = "X_total",
    nkey: str = "X_new",
    gamma_k_key: str = "gamma_k",
    add_init_r0_key: str = "init_r0_pulse",
    copy: bool = False,
) -> Union[anndata.AnnData, None]:

    """Get the total RNA at the initial time point for a kinetic experiment with the formula:
           :math:`r_0 = \frac{(r - l)}{(1 - k)}`, where :math: `k = 1 - e^{- \gamma t}

    Parameters
    ----------
        adata:
            an Annodata object
        genes: `list`
            A list of gene names that are going to be visualized.
        tkey:
            the key for normalized total layer in adata.layers.
        nkey:
            the key for normalized new layer in adata.layers.
        gamma_k_key:
            the key for the parameter k for each gene in adata.var.
        add_init_r0_key:
            the key that will be used to store the intial total RNA estimated, in adata.layers.
        copy:
            Whether copy the adata object.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An new or updated anndata object, based on copy parameter, that are updated with Size_Factor, normalized
            expression values, X and reduced dimensions, etc.
    """

    logger = LoggerManager.gen_logger("dynamo-estimate-initial-total-RNA")
    logger.log_time()

    adata = copy_adata(adata) if copy else adata

    if tkey not in adata.layers.keys():
        main_exception("{} is not existed in adata.layers.".format(tkey))
    if nkey not in adata.layers.keys():
        main_exception("{} is not existed in adata.layers.".format(nkey))
    if gamma_k_key not in adata.var.keys():
        main_exception("{} is not existed in adata.var.".format(gamma_k_key))

    if add_init_r0_key in adata.layers.keys():
        main_warning("{} is already existed in adata.layers.".format(add_init_r0_key))

    init_r0_res = csr_matrix(adata.shape)

    if type(genes) is str:
        gene_names = adata.var[genes]
    else:
        gene_names = list(adata.var_names.intersection(genes))
        if len(gene_names) == 0:
            main_exception(
                "The input gene list doesn't match up with any genes in adata.var_names. Please double "
                "check your gene list."
            )

    logger.info(
        "retrieving R (total RNA layer: %s), L (labeled RNA layer: %s), K (parameter K: %s) "
        % (tkey, nkey, gamma_k_key)
    )
    R, L = adata[:, gene_names].layers[tkey], adata[:, gene_names].layers[nkey]
    K = adata[:, gene_names].var[gamma_k_key].values.astype(float)

    logger.info("Calculate initial total RNA via r0 = (r - l) / (1 - k)")
    res = init_r0_pulse(R, L, K[None, :])

    if type(genes) is str:
        gene_indices = gene_names
    else:
        gene_indices = [adata.var_names.get_loc(i) for i in gene_names]

    init_r0_res[:, gene_indices] = csr_matrix(res)

    logger.info_insert_adata(add_init_r0_key, "layers")
    adata.layers[add_init_r0_key] = init_r0_res

    logger.finish_progress(progress_name="dynamo-estimate-initial-total-RNA")

    if copy:
        return adata
    return None
