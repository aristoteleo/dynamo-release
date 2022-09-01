from typing import Union

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.utils import sparsefuncs

from ..dynamo_logger import LoggerManager, main_tqdm
from ..utils import copy_adata


def lambda_correction(
    adata: anndata.AnnData,
    lambda_key: str = "lambda",
    inplace: bool = True,
    copy: bool = False,
) -> Union[anndata.AnnData, None]:
    """Use lambda (cell-wise detection rate) to estimate the labelled RNA.

    Args:
        adata: an adata object generated from dynast.
        lambda_key: the key to the cell-wise detection rate. Defaults to "lambda".
        inplace: whether to inplace update the layers. If False, new layers that append '_corrected" to the existing
            will be used to store the updated data. Defaults to True.
        copy: whether to copy the adata object or update adata object inplace. Defaults to False.

    Raises:
        ValueError: the `lambda_key` cannot be found in `adata.obs`
        ValueError: `data_type` is set to 'splicing_labeling' but the existing layers in the adata object don't meet the
            requirements.
        ValueError: `data_type` is set to 'labeling' but the existing layers in the adata object don't meet the
            requirements.
    Returns:
        A new AnnData object that are updated with lambda corrected layers if `copy` is true. Otherwise, return None.
    """

    logger = LoggerManager.gen_logger("dynamo-lambda_correction")
    logger.log_time()

    adata = copy_adata(adata) if copy else adata

    logger.info("apply detection rate correction to adata...", indent_level=1)

    if lambda_key not in adata.obs.keys():
        raise ValueError(
            f"the lambda_key {lambda_key} is not included in adata.obs! Please ensure you have calculated "
            "per-cell detection rate!"
        )

    logger.info("retrieving the cell-wise detection rate..", indent_level=1)
    detection_rate = adata.obs[lambda_key].values[:, None]

    logger.info("identify the data type..", indent_level=1)
    all_layers = adata.layers.keys()

    has_ul = np.any([i.contains("ul_") for i in all_layers])
    has_un = np.any([i.contains("un_") for i in all_layers])
    has_sl = np.any([i.contains("sl_") for i in all_layers])
    has_sn = np.any([i.contains("sn_") for i in all_layers])

    has_l = np.any([i.contains("_l_") for i in all_layers])
    has_n = np.any([i.contains("_n_") for i in all_layers])

    if sum(has_ul + has_un + has_sl + has_sn) == 4:
        datatype = "splicing_labeling"
    elif sum(has_l + has_n):
        datatype = "labeling"

    logger.info(f"the data type identified is {datatype}", indent_level=2)

    logger.info("retrieve relevant layers for detection rate correction", indent_level=1)
    if datatype == "splicing_labeling":
        layers, match_tot_layer = [], []
        for layer in all_layers:
            if "ul_" in layer:
                layers += layer
                match_tot_layer += "unspliced"
            elif "un_" in layer:
                layers += layer
                match_tot_layer += "unspliced"
            elif "sl_" in layer:
                layers += layer
                match_tot_layer += "spliced"
            elif "sn_" in layer:
                layers += layer
                match_tot_layer += "spliced"
            elif "spliced" in layer:
                layers += layer
            elif "unspliced" in layer:
                layers += layer

            if len(layers) != 6:
                raise ValueError(
                    "the adata object has to include ul, un, sl, sn, unspliced, spliced, "
                    "six relevant layers for splicing and labeling quantified datasets."
                )
    elif datatype == "labeling":
        layers, match_tot_layer = [], []
        for layer in all_layers:
            if "_l_" in layer:
                layers += layer
                match_tot_layer += ["total"]
            elif "_n_" in layer:
                layers += layer
                match_tot_layer += ["total"]
            elif "total" in layer:
                layers += layer

            if len(layers) != 3:
                raise ValueError(
                    "the adata object has to include labeled, unlabeled, three relevant layers for labeling quantified "
                    "datasets."
                )

    logger.info("detection rate correction starts", indent_level=1)
    for i, layer in enumerate(main_tqdm(layers, desc="iterating all relevant layers")):
        if i < len(match_tot_layer):
            cur_layer = adata.layers[layer] if inplace else adata.layers[layer].copy()
            cur_total = adata.layers[match_tot_layer[i]]

            # even layers is labeled RNA and odd unlabeled RNA
            if i % 2 == 0:
                # formula: min(L / lambda, (L + U)) from scNT-seq
                if issparse(cur_layer):
                    sparsefuncs.inplace_row_scale(cur_layer, 1 / detection_rate)
                else:
                    cur_layer /= detection_rate
                if inplace:
                    adata.layers[layer] = sparse_mimmax(cur_layer, cur_total)
                else:
                    adata.layers[layer + "_corrected"] = sparse_mimmax(cur_layer, cur_total)

            else:
                if inplace:
                    adata.layers[layer] = cur_total - adata.layers[layer[i - 1]]
                else:
                    adata.layers[layer + "_corrected"] = cur_total - adata.layers[layer[i - 1]]

    logger.finish_progress(progress_name="lambda_correction")

    if copy:
        return adata
    return None


def sparse_mimmax(A: csr_matrix, B: csr_matrix, type="min") -> csr_matrix:
    """Return the element-wise minimum/maximum of sparse matrices `A` and `B`.

    Args:
        A: The first sparse matrix
        B: The second sparse matrix
        type: The type of calculation, either "min" or "max". Defaults to "min".

    Returns:
        A sparse matrix that contain the element-wise maximal or minimal of two sparse matrices.
    """

    AgtB = (A < B).astype(int) if type == "min" else (A > B).astype(int)
    M = AgtB.multiply(A - B) + B

    return M
