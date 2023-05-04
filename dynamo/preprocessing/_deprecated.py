import re
from typing import List, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from ..configuration import DKM
from ..dynamo_logger import LoggerManager, main_debug, main_warning
from .utils import cook_dist


def _disp_calc_helper_NB(
    adata: AnnData, layers: str = "X", min_cells_detected: int = 1
) -> Tuple[List[str], List[pd.DataFrame]]:
    """Helper function to calculate the dispersion parameter.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an Anndata object.
        layers: the layer of data used for dispersion fitting. Defaults to "X".
        min_cells_detected: the minimal required number of cells with expression for selecting gene for dispersion
            fitting. Defaults to 1.

    Returns:
        layers: a list of layers available.
        res_list: a list of pd.DataFrames with mu, dispersion for each gene that passes filters.
    """
    main_warning(__name__ + " is deprecated.")
    layers = DKM.get_available_layer_keys(adata, layers=layers, include_protein=False)

    res_list = []
    for layer in layers:
        if layer == "raw":
            CM = adata.raw.X
            szfactors = adata.obs[layer + "Size_Factor"][:, None]
        elif layer == "X":
            CM = adata.X
            szfactors = adata.obs["Size_Factor"][:, None]
        else:
            CM = adata.layers[layer]
            szfactors = adata.obs[layer + "Size_Factor"][:, None]

        if issparse(CM):
            CM.data = np.round(CM.data, 0)
            rounded = CM
        else:
            rounded = CM.round().astype("int")

        lowerDetectedLimit = adata.uns["lowerDetectedLimit"] if "lowerDetectedLimit" in adata.uns.keys() else 1
        nzGenes = (rounded > lowerDetectedLimit).sum(axis=0)
        nzGenes = nzGenes > min_cells_detected

        nzGenes = nzGenes.A1 if issparse(rounded) else nzGenes
        if layer.startswith("X_"):
            x = rounded[:, nzGenes]
        else:
            x = (
                rounded[:, nzGenes].multiply(csr_matrix(1 / szfactors))
                if issparse(rounded)
                else rounded[:, nzGenes] / szfactors
            )

        xim = np.mean(1 / szfactors) if szfactors is not None else 1

        f_expression_mean = x.mean(axis=0)

        # For NB: Var(Y) = mu * (1 + mu / k)
        # x.A.var(axis=0, ddof=1)
        f_expression_var = (
            (x.multiply(x).mean(0).A1 - f_expression_mean.A1**2) * x.shape[0] / (x.shape[0] - 1)
            if issparse(x)
            else x.var(axis=0, ddof=0) ** 2
        )  # np.mean(np.power(x - f_expression_mean, 2), axis=0) # variance with n - 1
        # https://scialert.net/fulltext/?doi=ajms.2010.1.15 method of moments
        disp_guess_meth_moments = f_expression_var - xim * f_expression_mean  # variance - mu

        disp_guess_meth_moments = disp_guess_meth_moments / np.power(
            f_expression_mean, 2
        )  # this is dispersion parameter (1/k)

        res = pd.DataFrame(
            {
                "mu": np.array(f_expression_mean).flatten(),
                "disp": np.array(disp_guess_meth_moments).flatten(),
            }
        )
        res.loc[res["mu"] == 0, "mu"] = None
        res.loc[res["mu"] == 0, "disp"] = None
        res.loc[res["disp"] < 0, "disp"] = 0

        res["gene_id"] = adata.var_names[nzGenes]

        res_list.append(res)

    return layers, res_list


def _parametric_dispersion_fit(
    disp_table: pd.DataFrame, initial_coefs: np.ndarray = np.array([1e-6, 1])
) -> Tuple[sm.formula.glm, np.ndarray, pd.DataFrame]:
    """Perform the dispersion parameter fitting with initial guesses of coefficients.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        disp_table: A pandas dataframe with mu, dispersion for each gene that passes filters.
        initial_coefs: Initial parameters for the gamma fit of the dispersion parameters. Defaults to
            np.array([1e-6, 1]).

    Returns:
        A tuple (fit, coefs, good), where fit is a statsmodels fitting object, coefs contains the two resulting gamma
        fitting coefficient, and good is the subsetted dispersion table that is subjected to Gamma fitting.
    """
    main_warning(__name__ + " is deprecated.")
    coefs = initial_coefs
    iter = 0
    while True:
        residuals = disp_table["disp"] / (coefs[0] + coefs[1] / disp_table["mu"])
        good = disp_table.loc[(residuals > initial_coefs[0]) & (residuals < 10000), :]
        # https://stats.stackexchange.com/questions/356053/the-identity-link-function-does-not-respect-the-domain-of-the
        # -gamma-family
        fit = sm.formula.glm(
            "disp ~ I(1 / mu)",
            data=good,
            family=sm.families.Gamma(link=sm.genmod.families.links.identity),
        ).train(start_params=coefs)

        oldcoefs = coefs
        coefs = fit.params

        if coefs[0] < initial_coefs[0]:
            coefs[0] = initial_coefs[0]
        if coefs[1] < 0:
            main_warning("Parametric dispersion fit may be failed.")

        if np.sum(np.log(coefs / oldcoefs) ** 2 < coefs[0]):
            break
        iter += 1

        if iter > 10:
            main_warning("Dispersion fit didn't converge")
            break
        if not np.all(coefs > 0):
            main_warning("Parametric dispersion fit may be failed.")

    return fit, coefs, good


def _estimate_dispersion(
    adata: AnnData,
    layers: str = "X",
    modelFormulaStr: str = "~ 1",
    min_cells_detected: int = 1,
    removeOutliers: bool = False,
) -> AnnData:
    """This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Args:
        adata: an AnnData object.
        layers: the layer(s) to be used for calculating dispersion. Default is "X" if there is no spliced layers.
        modelFormulaStr: the model formula used to calculate dispersion parameters. Not used. Defaults to "~ 1".
        min_cells_detected: the minimum number of cells detected for calculating the dispersion. Defaults to 1.
        removeOutliers: whether to remove outliers when performing dispersion fitting. Defaults to False.

    Raises:
        Exception: there is no valid DataFrames with mu for genes.

    Returns:
        An updated annData object with dispFitInfo added to uns attribute as a new key.
    """
    main_warning(__name__ + " is deprecated.")
    logger = LoggerManager.gen_logger("dynamo-preprocessing")
    # mu = None
    model_terms = [x.strip() for x in re.compile("~|\\*|\\+").split(modelFormulaStr)]
    model_terms = list(set(model_terms) - set([""]))

    cds_pdata = adata.obs  # .loc[:, model_terms]
    cds_pdata["rowname"] = cds_pdata.index.values
    layers, disp_tables = _disp_calc_helper_NB(adata[:, :], layers, min_cells_detected)
    # disp_table['disp'] = np.random.uniform(0, 10, 11)
    # disp_table = cds_pdata.apply(disp_calc_helper_NB(adata[:, :], min_cells_detected))

    # cds_pdata <- dplyr::group_by_(dplyr::select_(rownames_to_column(pData(cds)), "rowname", .dots=model_terms), .dots
    # =model_terms)
    # disp_table <- as.data.frame(cds_pdata %>% do(disp_calc_helper_NB(cds[,.$rowname], cds@expressionFamily, min_cells_
    # detected)))
    for ind in range(len(layers)):
        layer, disp_table = layers[ind], disp_tables[ind]

        if disp_table is None:
            raise Exception("Parametric dispersion fitting failed, please set a different lowerDetectionLimit")

        disp_table = disp_table.loc[np.where(disp_table["mu"] != np.nan)[0], :]

        res = _parametric_dispersion_fit(disp_table)
        fit, coefs, good = res[0], res[1], res[2]

        if removeOutliers:
            # influence = fit.get_influence().cooks_distance()
            # #CD is the distance and p is p-value
            # (CD, p) = influence.cooks_distance

            CD = cook_dist(fit, 1 / good["mu"][:, None], good)
            cooksCutoff = 4 / good.shape[0]
            main_debug("Removing " + str(len(CD[CD > cooksCutoff])) + " outliers")
            outliers = CD > cooksCutoff
            # use CD.index.values? remove genes that lost when doing parameter fitting
            lost_gene = set(good.index.values).difference(set(range(len(CD))))
            outliers[lost_gene] = True
            res = _parametric_dispersion_fit(good.loc[~outliers, :])

            fit, coefs = res[0], res[1]

        def ans(q):
            return coefs[0] + coefs[1] / q

        if layer == "X":
            logger.info_insert_adata("dispFitInfo", "uns")
            adata.uns["dispFitInfo"] = {
                "disp_table": good,
                "disp_func": ans,
                "coefs": coefs,
            }
        else:
            logger.info_insert_adata(layer + "_dispFitInfo", "uns")
            adata.uns[layer + "_dispFitInfo"] = {
                "disp_table": good,
                "disp_func": ans,
                "coefs": coefs,
            }

    return adata


def _top_table(adata: AnnData, layer: str = "X", mode: Literal["dispersion", "gini"] = "dispersion") -> pd.DataFrame:
    """Retrieve a table that contains gene names and other info whose dispersions/gini index are highest.

    This function is partly based on Monocle R package (https://github.com/cole-trapnell-lab/monocle3).

    Get information of the top layer.

    Args:
        adata: an AnnData object.
        layer: the layer(s) that would be searched for. Defaults to "X".
        mode: either "dispersion" or "gini", deciding whether dispersion data or gini data would be acquired. Defaults
            to "dispersion".

    Raises:
        KeyError: if mode is set to dispersion but there is no available dispersion model.

    Returns:
        The data frame of the top layer with the gene_id, mean_expression, dispersion_fit and dispersion_empirical as
        the columns.
    """

    layer = DKM.get_available_layer_keys(adata, layers=layer, include_protein=False)[0]

    if layer in ["X"]:
        key = "dispFitInfo"
    else:
        key = layer + "_dispFitInfo"

    if mode == "dispersion":
        if adata.uns[key] is None:
            _estimate_dispersion(adata, layers=[layer])
            raise KeyError(
                "Error: for adata.uns.key=%s, no dispersion model found. Please call estimate_dispersion() before calling this function"
                % key
            )

        top_df = pd.DataFrame(
            {
                "gene_id": adata.uns[key]["disp_table"]["gene_id"],
                "mean_expression": adata.uns[key]["disp_table"]["mu"],
                "dispersion_fit": adata.uns[key]["disp_func"](adata.uns[key]["disp_table"]["mu"]),
                "dispersion_empirical": adata.uns[key]["disp_table"]["disp"],
            }
        )
        top_df = top_df.set_index("gene_id")

    elif mode == "gini":
        top_df = adata.var[layer + "_gini"]

    return top_df
