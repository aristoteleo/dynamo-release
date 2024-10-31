from typing import Any, Dict, Optional

import numpy as np
from anndata import AnnData

from ..configuration import DynamoAdataConfig
from ..preprocessing.pca import pca
from ..preprocessing.utils import (
    del_raw_layers,
    detect_experiment_datatype,
    reset_adata_X,
)
from .cell_velocities import cell_velocities
from .connectivity import neighbors, normalize_knn_graph
from .dimension_reduction import reduceDimension
from .dynamics import dynamics
from .moments import moments
from .utils import get_vel_params, set_transition_genes, update_vel_params

# add recipe_csc_data()


# support using just spliced/unspliced/new/total 4 layers, as well as uu, ul, su, sl layers
def recipe_one_shot_data(
    adata: AnnData,
    tkey: Optional[str] = None,
    reset_X: bool = True,
    X_total_layers: bool = False,
    splicing_total_layers: bool = False,
    n_top_genes: int = 1000,
    keep_filtered_cells: Optional[bool] = None,
    keep_filtered_genes: Optional[bool] = None,
    keep_raw_layers: Optional[bool] = None,
    one_shot_method: str = "sci-fate",
    del_2nd_moments: Optional[bool] = None,
    ekey: str = "M_t",
    vkey: str = "velocity_T",
    basis: str = "umap",
    rm_kwargs: Dict[str, Any] = {},
) -> AnnData:
    """An analysis recipe that properly pre-processes different layers for a one-shot experiment with both labeling and
    splicing data.

    Args:
        adata: AnnData object that stores data for the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to `adata.uns['pp']['tkey']` and
            used in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2nd
            moment or covariance. We recommend to use hour as the unit of `time`. Defaults to None.
        reset_X: Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure. Defaults to True.
        X_total_layers: Whether to also normalize adata.X by size factor from total RNA. Parameter to `recipe_monocle`
            function. Defaults to False.
        splicing_total_layers: Whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Parameter to `recipe_monocle` function. Defaults to False.
        n_top_genes: The number of top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Arguments required by the `recipe_monocle` function. Defaults to 1000.
        keep_filtered_cells: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_filtered_genes: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_raw_layers: Whether to keep layers with raw measurements in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        one_shot_method: The method to use for calculate the absolute labeling and splicing velocity for the one-shot
            data of use. Defaults to "sci-fate".
        del_2nd_moments: Whether to remove second moments or covariances. Argument used for `dynamics` function. If
            None, would be set according to `DynamoAdataConfig`. Defaults to None.
        ekey: The dictionary key that corresponds to the gene expression in the layer attribute. ekey and vkey will be
            automatically detected from the adata object. Parameters required by `cell_velocities`. Defaults to "M_t".
        vkey: The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities` Defaults to "velocity_T".
        basis: The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be
            `X_spliced_umap` or `X_total_umap`, etc. Parameters required by `cell_velocities`. Defaults to "umap".
        rm_kwargs: Other kwargs passed into the pp.recipe_monocle function. Defaults to {}.

    Raises:
        Exception: the recipe is only applicable to kinetics experiment datasets with labeling data.

    Returns:
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """

    from ..preprocessing import Preprocessor

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )
    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    # Preprocessing
    preprocessor = Preprocessor()
    preprocessor.config_monocle_recipe(adata, n_top_genes=n_top_genes)
    preprocessor.size_factor_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
        }
    )
    preprocessor.normalize_by_cells_function_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
            "keep_filtered": keep_filtered_genes,
            "total_szfactor": "total_Size_Factor",
        }
    )
    preprocessor.filter_cells_by_outliers_kwargs["keep_filtered"] = keep_filtered_cells
    preprocessor.select_genes_kwargs["keep_filtered"] = keep_filtered_genes

    if reset_X:
        reset_adata_X(adata, experiment_type="one-shot", has_labeling=has_labeling, has_splicing=has_splicing)
    preprocessor.preprocess_adata_monocle(adata=adata, tkey=tkey, experiment_type="one-shot")
    if not keep_raw_layers:
        del_raw_layers(adata)

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.
        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].toarray())
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(
            adata,
            model="deterministic",
            one_shot_method=one_shot_method,
            del_2nd_moments=del_2nd_moments,
        )
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
    else:
        dynamics(
            adata,
            model="deterministic",
            one_shot_method=one_shot_method,
            del_2nd_moments=del_2nd_moments,
        )
        reduceDimension(adata, reduction_method=basis)
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)

    return adata


def recipe_kin_data(
    adata: AnnData,
    tkey: Optional[str] = None,
    reset_X: bool = True,
    X_total_layers: bool = False,
    splicing_total_layers: bool = False,
    n_top_genes: int = 1000,
    keep_filtered_cells: Optional[bool] = None,
    keep_filtered_genes: Optional[bool] = None,
    keep_raw_layers: Optional[bool] = None,
    del_2nd_moments: Optional[bool] = None,
    ekey: str = "M_t",
    vkey: str = "velocity_T",
    basis: str = "umap",
    rm_kwargs: Dict["str", Any] = {},
) -> AnnData:
    """An analysis recipe that properly pre-processes different layers for an kinetics experiment with both labeling and
    splicing or only labeling data.

    Args:
        adata: An AnnData object that stores data for the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`. Defaults to None.
        reset_X: Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure. Defaults to True.
        X_total_layers: Whether to also normalize adata.X by size factor from total RNA. Parameter to `recipe_monocle`
            function. Defaults to False.
        splicing_total_layers: Whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Parameter to `recipe_monocle` function. Defaults to False.
        n_top_genes: The number of top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Arguments required by the `recipe_monocle` function. Defaults to 1000.
        keep_filtered_cells: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_filtered_genes: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_raw_layers: Whether to keep layers with raw measurements in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        del_2nd_moments: Whether to remove second moments or covariances. Argument used for `dynamics` function. If
            None, would be set according to `DynamoAdataConfig`. Defaults to None.
        ekey: The dictionary key that corresponds to the gene expression in the layer attribute. ekey and vkey will be
            automatically detected from the adata object. Parameters required by `cell_velocities`. Defaults to "M_t".
        vkey: The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities` Defaults to "velocity_T".
        basis: The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be
            `X_spliced_umap` or `X_total_umap`, etc. Parameters required by `cell_velocities`. Defaults to "umap".
        rm_kwargs: Other kwargs passed into the pp.recipe_monocle function. Defaults to {}.

    Raises:
        Exception: The recipe is only applicable to kinetics experiment datasets with labeling data.

    Returns:
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """

    from ..preprocessing import Preprocessor

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )
    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    # Preprocessing
    preprocessor = Preprocessor(cell_cycle_score_enable=True)
    preprocessor.config_monocle_recipe(adata, n_top_genes=n_top_genes)
    preprocessor.size_factor_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
        }
    )
    preprocessor.normalize_by_cells_function_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
            "keep_filtered": keep_filtered_genes,
            "total_szfactor": "total_Size_Factor",
        }
    )
    preprocessor.filter_cells_by_outliers_kwargs["keep_filtered"] = keep_filtered_cells
    preprocessor.select_genes_kwargs["keep_filtered"] = keep_filtered_genes

    if reset_X:
        reset_adata_X(adata, experiment_type="kin", has_labeling=has_labeling, has_splicing=has_splicing)
    preprocessor.preprocess_adata_monocle(adata=adata, tkey=tkey, experiment_type="kin")
    if not keep_raw_layers:
        del_raw_layers(adata)

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.

        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].toarray())
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(adata, model="deterministic", est_method="twostep", del_2nd_moments=del_2nd_moments)
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
    else:
        dynamics(adata, model="deterministic", est_method="twostep", del_2nd_moments=del_2nd_moments)
        reduceDimension(adata, reduction_method=basis)
        cell_velocities(adata, basis=basis)

    return adata


def recipe_deg_data(
    adata: AnnData,
    tkey: Optional[str] = None,
    reset_X: bool = True,
    X_total_layers: bool = False,
    splicing_total_layers: bool = False,
    n_top_genes: int = 1000,
    keep_filtered_cells: Optional[bool] = None,
    keep_filtered_genes: Optional[bool] = None,
    keep_raw_layers: Optional[bool] = None,
    del_2nd_moments: Optional[bool] = True,
    fraction_for_deg: bool = False,
    ekey: str = "M_s",
    vkey: str = "velocity_S",
    basis: str = "umap",
    rm_kwargs: Dict[str, Any] = {},
):
    """An analysis recipe that properly pre-processes different layers for a degradation experiment with both
    labeling and splicing data or only labeling. Functions need to be updated.

    Args:
        adata: An AnnData object that stores data for the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to adata.uns['pp']['tkey'] and used
            in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2st moment
            or covariance. We recommend to use hour as the unit of `time`. Defaults to None.
        reset_X: Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure. Defaults to True.
        X_total_layers: Whether to also normalize adata.X by size factor from total RNA. Parameter to `recipe_monocle`
            function. Defaults to False.
        splicing_total_layers: Whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Parameter to `recipe_monocle` function. Defaults to False.
        n_top_genes: The number of top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Arguments required by the `recipe_monocle` function. Defaults to 1000.
        keep_filtered_cells: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_filtered_genes: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_raw_layers: Whether to keep layers with raw measurements in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        del_2nd_moments: Whether to remove second moments or covariances. Argument used for `dynamics` function. If
            None, would be set according to `DynamoAdataConfig`. Defaults to None.
        fraction_for_deg: Whether to use the fraction of labeled RNA instead of the raw labeled RNA to estimate the
            degradation parameter. Defaults to False.
        ekey: The dictionary key that corresponds to the gene expression in the layer attribute. ekey and vkey will be
            automatically detected from the adata object. Parameters required by `cell_velocities`. Defaults to "M_s".
        vkey: The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities` Defaults to "velocity_S".
        basis: The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be
            `X_spliced_umap` or `X_total_umap`, etc. Parameters required by `cell_velocities`. Defaults to "umap".
        rm_kwargs: Other kwargs passed into the pp.recipe_monocle function. Defaults to {}.

    Raises:
        Exception: The recipe is only applicable to kinetics experiment datasets with labeling data.

    Returns:
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """

    from ..preprocessing import Preprocessor

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    preprocessor = Preprocessor()
    preprocessor.config_monocle_recipe(adata, n_top_genes=n_top_genes)
    preprocessor.size_factor_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
        }
    )
    preprocessor.normalize_by_cells_function_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
            "keep_filtered": keep_filtered_genes,
            "total_szfactor": "total_Size_Factor",
        }
    )
    preprocessor.filter_cells_by_outliers_kwargs["keep_filtered"] = keep_filtered_cells
    preprocessor.select_genes_kwargs["keep_filtered"] = keep_filtered_genes

    if reset_X:
        reset_adata_X(adata, experiment_type="deg", has_labeling=has_labeling, has_splicing=has_splicing)
    preprocessor.preprocess_adata_monocle(adata=adata, tkey=tkey, experiment_type="deg")
    if not keep_raw_layers:
        del_raw_layers(adata)

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.
        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, layers=["X_spliced", "X_unspliced"])

        # then calculate moments for labeling data relevant layers using total based connectivity graph
        # first get X_total based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_total"].toarray())
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()
        pca(adata, CM[:, valid_ind], pca_key="X_total_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_total_pca"], layer="X_total")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        moments(adata, conn=conn, group=tkey, layers=layers)

        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
            fraction_for_deg=fraction_for_deg,
        )
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        try:
            set_transition_genes(adata)
            cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
        except BaseException:
            vel_params_df = get_vel_params(adata)
            cell_velocities(
                adata,
                min_r2=vel_params_df.gamma_r2.min(),
                enforce=True,
                vkey=vkey,
                ekey=ekey,
                basis=basis,
            )

    else:
        dynamics(adata, model="deterministic", del_2nd_moments=del_2nd_moments, fraction_for_deg=fraction_for_deg)
        reduceDimension(adata, reduction_method=basis)

    return adata


def recipe_mix_kin_deg_data(
    adata: AnnData,
    tkey: Optional[str] = None,
    reset_X: bool = True,
    X_total_layers: bool = False,
    splicing_total_layers: bool = False,
    n_top_genes: int = 1000,
    keep_filtered_cells: Optional[bool] = None,
    keep_filtered_genes: Optional[bool] = None,
    keep_raw_layers: Optional[bool] = None,
    del_2nd_moments: Optional[bool] = None,
    ekey: str = "M_t",
    vkey: str = "velocity_T",
    basis: str = "umap",
    rm_kwargs: Dict[str, Any] = {},
):
    """An analysis recipe that properly pre-processes different layers for a mixture kinetics and degradation
    experiment with both labeling and splicing or only labeling data.

    Args:
        adata: An AnnData object that stores data for the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        tkey: The column key for the labeling time  of cells in .obs. Used for labeling based scRNA-seq data (will also
            support for conventional scRNA-seq data). Note that `tkey` will be saved to `adata.uns['pp']['tkey']` and
            used in `dyn.tl.dynamics` in which when `group` is None, `tkey` will also be used for calculating  1st/2nd
            moment or covariance. We recommend to use hour as the unit of `time`. Defaults to None.
        reset_X: Whether do you want to let dynamo reset `adata.X` data based on layers stored in your experiment. One
            critical functionality of dynamo is about visualizing RNA velocity vector flows which requires proper data
            into which the high dimensional RNA velocity vectors will be projected.
            (1) For `kinetics` experiment, we recommend the use of `total` layer as `adata.X`;
            (2) For `degradation/conventional` experiment scRNA-seq, we recommend using `splicing` layer as `adata.X`.
            Set `reset_X` to `True` to set those default values if you are not sure. Defaults to True.
        X_total_layers: Whether to also normalize adata.X by size factor from total RNA. Parameter to `recipe_monocle`
            function. Defaults to False.
        splicing_total_layers: Whether to also normalize spliced / unspliced layers by size factor from total RNA.
            Parameter to `recipe_monocle` function. Defaults to False.
        n_top_genes: The number of top genes based on scoring method (specified by sort_by) will be selected as feature
            genes. Arguments required by the `recipe_monocle` function. Defaults to 1000.
        keep_filtered_cells: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_filtered_genes: Whether to keep genes that don't pass the filtering in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        keep_raw_layers: Whether to keep layers with raw measurements in the returned adata object. Used in
            `recipe_monocle`. If None, would be set according to `DynamoAdataConfig`. Defaults to None.
        del_2nd_moments: Whether to remove second moments or covariances. Argument used for `dynamics` function. If
            None, would be set according to `DynamoAdataConfig`. Defaults to None.
        ekey: The dictionary key that corresponds to the gene expression in the layer attribute. ekey and vkey will be
            automatically detected from the adata object. Parameters required by `cell_velocities`. Defaults to "M_t".
        vkey: The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities` Defaults to "velocity_T".
        basis: The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be
            `X_spliced_umap` or `X_total_umap`, etc. Parameters required by `cell_velocities`. Defaults to "umap".
        rm_kwargs: Other kwargs passed into the pp.recipe_monocle function. Defaults to {}.

    Raises:
        Exception: the recipe is only applicable to kinetics experiment datasets with labeling data.

    Returns:
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """

    from ..preprocessing import Preprocessor

    keep_filtered_cells = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_cells, DynamoAdataConfig.RECIPE_KEEP_FILTERED_CELLS_KEY
    )
    keep_filtered_genes = DynamoAdataConfig.use_default_var_if_none(
        keep_filtered_genes, DynamoAdataConfig.RECIPE_KEEP_FILTERED_GENES_KEY
    )
    keep_raw_layers = DynamoAdataConfig.use_default_var_if_none(
        keep_raw_layers, DynamoAdataConfig.RECIPE_KEEP_RAW_LAYERS_KEY
    )
    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    # Preprocessing
    preprocessor = Preprocessor()
    preprocessor.config_monocle_recipe(adata, n_top_genes=n_top_genes)
    preprocessor.size_factor_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
        }
    )
    preprocessor.normalize_by_cells_function_kwargs.update(
        {
            "X_total_layers": X_total_layers,
            "splicing_total_layers": splicing_total_layers,
            "keep_filtered": keep_filtered_genes,
            "total_szfactor": "total_Size_Factor",
        }
    )
    preprocessor.filter_cells_by_outliers_kwargs["keep_filtered"] = keep_filtered_cells
    preprocessor.select_genes_kwargs["keep_filtered"] = keep_filtered_genes

    if reset_X:
        reset_adata_X(adata, experiment_type="mix_pulse_chase", has_labeling=has_labeling, has_splicing=has_splicing)
    preprocessor.preprocess_adata_monocle(adata=adata, tkey=tkey, experiment_type="mix_pulse_chase")
    if not keep_raw_layers:
        del_raw_layers(adata)

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.
        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].toarray())
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing
        # data
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
        )
        # then perform dimension reduction
        reduceDimension(adata, reduction_method=basis)
        # lastly, project RNA velocity to low dimensional embedding.
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)
    else:
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            del_2nd_moments=del_2nd_moments,
        )
        reduceDimension(adata, reduction_method=basis)
        cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)

    return adata


def velocity_N(
    adata: AnnData,
    group: Optional[str] = None,
    recalculate_pca: bool = True,
    recalculate_umap: bool = True,
    del_2nd_moments: Optional[bool] = None,
) -> None:
    """Use new RNA based pca, umap, for velocity calculation and projection for kinetics or one-shot experiment.

    The AnnData object will be updated inplace with the low dimensional (umap or pca) velocity projections with the new
    RNA or pca based RNA velocities.

    Note that currently velocity_N function only considers labeling data and removes splicing data if they exist.

    Args:
        adata: AnnData object that stores data for the kinetics or one-shot experiment, must include `X_new`, `X_total`
            layers.
        group: The cell group that will be used to calculate velocity in each separate group. This is useful if your
            data comes from different labeling condition, etc. Defaults to None.
        recalculate_pca: Whether to recalculate pca with the new RNA data. If setting to be False, you need to make sure
            the pca is already generated via new RNA. Defaults to True.
        recalculate_umap: Whether to recalculate umap with the new RNA data. If setting to be False, you need to make
            sure the umap is already generated via new RNA. Defaults to True.
        del_2nd_moments: Whether to remove second moments or covariances. If None, would be set according to
            `DynamoAdataConfig`. Defaults to None.

    Raises:
        Exception: `X_new` or `X_total` layer unavailable.
        Exception: experiment type is not supported.
    """

    del_2nd_moments = DynamoAdataConfig.use_default_var_if_none(
        del_2nd_moments, DynamoAdataConfig.RECIPE_DEL_2ND_MOMENTS_KEY
    )

    var_columns = adata.var.columns
    layer_keys = adata.layers.keys()
    vel_params_df = get_vel_params(adata)

    # check velocity_N, velocity_T, X_new, X_total
    if not np.all([i in layer_keys for i in ["X_new", "X_total"]]):
        raise Exception(f"The `X_new`, `X_total` has to exist in your data before running velocity_N function.")

    # delete the moments and velocities that generated via total RNA
    for i in ["M_t", "M_tt", "M_n", "M_tn", "M_nn", "velocity_N", "velocity_T"]:
        if i in layer_keys:
            del adata.layers[i]

    # delete the kinetic paraemters that generated via total RNA
    for i in [
        "alpha",
        "beta",
        "gamma",
        "half_life",
        "alpha_b",
        "alpha_r2",
        "gamma_b",
        "gamma_r2",
        "gamma_logLL",
        "delta_b",
        "delta_r2",
        "bs",
        "bf",
        "uu0",
        "ul0",
        "su0",
        "sl0",
        "U0",
        "S0",
        "total0",
        "beta_k",
        "gamma_k",
    ]:
        if i in vel_params_df.columns:
            del vel_params_df[i]

    if group is not None:
        group_prefixes = [group + "_" + str(i) + "_" for i in adata.obs[group].unique()]
        for i in group_prefixes:
            for j in [
                "alpha",
                "beta",
                "gamma",
                "half_life",
                "alpha_b",
                "alpha_r2",
                "gamma_b",
                "gamma_r2",
                "gamma_logLL",
                "delta_b",
                "delta_r2",
                "bs",
                "bf",
                "uu0",
                "ul0",
                "su0",
                "sl0",
                "U0",
                "S0",
                "total0",
                "beta_k",
                "gamma_k",
            ]:
                if i + j in vel_params_df.columns:
                    del vel_params_df[i + j]
    update_vel_params(adata, params_df=vel_params_df)

    # now let us first run pca with new RNA
    if recalculate_pca:
        pca(adata, np.log1p(adata[:, adata.var.use_for_pca].layers["X_new"]), pca_key="X_pca")

    # if there are unspliced / spliced data, delete them for now:
    for i in ["spliced", "unspliced", "X_spliced", "X_unspliced"]:
        if i in layer_keys:
            del adata.layers[i]

    # now redo the RNA velocity analysis with moments generated with pca space of new RNA
    # let us also check whether it is a one-shot or kinetics experiment
    if adata.uns["pp"]["experiment_type"] == "one-shot":
        dynamics(
            adata,
            one_shot_method="sci_fate",
            model="deterministic",
            group=group,
            del_2nd_moments=del_2nd_moments,
        )
    elif adata.uns["pp"]["experiment_type"] == "kin":
        dynamics(
            adata,
            model="deterministic",
            est_method="twostep",
            group=group,
            del_2nd_moments=del_2nd_moments,
        )
    else:
        raise Exception(
            f"velocity_N function only supports either the one-shot or kinetics (kin) metabolic labeling "
            f"experiment."
        )

    # umap based on new RNA
    if recalculate_umap:
        reduceDimension(adata, enforce=True)

    # project new RNA velocity to new RNA pca
    cell_velocities(
        adata,
        basis="pca",
        X=adata.layers["M_n"],
        V=adata.layers["velocity_N"],
        enforce=True,
    )

    # project new RNA velocity to new RNA umap
    cell_velocities(
        adata,
        basis="umap",
        X=adata.layers["M_n"],
        V=adata.layers["velocity_N"],
        enforce=True,
    )
