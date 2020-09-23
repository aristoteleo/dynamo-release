import numpy as np
from .moments import moments
from .connectivity import neighbors, normalize_knn_graph
from .dynamics import dynamics
from .dimension_reduction import reduceDimension
from .cell_velocities import cell_velocities


def recipe_splicing_labeling_kinetics_data(adata,
                                           n_top_genes=1000,
                                           tkey='time',
                                           ekey='M_t',
                                           vkey='velocity_T',
                                           basis='umap',
                                           ):
    """An analysis recipe that properly pre-processes different layers for an kinetics experiment with both labeling and
    splicing data.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that stores data for the the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        n_top_genes: `int` (default: `1000`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
            Arguments required by the `recipe_monocle` function.
        tkey: `str` (default: `time`)
            The column key for the time label of cells in .obs. Used for  the "kinetic" model.
            mode  with labeled data. When `group` is None, `tkey` will also be used for calculating  1st/2st moment or
            covariance. `{tkey}` column must exist in your adata object and indicates the labeling time period.
            Parameters required for `dynamics` function.
        ekey: str or None (optional, default None)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object. Parameters required by `cell_velocities`.
        vkey: str or None (optional, default None)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities`
        basis: int (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be `X_spliced_umap`
            or `X_total_umap`, etc. Parameters required by `cell_velocities`

    Returns
    -------
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """
    from ..preprocessing import recipe_monocle
    from ..preprocessing import pca

    if not (all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]) or
            all([i in adata.layers.keys() for i in ['new', 'total', 'spliced', 'unspliced']])):
        raise Exception(f"this recipe is only applicable to kinetics experiment dataset that have "
                        f"`'uu', 'ul', 'su', 'sl'` four layers.")

    # new, total, uu, ul, su, sl layers will be normalized with size factor calculated with total layers
    # spliced / unspliced layers will be normalized independently.
    recipe_monocle(adata, n_top_genes=n_top_genes, total_layers=True)

    # first calculate moments for labeling data relevant layers using total based connectivity graph
    moments(adata, group=tkey, layers=['X_new', 'X_total', 'X_uu', 'X_ul', 'X_su', 'X_sl'])

    # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced data.
    # first get X_spliced based pca embedding
    CM = np.log1p(adata[:, adata.var.use_for_pca].layers['X_spliced'].A)
    cm_genesums = CM.sum(axis=0)
    valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
    valid_ind = np.array(valid_ind).flatten()

    pca(adata, CM[:, valid_ind], pca_key='X_spliced_pca')
    # then get neighbors graph based on X_spliced_pca
    neighbors(adata, X_data=adata.obsm['X_spliced_pca'], layer='X_spliced')
    # then normalize neighbors graph so that each row sums up to be 1
    conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
    # then calculate moments for spliced related layers using spliced based connectivity graph
    moments(adata, group='time', conn=conn, layers=['X_spliced', 'X_unspliced'])
    # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing data
    dynamics(adata, model='kinetic', tkey='time', est_method='twostep')  # no correction
    # then perform dimension reduction
    reduceDimension(adata, reduction_method='umap')
    # lastly, project RNA velocity to low dimensional embedding.
    cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)

    return adata


def recipe_splicing_labeling_degradation_data(adata,
                                              n_top_genes=1000,
                                              tkey='time',
                                              ekey='M_s',
                                              vkey='velocity_S',
                                              basis='umap',
                                              ):
    """An analysis recipe that properly pre-processes different layers for an degradatation experiment with both
    labeling and splicing data. Functions need to be updated.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that stores data for the the kinetics experiment, must include `uu, ul, su, sl` four
            different layers.
        n_top_genes: `int` (default: `1000`)
            How many top genes based on scoring method (specified by sort_by) will be selected as feature genes.
            Arguments required by the `recipe_monocle` function.
        tkey: `str` (default: `time`)
            The column key for the time label of cells in .obs. Used for  the "kinetic" model.
            mode  with labeled data. When `group` is None, `tkey` will also be used for calculating  1st/2st moment or
            covariance. `{tkey}` column must exist in your adata object and indicates the labeling time period.
            Parameters required for `dynamics` function.
        ekey: str or None (optional, default None)
            The dictionary key that corresponds to the gene expression in the layer attribute. By default, ekey and vkey
            will be automatically detected from the adata object. Parameters required by `cell_velocities`.
        vkey: str or None (optional, default None)
            The dictionary key that corresponds to the estimated velocity values in the layers attribute. Parameters
            required by `cell_velocities`
        basis: int (optional, default `umap`)
            The dictionary key that corresponds to the reduced dimension in `.obsm` attribute. Can be `X_spliced_umap`
            or `X_total_umap`, etc. Parameters required by `cell_velocities`

    Returns
    -------
        An updated adata object that went through a proper and typical time-resolved RNA velocity analysis.
    """
    from ..preprocessing import recipe_monocle
    from ..preprocessing import pca

    if not (all([i in adata.layers.keys() for i in ['uu', 'ul', 'su', 'sl']]) or
            all([i in adata.layers.keys() for i in ['new', 'total', 'spliced', 'unspliced']])):
        raise Exception(f"this recipe is only applicable to kinetics experiment dataset that have "
                        f"`'uu', 'ul', 'su', 'sl'` four layers.")

    # new, total, uu, ul, su, sl layers will be normalized with size factor calculated with total layers
    # spliced / unspliced layers will be normalized independently.
    recipe_monocle(adata, n_top_genes=n_top_genes, total_layers=True)

    # first calculate moments for labeling data relevant layers using total based connectivity graph
    moments(adata, group=tkey, layers=['X_new', 'X_total', 'X_uu', 'X_ul', 'X_su', 'X_sl'])

    # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced data.
    # first get X_spliced based pca embedding
    pca(adata, adata[:, adata.var.use_for_pca].layers['X_spliced'].A, pca_key='X_spliced_pca')
    # then get neighbors graph based on X_spliced_pca
    neighbors(adata, X_data=adata.obsm['X_spliced_pca'], layer='X_spliced')
    # then normalize neighbors graph so that each row sums up to be 1
    conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
    # then calculate moments for spliced related layers using spliced based connectivity graph
    moments(adata, group='time', conn=conn, layers=['X_spliced', 'X_unspliced'])
    # then perform kinetic estimations with properly preprocessed layers for either the labeling or the splicing data
    dynamics(adata, model='kinetic', tkey='time', est_method='twostep')  # no correction
    # then perform dimension reduction
    reduceDimension(adata, reduction_method='umap')
    # lastly, project RNA velocity to low dimensional embedding.
    cell_velocities(adata, enforce=True, vkey=vkey, ekey=ekey, basis=basis)

    return adata
