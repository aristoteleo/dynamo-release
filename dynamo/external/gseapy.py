from pathlib import Path
from typing import Union

from ..dynamo_logger import LoggerManager


def enrichr(
    genes: Union[list, str],
    organism: str,
    background: Union[str, None] = None,
    gene_sets: Union[str, list, tuple] = ["GO_Biological_Process_2018"],
    description: Union[str, None] = None,
    outdir: str = "./enrichr",
    cutoff: float = 0.05,
    no_plot: bool = False,
    **kwargs,
):
    """Perform gene list enrichment with gseapy.

    Parameters
    ----------
        genes:
            Flat file with list of genes, one gene id per row, or a python list object.
        organism:
            Enrichr supported organism. Select from (human, mouse, yeast, fly, fish, worm).
            see here for details: https://amp.pharm.mssm.edu/modEnrichr
              :param gene_sets:
        gene_sets:
            str, list, tuple of Enrichr Library name(s).
        description:
            name of analysis. optional.
        outdir:
             Output file directory
        cutoff:
            Show enriched terms which Adjusted P-value < cutoff. Only affects the output figure. Default: 0.05
        kwargs:
            additional arguments passed to the `gp.enrichr` function.

    Returns
    -------
        An Enrichr object, which obj.res2d stores your last query, obj.results stores your all queries.

    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.pancreatic_endocrinogenesis()
    >>> dyn.pp.recipe_monocle(adata, n_top_genes=1000, fg_kwargs={'shared_count': 20})
    >>> dyn.tl.dynamics(adata, model='stochastic')
    >>> dyn.tl.reduceDimension(adata, n_pca_components=30)
    >>> dyn.tl.cell_velocities(adata)
    >>> dyn.pl.streamline_plot(adata, color=['clusters'], basis='umap', show_legend='on data', show_arrowed_spines=False)
    >>> # perform gene enrichment analysis which will create the enrichr folder with saved figures and txt file of the
    >>> # enrichment analysis results and return an Enrichr object
    >>> enr = dyn.ext.enrichr(adata.var_names[adata.var.use_for_transition].to_list(), organism='mouse', outdir='./enrichr')
    >>> enr.results.head(5)
    >>> # simple plotting function
    >>> from gseapy.plot import barplot, dotplot
    >>> # to save your figure, make sure that ``ofname`` is not None
    >>> barplot(enr.res2d, title='GO_Biological_Process_2018', cutoff=0.05)
    >>> dotplot(enr.res2d, title='KEGG_2016',cmap='viridis_r', cutoff=0.05)
    """

    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("You need to install the package `gseapy`." "install gseapy via `pip install gseapy`")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_sets,  # GO_Biological_Process_2018
        organism=organism,  # don't forget to set organism to the one you desired! e.g. Yeast
        background=background,
        description=description,
        outdir=outdir,
        no_plot=no_plot,
        cutoff=cutoff,  # test dataset, use lower value from range(0,1)
        **kwargs,
    )

    return enr
