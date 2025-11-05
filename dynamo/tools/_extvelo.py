from anndata import AnnData
from typing import Literal, List


def extvelo(
    adata: AnnData,
    method: Literal["latentvelo", "celldancer"] = "celldancer",
    celltype_key: str = "clusters",
    basis: str = "X_umap",
    Ms_key: str = "M_s",
    Mu_key: str = "M_u",
    gene_list: List[str] = None,
    **kwargs,
) -> AnnData:
    if method == "celldancer":
        from ..external.celldancer.utilities import adata_to_df_with_embed 
        from ..external.celldancer import velocity
        from ..external.celldancer.utilities import export_velocity_to_dynamo
        if gene_list is None:
            gene_list = adata.var.query("use_for_pca==True").index.tolist()
        cell_type_u_s=adata_to_df_with_embed(adata,
                      us_para=[Mu_key,Ms_key],
                      cell_type_para=celltype_key,
                      embed_para=basis,
                      save_path='temp/test_cell_type_u_s.csv',
                      gene_list=gene_list)
        
        loss_df, cellDancer_df = velocity(cell_type_u_s, **kwargs)
        adata = export_velocity_to_dynamo(cellDancer_df,adata)
        return cellDancer_df,adata
    else:
        raise ValueError(f"Method {method} not supported")