import numpy as np

def specificity(percentage, perfect_specificity):
    """Calculate specificity"""

    spec = 1 - JSdistVec(makeprobsvec(percentage), perfect_specificity)

    return spec


def makeprobsvec(p):
    """Calculate the probability matrix for a relative abundance matrix"""

    phat = p / np.sum(p)
    phat[np.isnan((phat))] = 0

    return phat


def shannon_entropy(p):
    """Calculate the Shannon entropy based on the probability vector"""

    if np.min(p) < 0 or np.sum(p) <= 0:
        return np.inf
    p_norm = p[p > 0]/np.sum(p)

    return - np.sum(np.log(p_norm) * p_norm)


def JSdistVec(p, q):
    """Calculate the Jessen-Shannon distance for two probability distribution"""

    Jsdiv = shannon_entropy((p + q) / 2) - (shannon_entropy(p) + shannon_entropy(q)) / 2
    if np.isinf(Jsdiv): Jsdiv = 1
    if Jsdiv < 0: Jsdiv = 0
    JSdist = np.sqrt(Jsdiv)

    return JSdist


def fetch_X_data(adata, genes, layer):
    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError(f'No genes from your genes list appear in your adata object.')

    if layer == None:
        if genes is not None:
            X_data = adata[:, genes].X
        else:
            X_data = adata.X if 'use_for_dynamo' not in adata.var.keys() \
                else adata[:, adata.var.use_for_dynamo].X
            genes = adata.var_names[adata.var.use_for_dynamo]
    else:
        if genes is not None:
            X_data = adata[:, genes].layers[layer]
        else:
            X_data = adata.layers[layer] if 'use_for_dynamo' not in adata.var.keys() \
                else adata[:, adata.var.use_for_dynamo].layers[layer]
            genes = adata.var_names[adata.var.use_for_dynamo]

    return genes, X_data
