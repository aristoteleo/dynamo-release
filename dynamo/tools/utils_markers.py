import numpy as np

# ---------------------------------------------------------------------------------------------------
# specificity related


def specificity(percentage: np.ndarray, perfect_specificity: np.ndarray) -> float:
    """Calculate specificity"""

    spec = 1 - JSdistVec(makeprobsvec(percentage), perfect_specificity)

    return spec


def makeprobsvec(p: np.ndarray) -> np.ndarray:
    """Calculate the probability matrix for a relative abundance matrix"""

    phat = p / np.sum(p)
    phat[np.isnan((phat))] = 0

    return phat


def shannon_entropy(p: np.ndarray) -> float:
    """Calculate the Shannon entropy based on the probability vector"""

    if np.min(p) < 0 or np.sum(p) <= 0:
        return np.inf
    p_norm = p[p > 0] / np.sum(p)

    return -np.sum(np.log(p_norm) * p_norm)


def JSdistVec(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the Jessen-Shannon distance for two probability distribution"""

    Jsdiv = shannon_entropy((p + q) / 2) - (shannon_entropy(p) + shannon_entropy(q)) / 2
    if np.isinf(Jsdiv):
        Jsdiv = 1
    if Jsdiv < 0:
        Jsdiv = 0
    JSdist = np.sqrt(Jsdiv)

    return JSdist


# ---------------------------------------------------------------------------------------------------
# differential gene expression test related


def fdr(p_vals: np.ndarray) -> np.ndarray:
    """Calculate fdr_bh (Benjamini/Hochberg (non-negative))"""
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr
