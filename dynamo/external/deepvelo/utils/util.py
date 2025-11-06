import json
from typing import Dict

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from collections.abc import Mapping
from scvelo.core import l2_norm, prod_sum, sum


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
    return dirname


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def save_model_and_config(model, config, directory):
    """saves model and config to directory."""
    directory = ensure_dir(directory)
    torch.save(model.state_dict(), directory / "model.pt")
    write_json(config, directory / "config.json")


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def validate_config(config: Mapping) -> Mapping:
    """
    Return config if it is valid, otherwise raise an error.
    """

    # check if the gpu verion of dgl is installed
    if config["n_gpu"] > 0:
        import dgl

        try:
            dgl.graph([]).to("cuda")
        except dgl.DGLError:
            print(
                "Config Warning: Set to use GPU, but GPU version of DGL is not "
                "installed. Reset to use CPU instead."
            )
            config["n_gpu"] = 0

    return config


def update_dict(d: Dict, u: Mapping, copy=False):
    """recursively updates nested dict with values from u."""
    if copy:
        d = d.copy()
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = update_dict(d.get(k, {}), v, copy)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def get_indices(dist, n_neighbors=None, mode_neighbors="distances"):
    from scvelo.preprocessing.neighbors import compute_connectivities_umap

    D = dist.copy()
    D.data += 1e-6

    n_counts = sum(D > 0, axis=1)
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

    D.data -= 1e-6
    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D


def get_indices_from_csr(conn):
    # extracts indices from connectivity matrix, pads with nans
    ixs = np.ones((conn.shape[0], np.max((conn > 0).sum(1)))) * np.nan
    for i in range(ixs.shape[0]):
        cell_indices = conn[i, :].indices
        ixs[i, : len(cell_indices)] = cell_indices
    return ixs


def make_dense(X):
    from scipy.sparse import issparse

    XA = X.A if issparse(X) and X.ndim == 2 else X.A1 if issparse(X) else X
    if XA.ndim == 2:
        XA = XA[0] if XA.shape[0] == 1 else XA[:, 0] if XA.shape[1] == 1 else XA
    return np.array(XA)


def get_weight(x, y=None, perc=95):
    from scipy.sparse import issparse

    xy_norm = np.array(x.A if issparse(x) else x)
    if y is not None:
        if issparse(y):
            y = y.A
        xy_norm = xy_norm / np.clip(np.max(xy_norm, axis=0), 1e-3, None)
        xy_norm += y / np.clip(np.max(y, axis=0), 1e-3, None)

    if isinstance(perc, int):
        weights = xy_norm >= np.percentile(xy_norm, perc, axis=0)
    else:
        lb, ub = np.percentile(xy_norm, perc, axis=0)
        weights = (xy_norm <= lb) | (xy_norm >= ub)

    return weights


def R2(residual, total):
    r2 = np.ones(residual.shape[1]) - np.sum(residual * residual, axis=0) / np.sum(
        total * total, axis=0
    )
    r2[np.isnan(r2)] = 0
    return r2


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
