import pickle
from pathlib import Path

# from torchvision import datasets, transforms
import numpy as np
import torch
import dgl
import hnswlib
import scipy.sparse as sp
try:
    from dgl.contrib.sampling import NeighborSampler
except ImportError:
    from dgl.dataloading.neighbor_sampler import NeighborSampler

from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from anndata import AnnData

from ..base import BaseDataLoader


class VeloDataset(Dataset):
    def __init__(
        self,
        data_source,
        train=True,
        type="average",
        topC=30,
        topG=20,
        velocity_genes=False,
        use_scaled_u=False,
    ):
        # check if data_source is a file path or inmemory data
        if isinstance(data_source, str):
            data_source = Path(data_source)
            with open(data_source, "rb") as f:
                adata = pickle.load(f)
        elif isinstance(data_source, AnnData):
            adata = data_source
        else:
            raise ValueError("data_source must be a file path or anndata object")
        self.Ux_sz = adata.layers["Mu"]
        self.Sx_sz = adata.layers["Ms"]

        # Convert sparse matrices to dense arrays if needed
        if sp.issparse(self.Ux_sz):
            self.Ux_sz = self.Ux_sz.toarray()
        if sp.issparse(self.Sx_sz):
            self.Sx_sz = self.Sx_sz.toarray()

        if velocity_genes:
            self.Ux_sz = self.Ux_sz[:, adata.var["velocity_genes"]]
            self.Sx_sz = self.Sx_sz[:, adata.var["velocity_genes"]]
        if use_scaled_u:
            scaling = np.std(self.Ux_sz, axis=0) / np.std(self.Sx_sz, axis=0)
            self.Ux_sz = self.Ux_sz / scaling
        self.connectivities = adata.obsp["connectivities"]  # shape (cells, features)

        self.topG = topG
        N_cell, N_gene = self.Sx_sz.shape

        # build the knn graph in the original space
        # TODO: try using the original connectivities to build the graph
        if "pca" in type:
            n_pcas = 30
            pca_ = PCA(
                n_components=n_pcas,
                svd_solver="arpack",
            )
            Sx_sz_pca = pca_.fit_transform(self.Sx_sz)
            if N_cell < 3000:
                ori_dist = pairwise_distances(Sx_sz_pca, Sx_sz_pca)
                self.ori_idx = np.argsort(ori_dist, axis=1)[:, :topG]  # (1720, 20)
                self.nn_t_idx = np.argsort(ori_dist, axis=1)[:, 1:topC]
            else:
                p = hnswlib.Index(space="l2", dim=n_pcas)
                p.init_index(max_elements=N_cell, ef_construction=200, M=30)
                p.add_items(Sx_sz_pca)
                p.set_ef(max(topC, topG) + 10)
                self.ori_idx = p.knn_query(Sx_sz_pca, k=topG)[0].astype(int)
                self.nn_t_idx = p.knn_query(Sx_sz_pca, k=topC)[0][:, 1:].astype(int)
        else:
            raise NotImplementedError(
                "the argument type of VeloDataset has to include original "
                "distance method 'pca' or 'raw'"
            )

        self.g = self.build_graph(self.ori_idx)
        self.nn_idx = self.nn_t_idx
        self.neighbor_time = 0

        # update the velocity target vectors for spliced and unspliced counts
        self.velo = np.zeros(self.Sx_sz.shape, dtype=np.float32)
        self.velo_u = np.zeros(self.Ux_sz.shape, dtype=np.float32)
        for i in range(N_cell):
            self.velo[i] = np.mean(self.Sx_sz[self.nn_idx[i]], axis=0) - self.Sx_sz[i]
            self.velo_u[i] = np.mean(self.Ux_sz[self.nn_idx[i]], axis=0) - self.Ux_sz[i]

        # build masks
        mask = np.ones([N_cell, N_gene])
        mask[self.Ux_sz == 0] = 0
        mask[self.Sx_sz == 0] = 0

        self.Ux_sz = torch.tensor(self.Ux_sz, dtype=torch.float32)
        self.Sx_sz = torch.tensor(self.Sx_sz, dtype=torch.float32)
        self.velo = torch.tensor(self.velo, dtype=torch.float32)
        self.velo_u = torch.tensor(self.velo_u, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        print("velo data shape:", self.velo.shape)

    def large_batch(self, device):
        """
        build the large batch for training
        """
        # check if self._large_batch is already built
        if hasattr(self, "_large_batch"):
            return self._large_batch
        self._large_batch = [
            {
                "Ux_sz": self.Ux_sz.to(device),
                "Sx_sz": self.Sx_sz.to(device),
                "velo": self.velo.to(device),
                "velo_u": self.velo_u.to(device),
                "mask": self.mask.to(device),
                "t+1 neighbor idx": torch.tensor(
                    self.nn_t_idx,
                    dtype=torch.long,
                ).to(device),
            }
        ]
        return self._large_batch

    def __len__(self):
        return len(self.Ux_sz)  # 1720

    def __getitem__(self, i):
        data_dict = {
            "Ux_sz": self.Ux_sz[i],
            "Sx_sz": self.Sx_sz[i],
            "velo": self.velo[i],
            "velo_u": self.velo_u[i],
            "mask": self.mask[i],
            "t+1 neighbor idx": self.nn_t_idx[i],
        }
        return data_dict

    def gen_neighbor_batch(self, size):
        rng = np.random.default_rng()
        indices = rng.integers(0, high=len(self), size=size)
        # self.neighbors_per_gene is the neighbor indices for all cells, shape
        # (N_cells, topG, genes)

        # TODO(Haotian): try the per gene version
        # Here since the per gene version encounters the 0 gene count bug, we first
        # use the per cell version, which is using self.ind
        return self.ind[indices, : self.topG].flatten()

    def build_graph(self, ind):
        """ind (N,k) contains neighbor index"""
        print("building graph")
        g = dgl.DGLGraph()
        g.add_nodes(self.Ux_sz.shape[0])
        edge_list = []
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                edge_list.append((i, ind[i, j]))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        return g


class VeloDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_source,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
        type="average",
        topC=30,
        topG=16,
        velocity_genes=False,
        use_scaled_u=False,
    ):
        self.data_source = data_source
        self.dataset = VeloDataset(
            data_source,
            train=training,
            type=type,
            topC=topC,
            topG=topG,
            velocity_genes=velocity_genes,
            use_scaled_u=use_scaled_u,
        )
        self.shuffle = shuffle
        self.is_large_batch = batch_size == len(self.dataset)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class VeloNeighborSampler(NeighborSampler, BaseDataLoader):
    """
    minibatch neighbor sampler using DGL NeighborSampler
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        num_neighbors,
        num_hops,
        shuffle=True,
        validation_split=0.0,
        num_workers=32,
        training=True,
    ):
        self.data_dir = data_dir
        self.dataset = VeloDataset(self.data_dir, train=training)
        # FIXME: the split_validation here is not working as in the BaseDataLoader
        # BaseDataLoader.__init__(self, self.dataset, batch_size, shuffle,
        # validation_split, num_workers)

        g = self.dataset.g
        norm = 1.0 / g.in_degrees().float().unsqueeze(1)
        g.ndata["Ux_sz"] = self.dataset.Ux_sz
        g.ndata["Sx_sz"] = self.dataset.Sx_sz
        g.ndata["velo"] = self.dataset.velo
        g.ndata["norm"] = norm
        # need to set to readonly for nodeflow
        g.readonly()

        NeighborSampler.__init__(
            self,
            g,
            batch_size,
            num_neighbors,
            neighbor_type="in",
            shuffle=shuffle,
            num_workers=num_workers,
            num_hops=num_hops,
            #  seed_nodes=train_nid
        )

    # FIXME: the split_validation here is not working as in the BaseDataLoader
    def split_validation(self):
        return None

    def __len__(self):
        return self.dataset.__len__()


if __name__ == "__main__":
    VeloDataset("./data/DG_norm_genes.npz")
    VeloNeighborSampler("./data/DG_norm_genes.npz", 32, 15, 4)
