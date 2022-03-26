import numpy as np
import pandas as pd
import anndata

from .utils import simulate_2bifurgenes

# dynamo logger related
from ..dynamo_logger import (
    LoggerManager,
    main_tqdm,
    main_info,
    main_warning,
    main_critical,
    main_exception,
)


class AnnDataSimulator:
    def __init__(self, simulator, C0s, param_dict, gene_names=None, gene_param_names=None) -> None:
        self.simulator = simulator
        self.C0s = np.atleast_2d(C0s)
        self.n_species = self.C0s.shape[1]
        self.n_genes = self.n_species  # will be changed later
        self.param_dict = param_dict

        if gene_names:
            self.gene_names = gene_names
        else:
            self.gene_names = ["gene_%d" % i for i in range(self.n_genes)]

        self.C = None
        self.T = None

    def augment_C0_gaussian(self, n, sigma=5):
        C0s = np.array(self.C0s, copy=True)
        for C0 in self.C0s:
            c = np.random.normal(scale=sigma, size=(n, self.n_species))
            c += C0
            c = np.clip(c, 0)
            C0s = np.vstack((C0s, c))
        self.C0s = C0s

    def simulate(self, t_span, n_traj=1, **simulator_kwargs):
        Ts, Cs, traj_id = [], [], []
        count = 0
        for C0 in self.C0s:
            trajs_T, trajs_C = self.simulator(
                C0=C0, t_span=t_span, n_traj=n_traj, param_dict=self.param_dict, **simulator_kwargs
            )
            for i, traj_T in enumerate(trajs_T):
                Ts.append(traj_T)
                Cs = np.hstack((Cs, trajs_C[i]))
                traj_id.append(count)
                count += 1

        self.T = np.array(Ts)
        self.C = Cs.T
        self.traj_id = traj_id

    def generate_anndata(self, remove_empty_cells=False, verbose=True):
        if self.T and self.C is not None:
            n_cells = self.C.shape[0]

            obs = pd.DataFrame(
                {
                    "cell_name": np.arange(n_cells),
                    "trajectory": self.traj_id,
                    "time": self.T,
                }
            )
            obs.set_index("cell_name", inplace=True)

            # work on params later
            var = pd.DataFrame(
                {
                    "gene_short_name": self.gene_names,
                }
            )  # use the real name in simulation?
            var.set_index("gene_short_name", inplace=True)

            # reserve for species
            layers = {
                "X": (self.C).astype(int),
            }

            adata = anndata.AnnData(
                self.C.astype(int),
                obs.copy(),
                var.copy(),
                layers=layers.copy(),
            )

            if remove_empty_cells:
                # remove cells that has no expression
                adata = adata[np.array(adata.X.sum(1)).flatten() > 0, :]

            if verbose:
                main_info("%s cell with %s genes stored in AnnData." % (n_cells, self.n_genes))
        else:
            raise Exception("No trajectory has been generated; Run simulation first.")


class Differentiation2Genes(AnnDataSimulator):
    def __init__(self, C0s=None, param_dict=None, gene_names=None, gene_param_names=None) -> None:
        super().__init__(simulate_2bifurgenes, C0s, param_dict, gene_names, gene_param_names)

        self.splicing = True if len(param_dict) > 8 else False
