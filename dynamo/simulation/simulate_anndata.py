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

diff2genes_params = {"gamma": 0.2, "a": 0.5, "b": 0.5, "S": 2.5, "K": 2.5, "m": 5, "n": 5}

diff2genes_splicing_params = {"beta": 0.5, "gamma": 0.2, "a1": 1.5, "b1": 1, "a2": 0.5, "b2": 2.5, "K": 2.5, "n": 10}


class AnnDataSimulator:
    def __init__(
        self,
        simulator,
        C0s,
        param_dict,
        gene_names=None,
        species_to_gene=None,
        gene_param_names=[],
        required_param_names=[],
    ) -> None:
        self.simulator = simulator
        self.C0s = np.atleast_2d(C0s)
        self.n_species = self.C0s.shape[1]
        self.param_dict = param_dict
        self.gene_param_names = gene_param_names
        self.required_param_names = required_param_names

        if species_to_gene is None:
            self.n_genes = self.n_species
        else:
            pass  # will be implemented later

        if gene_names:
            self.gene_names = gene_names
        else:
            self.gene_names = ["gene_%d" % i for i in range(self.n_genes)]

        self.C = None
        self.T = None

        self.fix_param_dict()
        main_info(f"The model contains {self.n_genes} genes and {self.n_species} species")

    def fix_param_dict(self):
        """
        Fixes parameters in place.
        """
        param_dict = self.param_dict.copy()

        # required parameters
        for param_name in self.required_param_names:
            if param_name in self.required_param_names and param_name not in param_dict:
                raise Exception(f"Required parameter `{param_name}` not defined.")

        # gene specific parameters
        for param_name in self.gene_param_names:
            if param_name in param_dict.keys():
                param = np.atleast_1d(param_dict[param_name])
                if len(param) == 1:
                    param_dict[param_name] = np.ones(self.n_genes) * param[0]
                    main_info(f"Universal value for parameter {param_name}: {param[0]}")
                else:
                    param_dict[param_name] = param

        self.param_dict = param_dict

    def augment_C0_gaussian(self, n, sigma=5, inplace=True):
        C0s = np.array(self.C0s, copy=True)
        for C0 in self.C0s:
            c = np.random.normal(scale=sigma, size=(n, self.n_species))
            c += C0
            c = np.clip(c, 0, None)
            C0s = np.vstack((C0s, c))
        if inplace:
            self.C0s = C0s
        return C0s

    def simulate(self, t_span, n_traj=1, **simulator_kwargs):
        Ts, Cs, traj_id = [], None, []
        count = 0
        for C0 in self.C0s:
            trajs_T, trajs_C = self.simulator(
                C0=C0, t_span=t_span, n_traj=n_traj, param_dict=self.param_dict, **simulator_kwargs
            )
            for i, traj_T in enumerate(trajs_T):
                Ts = np.hstack((Ts, traj_T))
                Cs = trajs_C[i].T if Cs is None else np.vstack((Cs, trajs_C[i].T))
                traj_id = np.hstack((traj_id, [count] * len(traj_T)))
                count += 1

        self.T = Ts
        self.C = Cs
        self.traj_id = traj_id

    def generate_anndata(self, remove_empty_cells=False, verbose=True):
        if self.T is not None and self.C is not None:
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

        return adata


class Differentiation2Genes(AnnDataSimulator):
    def __init__(self, param_dict, C0s=None, r=20, tau=3, n_C0s=10, gene_names=None) -> None:
        """
        Two gene toggle switch model anndata simulator.
        r: controls steady state copy number for x1 and x2. At steady state, x1_s ~ r*(a1+b1)/ga1; x2_s ~ r*(a2+b2)/ga2
        tau: a time scale parameter which does not affect steady state solutions.
        """

        self.splicing = True if "beta" in param_dict.keys() else False
        if C0s is None:
            C0s_ = np.zeros(4) if self.splicing else np.zeros(2)

        super().__init__(
            simulate_2bifurgenes,
            C0s_,
            param_dict,
            gene_names,
            gene_param_names=["a", "b", "S", "K", "m", "n", "beta", "gamma"],
            required_param_names=["a", "b", "S", "K", "m", "n", "gamma"],
        )

        main_info("Adjusting parameters based on r and tau...")

        if self.splicing:
            self.param_dict["beta"] /= tau
        self.param_dict["gamma"] /= tau
        self.param_dict["a"] *= r / tau
        self.param_dict["b"] *= r / tau
        self.param_dict["S"] *= r
        self.param_dict["K"] *= r

        # calculate C0 if not specified, C0 ~ [x1_s, x2_s]
        if C0s is None:
            a, b = self.param_dict["a"], self.param_dict["b"]
            ga = self.param_dict["gamma"]

            x1_s = (a[0] + b[0]) / ga[0]
            x2_s = (a[1] + b[1]) / ga[1]
            if self.splicing:
                be = self.param_dict["beta"]
                C0s = np.array([ga[0] / be[0] * x1_s, x1_s, ga[1] / be[1] * x2_s, x2_s])
            else:
                C0s = np.array([x1_s, x2_s])

        self.C0s = C0s
        self.augment_C0_gaussian(n_C0s, sigma=5)
