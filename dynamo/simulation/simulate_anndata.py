import anndata
import numpy as np
import pandas as pd

# dynamo logger related
from ..dynamo_logger import (
    LoggerManager,
    main_critical,
    main_exception,
    main_info,
    main_tqdm,
    main_warning,
)
from ..tools.sampling import sample
from ..tools.utils import flatten
from .ODE import ode_2bifurgenes
from .utils import simulate_2bifurgenes

diff2genes_params = {"gamma": 0.2, "a": 0.5, "b": 0.5, "S": 2.5, "K": 2.5, "m": 5, "n": 5}

diff2genes_splicing_params = {"beta": 0.5, "gamma": 0.2, "a": 0.5, "b": 0.5, "S": 2.5, "K": 2.5, "m": 5, "n": 5}


class AnnDataSimulator:
    def __init__(
        self,
        simulator,
        C0s,
        param_dict,
        gene_names=None,
        gene_species_dict=None,
        gene_param_names=[],
        required_param_names=[],
        velocity_func=None,
    ) -> None:

        # initialization of variables
        self.simulator = simulator
        self.C0s = np.atleast_2d(C0s)
        self.n_species = self.C0s.shape[1]
        self.param_dict = param_dict
        self.gene_param_names = gene_param_names
        self.required_param_names = required_param_names
        self.vfunc = velocity_func
        self.V = None

        # create/check species-to-gene mapping
        if gene_species_dict is None:
            main_info("No species-to-gene mapping is given: each species is considered a gene in `C0`.")
            if gene_names is None:
                self.gene_names = ["gene_%d" % i for i in range(self.n_species)]
            else:
                if len(gene_names) != self.n_species:
                    raise Exception(f"There are {len(gene_names)} gene names but {self.n_species} elements in `C0`.")
                else:
                    self.gene_names = gene_names
            self.gene_species_dict = {"x": np.arange(self.n_species)}
        else:
            self.gene_names = gene_names
            for k, v in gene_species_dict.items():
                v = np.atleast_1d(v)
                if self.gene_names is None:
                    self.gene_names = ["gene_%d" % i for i in range(len(v))]

                if len(v) != len(self.gene_names):
                    raise Exception(f"There are {len(self.gene_names)} genes but {len(v)} mappings for species {k}.")
                gene_species_dict[k] = v
            self.gene_species_dict = gene_species_dict

        # initialization of simulation results
        self.C = None
        self.T = None

        # fix parameters
        self.fix_param_dict()
        main_info(f"The model contains {self.get_n_genes()} genes and {self.n_species} species")

    def get_n_genes(self):
        return len(self.gene_names)

    def get_n_cells(self):
        if self.C is None:
            raise Exception("Simulation results not found. Run simulation first.")
        return self.C.shape[0]

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
                    param_dict[param_name] = np.ones(self.get_n_genes()) * param[0]
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

    def simulate(self, t_span, n_traj=1, n_cells=None, **simulator_kwargs):
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

        if n_cells is not None:
            n = Cs.shape[0]
            if n_cells > n:
                main_warning(f"Cannot sample {n_cells} from {n} simulated data points. Using all data points instead.")
            else:
                main_info(f"Sampling {n_cells} from {n} simulated data points.")
                cell_idx = sample(np.arange(n), n_cells, method="random")
                Ts = Ts[cell_idx]
                Cs = Cs[cell_idx]
                traj_id = traj_id[cell_idx]

        self.T = Ts
        self.C = Cs
        self.traj_id = traj_id

        if self.vfunc is not None:
            V = []
            for c in self.C:
                v = self.vfunc(c)
                V.append(flatten(v))
            self.V = np.array(V)

    def generate_anndata(self, remove_empty_cells=False, verbose=True):
        if self.T is not None and self.C is not None:

            obs = pd.DataFrame(
                {
                    "cell_name": np.arange(self.get_n_cells()),
                    "trajectory": self.traj_id,
                    "time": self.T,
                }
            )
            obs.set_index("cell_name", inplace=True)

            # work on params later
            var = pd.DataFrame(
                {
                    "gene_name": self.gene_names,
                }
            )
            var.set_index("gene_name", inplace=True)
            for param_name in self.gene_param_names:
                var[param_name] = self.param_dict[param_name]

            # gene species go here
            layers = {}
            if self.V is not None:
                layers["V"] = self.V

            X = np.zeros((self.get_n_cells(), self.get_n_genes()))
            for species, indices in self.gene_species_dict.items():
                S = self.C[:, indices]
                layers[species] = S
                X += S
            layers["X"] = X

            adata = anndata.AnnData(
                X,
                obs.copy(),
                var.copy(),
                layers=layers.copy(),
            )

            if remove_empty_cells:
                # remove cells that has no expression
                adata = adata[np.array(adata.X.sum(1)).flatten() > 0, :]

            if verbose:
                main_info("%s cell with %s genes stored in AnnData." % (self.get_n_cells(), self.get_n_genes()))
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

        if self.splicing:
            gene_param_names = ["a", "b", "S", "K", "m", "n", "beta", "gamma"]
            gene_species_dict = {"u": [0, 2], "s": [1, 3]}
        else:
            gene_param_names = ["a", "b", "S", "K", "m", "n", "gamma"]
            gene_species_dict = None

        super().__init__(
            simulate_2bifurgenes,
            C0s_,
            param_dict,
            gene_names,
            gene_species_dict=gene_species_dict,
            gene_param_names=gene_param_names,
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
        main_info(f"{n_C0s} initial conditions have been augmented.")

        if self.splicing:
            param_dict = self.param_dict.copy()
            del param_dict["beta"]
            self.vfunc = lambda x: ode_2bifurgenes(x[self.gene_species_dict["s"]], **param_dict)
        else:
            self.vfunc = lambda x: ode_2bifurgenes(x, **self.param_dict)
