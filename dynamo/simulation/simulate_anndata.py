import anndata
import numpy as np
import pandas as pd

from typing import Callable, Union

from .utils import simulate_2bifurgenes, CellularSpecies, GillespieReactions
from .ODE import ode_2bifurgenes, hill_inh_func, hill_act_func
from ..tools.utils import flatten
from ..tools.sampling import sample


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

bifur2genes_params = {"gamma": 0.2, "a": 0.5, "b": 0.5, "S": 2.5, "K": 2.5, "m": 5, "n": 5}

bifur2genes_splicing_params = {"beta": 0.5, "gamma": 0.2, "a": 0.5, "b": 0.5, "S": 2.5, "K": 2.5, "m": 5, "n": 5}


class AnnDataSimulator:
    def __init__(
        self,
        simulator: Callable,
        C0s,
        param_dict,
        species: Union[None, CellularSpecies] = None,
        gene_param_names=[],
        required_param_names=[],
        velocity_func=None,
    ) -> None:

        # initialization of variables
        self.simulator = simulator
        self.C0s = np.atleast_2d(C0s)
        self.param_dict = param_dict
        self.gene_param_names = gene_param_names
        self.vfunc = velocity_func
        self.V = None

        # create/check species-to-gene mapping
        n_species = self.C0s.shape[1]
        if species is None:
            main_info("No species-to-gene mapping is given: each species is considered a gene in `C0`.")
            gene_names = ["gene_%d" % i for i in range(n_species)]
            species = CellularSpecies(gene_names)
            species.register_species("x", True)
        self.species = species

        # initialization of simulation results
        self.C = None
        self.T = None

        # fix parameters
        self.fix_param_dict(required_param_names)

        main_info(f"The model contains {self.get_n_genes()} genes and {self.get_n_species()} species")

    def get_n_genes(self):
        return self.species.get_n_genes()

    def get_n_species(self):
        return len(self.species)

    def get_n_cells(self):
        if self.C is None:
            raise Exception("Simulation results not found. Run simulation first.")
        return self.C.shape[0]

    def fix_param_dict(self, required_param_names):
        """
        Fixes parameters in place.
        """
        param_dict = self.param_dict.copy()

        # required parameters
        for param_name in required_param_names:
            if param_name in required_param_names and param_name not in param_dict:
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
            c = np.random.normal(scale=sigma, size=(n, self.get_n_species()))
            c += C0
            c = np.clip(c, 0, None)
            C0s = np.vstack((C0s, c))
        if inplace:
            self.C0s = C0s
        return C0s

    def simulate(self, t_span, n_cells=None, **simulator_kwargs):
        Ts, Cs, traj_id = [], None, []
        count = 0
        for C0 in self.C0s:
            T, C = self.simulator(t_span=t_span, C0=C0, **simulator_kwargs)
            Ts = np.hstack((Ts, T))
            Cs = C.T if Cs is None else np.vstack((Cs, C.T))
            traj_id = np.hstack((traj_id, [count] * len(T)))
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

            var = pd.DataFrame(
                {
                    "gene_name": self.species.gene_names,
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
            for species, indices in self.species.iter_gene_species():
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


class BifurcationTwoGenes(AnnDataSimulator):
    def __init__(self, param_dict, C0s=None, r=20, tau=3, n_C0s=10, gene_names=None) -> None:
        """
        Two gene toggle switch model anndata simulator.

        Parameters
        ----------
            param_dict: dict
                The parameter dictionary containing "a", "b", "S", "K", "m", "n", "beta" (optional), "gamma"
                if `param_dict` has the key "beta", the simulation includes the splicing process and therefore has 4 species (u and s for each gene).
            C0s: None or :class:`~numpy.ndarray`
                Initial conditions (# init cond. x # species). If None, the simulator automatically generates `n_C0s` initial conditions based on the steady states.
            r: float
                Controls steady state copy number for x1 and x2. At steady state, x1_s ~ r*(a1+b1)/ga1; x2_s ~ r*(a2+b2)/ga2
            tau: float
                A time scale parameter which does not affect steady state solutions.
            n_C0s: int
                Number of augmented initial conditions, if C0s is `None`.
            gene_names: None or list
                A list of gene names. If `None`, "gene_1", "gene_2", etc., are used.
        """
        self.splicing = True if "beta" in param_dict.keys() else False
        if C0s is None:
            C0s_ = np.zeros(4) if self.splicing else np.zeros(2)  # splicing: 4 species (u1, s1, u2, s2)

        if gene_names is None:
            gene_names = ["gene_1", "gene_2"]

        # register species
        species = CellularSpecies(gene_names)
        if self.splicing:
            species.register_species("u", True)
            species.register_species("s", True)
        else:
            species.register_species("x", True)

        if self.splicing:
            gene_param_names = ["a", "b", "S", "K", "m", "n", "beta", "gamma"]
        else:
            gene_param_names = ["a", "b", "S", "K", "m", "n", "gamma"]

        # utilize super's init to initialize the class and fix param dict, w/o setting the simulator
        super().__init__(
            None,
            C0s_,
            param_dict,
            species=species,
            gene_param_names=gene_param_names,
        )

        main_info("Adjusting parameters based on r and tau...")
        if self.splicing:
            self.param_dict["beta"] /= tau
        self.param_dict["gamma"] /= tau
        self.param_dict["a"] *= r / tau
        self.param_dict["b"] *= r / tau
        self.param_dict["S"] *= r
        self.param_dict["K"] *= r

        # register reactions and set the simulator
        reactions = self.register_reactions()
        main_info("Stoichiometry Matrix:")
        reactions.display_stoich()
        self.simulator = reactions.simulate

        # calculate C0 if not specified, C0 ~ [x1_s, x2_s]
        if C0s is None:
            a, b = self.param_dict["a"], self.param_dict["b"]
            ga = self.param_dict["gamma"]

            x1_s = (a[0] + b[0]) / ga[0]
            x2_s = (a[1] + b[1]) / ga[1]
            if self.splicing:
                be = self.param_dict["beta"]
                C0s = np.zeros(len(self.species))
                C0s[self.species["u", 0]] = ga[0] / be[0] * x1_s
                C0s[self.species["s", 0]] = x1_s
                C0s[self.species["u", 1]] = ga[1] / be[1] * x2_s
                C0s[self.species["s", 1]] = x2_s
            else:
                C0s = np.array([x1_s, x2_s])

        self.C0s = C0s
        self.augment_C0_gaussian(n_C0s, sigma=5)
        main_info(f"{n_C0s} initial conditions have been created by augmentation.")

        # set the velocity func
        if self.splicing:
            param_dict = self.param_dict.copy()
            del param_dict["beta"]
            self.vfunc = lambda x: ode_2bifurgenes(x[self.species["s"]], **param_dict)
        else:
            self.vfunc = lambda x: ode_2bifurgenes(x, **self.param_dict)

    def register_reactions(self):
        reactions = GillespieReactions(self.species)

        def rate_syn(x, y, gene):
            activation = hill_act_func(
                x, self.param_dict["a"][gene], self.param_dict["S"][gene], self.param_dict["m"][gene]
            )
            inhibition = hill_inh_func(
                y, self.param_dict["b"][gene], self.param_dict["K"][gene], self.param_dict["n"][gene]
            )
            return activation + inhibition

        if self.splicing:
            u1, u2 = self.species["u", 0], self.species["u", 1]
            s1, s2 = self.species["s", 0], self.species["s", 1]
            # 0 -> u1
            reactions.register_reaction([], [u1], lambda C: rate_syn(C[s1], C[s2], 0), desc="synthesis")
            # u1 -> s1
            reactions.register_reaction([u1], [s1], lambda C: self.param_dict["beta"][0] * C[u1], desc="splicing")
            # s1 -> 0
            reactions.register_reaction([s1], [], lambda C: self.param_dict["gamma"][0] * C[s1], desc="degradation")
            # 0 -> u2
            reactions.register_reaction([], [u2], lambda C: rate_syn(C[s2], C[s1], 0), desc="synthesis")
            # u1 -> s1
            reactions.register_reaction([u2], [s2], lambda C: self.param_dict["beta"][1] * C[u2], desc="splicing")
            # s2 -> 0
            reactions.register_reaction([s2], [], lambda C: self.param_dict["gamma"][1] * C[s2], desc="degradation")
        else:
            x1, x2 = self.species["x", 0], self.species["x", 1]
            # 0 -> x1
            reactions.register_reaction([], [x1], lambda C: rate_syn(C[x1], C[x2], 0), desc="synthesis")
            # x1 -> 0
            reactions.register_reaction([x1], [], lambda C: self.param_dict["gamma"][0] * C[x1], desc="degradation")
            # 0 -> x2
            reactions.register_reaction([], [x2], lambda C: rate_syn(C[x2], C[x1], 0), desc="synthesis")
            # x2 -> 0
            reactions.register_reaction([x2], [], lambda C: self.param_dict["gamma"][1] * C[x2], desc="degradation")

        return reactions
