from typing import Callable, Dict, List, Optional, Union

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
from ..tools.utils import flatten, isarray
from .ODE import (
    hill_act_func,
    hill_inh_func,
    neurongenesis,
    ode_bifur2genes,
    ode_neurongenesis,
    ode_osc2genes,
)
from .utils import CellularSpecies, GillespieReactions, Reaction

bifur2genes_params = {
    "gamma": [0.2, 0.2],
    "a": [0.5, 0.5],
    "b": [0.5, 0.5],
    "S": [2.5, 2.5],
    "K": [2.5, 2.5],
    "m": [5, 5],
    "n": [5, 5],
}
bifur2genes_splicing_params = {
    "beta": [1.0, 1.0],
    "gamma": [0.2, 0.2],
    "a": [0.5, 0.5],
    "b": [0.5, 0.5],
    "S": [2.5, 2.5],
    "K": [2.5, 2.5],
    "m": [5, 5],
    "n": [5, 5],
}
osc2genes_params = {
    "gamma": [0.5, 0.5],
    "a": [1.5, 0.5],
    "b": [1.0, 2.5],
    "S": [2.5, 2.5],
    "K": [2.5, 2.5],
    "m": [5, 5],
    "n": [10, 10],
}
osc2genes_splicing_params = {
    "beta": [1.0, 1.0],
    "gamma": [0.5, 0.5],
    "a": [1.5, 0.5],
    "b": [1.0, 2.5],
    "S": [2.5, 2.5],
    "K": [2.5, 2.5],
    "m": [5, 5],
    "n": [10, 10],
}
neurongenesis_params = {
    "gamma": np.ones(12),
    "a": [2.2, 4, 3, 3, 3, 4, 5, 5, 3, 3, 3, 3],
    "K": [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "n": 4 * np.ones(12),
}


class AnnDataSimulator:
    def __init__(
        self,
        reactions: GillespieReactions,
        C0s: Optional[np.ndarray],
        param_dict: Dict,
        species: Union[None, CellularSpecies] = None,
        gene_param_names: List = [],
        required_param_names: List = [],
        velocity_func: Optional[Callable] = None,
    ) -> None:

        # initialization of variables
        self.reactions = reactions
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
            species.register_species("r", True)
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
        param_dict = {}
        for k, v in self.param_dict.items():
            if isarray(v):
                param_dict[k] = np.array(v, copy=True)
            else:
                param_dict[k] = v

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
            T, C = self.reactions.simulate(t_span=t_span, C0=C0, **simulator_kwargs)
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
        self.traj_id = np.array(traj_id, dtype=int)

        if self.vfunc is not None:
            V = []
            for c in self.C:
                v = self.vfunc(c)
                V.append(flatten(v))
            self.V = np.array(V)

    def generate_anndata(self, remove_empty_cells: bool = False):
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

            # total velocity
            layers = {}
            if self.V is not None:
                layers["velocity_T"] = self.V

            # gene species
            X = np.zeros((self.get_n_cells(), self.get_n_genes()))
            for species, indices in self.species.iter_gene_species():
                S = self.C[:, indices]
                layers[species] = S
                X += S
            if "total" not in layers.keys():
                layers["total"] = X

            adata = anndata.AnnData(
                X,
                obs.copy(),
                var.copy(),
                layers=layers.copy(),
            )

            if remove_empty_cells:
                # remove cells that has no expression
                adata = adata[np.array(adata.X.sum(1)).flatten() > 0, :]

            main_info("%s cell with %s genes stored in AnnData." % (self.get_n_cells(), self.get_n_genes()))
        else:
            raise Exception("No trajectory has been generated; Run simulation first.")

        return adata


class CellularModelSimulator(AnnDataSimulator):
    def __init__(
        self,
        gene_names: List,
        synthesis_param_names: List,
        param_dict: Dict,
        molecular_param_names: List = [],
        kinetic_param_names: List = [],
        C0s: Optional[np.ndarray] = None,
        r_aug: float = 1,
        tau: float = 1,
        n_C0s: int = 10,
        velocity_func: Optional[Callable] = None,
        report_stoich: bool = False,
    ) -> None:
        """
        An anndata simulator class handling models with synthesis, splicing (optional), and first-order degrdation reactions.

        Args:
            gene_names: List of gene names.
            synthesis_param_names: List of kinetic parameters used to calculate synthesis rates.
            param_dict: The parameter dictionary containing "a", "b", "S", "K", "m", "n", "beta" (optional), "gamma"
                if `param_dict` has the key "beta", the simulation includes the splicing process and therefore has 4 species (`unspliced` and `spliced` for each gene).
            molecular_param_names: List of names of parameters which has `number of molecules` in their dimensions. These parameters will be multiplied with `r_aug` for scaling.
            kinetic_param_names: List of names of parameters which has `time` in their dimensions. These parameters will be multiplied with `tau` to scale the kinetics.
            C0s: Initial conditions (# init cond. by # species). If None, the simulator automatically generates `n_C0s` initial conditions.
            r_aug: Parameter which augments steady state copy number for r1 and r2. At steady state, r1_s ~ r*(a1+b1)/ga1; r2_s ~ r*(a2+b2)/ga2
            tau: Time scale parameter which does not affect steady state solutions.
            n_C0s: Number of augmented initial conditions, if C0s is `None`.
            velocity_func: Function used to calculate velocity. If `None`, the velocity will not be calculated.
            report_stoich: Whether to report the Stoichiometry Matrix.
        """
        self.splicing = True if "beta" in param_dict.keys() else False
        self.gene_names = gene_names

        # register species
        species = CellularSpecies(gene_names)
        if self.splicing:
            species.register_species("unspliced", True)
            species.register_species("spliced", True)
        else:
            species.register_species("total", True)

        if C0s is None:
            C0s_ = np.zeros(len(species))
        else:
            C0s_ = C0s

        gene_param_names = synthesis_param_names.copy()
        if self.splicing:
            gene_param_names.append("beta")
        gene_param_names.append("gamma")

        # utilize super's init to initialize the class and fix param dict, w/o setting the simulation function
        super().__init__(
            None,
            C0s_,
            param_dict,
            species=species,
            gene_param_names=gene_param_names,
        )

        main_info("Adjusting parameters based on `r_aug` and `tau`...")
        if self.splicing:
            self.param_dict["beta"] /= tau
        self.param_dict["gamma"] /= tau

        for param in molecular_param_names:
            self.param_dict[param] *= r_aug

        for param in kinetic_param_names:
            self.param_dict[param] /= tau

        # register reactions and set the simulation function
        reactions = GillespieReactions(species)
        self.register_reactions(reactions)
        if report_stoich:
            main_info("Stoichiometry Matrix:")
            reactions.display_stoich()
        self.reactions = reactions

        # calculate C0 if not specified
        if C0s is None:
            C0s = self.auto_C0(r_aug, tau)

        self.C0s = C0s
        self.augment_C0_gaussian(n_C0s, sigma=5)
        main_info(f"{n_C0s} initial conditions have been created by augmentation.")

        # set the velocity func
        if velocity_func is not None:
            if self.splicing:
                param_dict = self.param_dict.copy()
                del param_dict["beta"]
                self.vfunc = lambda x, s=self.species["spliced"], param=param_dict: velocity_func(x[s], **param)
            else:
                self.vfunc = lambda x, param=self.param_dict: velocity_func(x, **param)

    def get_synthesized_species(self):
        """return species which are either `total` or `unspliced` when there is splicing."""
        name = "unspliced" if self.splicing else "total"
        return self.species[name]

    def get_primary_species(self):
        """return species which are either `total` or `spliced` when there is splicing."""
        name = "spliced" if self.splicing else "total"
        return self.species[name]

    def auto_C0(self, r_aug, tau):
        return np.zeros(len(self.species))

    def register_reactions(self, reactions: GillespieReactions):
        for i_gene in range(self.get_n_genes()):
            if self.splicing:
                u, s = self.species["unspliced", i_gene], self.species["spliced", i_gene]
                beta, gamma = self.param_dict["beta"][i_gene], self.param_dict["gamma"][i_gene]
                # u -> s
                reactions.register_reaction(Reaction([u], [s], lambda C, u=u, beta=beta: beta * C[u], desc="splicing"))
                # s -> 0
                reactions.register_reaction(
                    Reaction([s], [], lambda C, s=s, gamma=gamma: gamma * C[s], desc="degradation")
                )
            else:
                r = self.species["total", i_gene]
                gamma = self.param_dict["gamma"][i_gene]
                # r -> 0
                reactions.register_reaction(
                    Reaction([r], [], lambda C, r=r, gamma=gamma: gamma * C[r], desc="degradation")
                )
        return reactions

    def generate_anndata(self, remove_empty_cells: bool = False):
        adata = super().generate_anndata(remove_empty_cells)

        if self.splicing:
            beta, gamma = np.array(self.param_dict["beta"]), np.array(self.param_dict["gamma"])
            V = beta * adata.layers["unspliced"] - gamma * adata.layers["spliced"]
            adata.layers["velocity_S"] = V
        return adata


class KinLabelingSimulator:
    def __init__(
        self,
        simulator: CellularModelSimulator,
        syn_rxn_tag: str = "synthesis",
    ) -> None:

        self.n_cells = simulator.C.shape[0]
        self.splicing = simulator.splicing

        # register species
        if self.splicing:
            label_dict = {"unspliced": "ul", "spliced": "sl"}
        else:
            label_dict = {"total": "new"}

        label_species = []
        self.species = CellularSpecies(simulator.species.gene_names)
        for _, sp in label_dict.items():
            self.species.register_species(sp)
            label_species.append(sp)

        # register reactions
        self.reactions, self.syn_rxns = self.register_reactions(self.species, label_species, simulator.param_dict)

        # calculate synthesis rate (alpha) for each cell
        self.alpha = np.zeros((simulator.C.shape[0], self.species.get_n_genes()))
        for i, c in enumerate(simulator.C):
            for rxn in simulator.reactions:
                if rxn.desc == syn_rxn_tag:  # The reaction is synthesis
                    product = rxn.products[0]
                    if product in simulator.get_synthesized_species():  # The product is either total or unspliced
                        gene = simulator.species.get_species(product, return_gene_name=False)[1]
                        self.alpha[i, gene] = rxn.rate_func(c)

        self.Cl = None
        self.Tl = None
        self._label_time = None

    def get_n_cells(self):
        return self.n_cells

    def register_reactions(self, species: CellularSpecies, label_species, param_dict):
        reactions = GillespieReactions(species)
        syn_rxns = []
        if self.splicing:
            u = species[label_species[0]]
            s = species[label_species[1]]
            for i_gene in range(species.get_n_genes()):
                i_rxn = reactions.register_reaction(Reaction([], [u[i_gene]], None, desc="synthesis"))
                syn_rxns.append(i_rxn)
                reactions.register_reaction(
                    Reaction(
                        [u[i_gene]],
                        [s[i_gene]],
                        lambda C, u=u[i_gene], beta=param_dict["beta"][i_gene]: beta * C[u],
                        desc="splicing",
                    )
                )
                reactions.register_reaction(
                    Reaction(
                        [s[i_gene]],
                        [],
                        lambda C, s=s[i_gene], gamma=param_dict["gamma"][i_gene]: gamma * C[s],
                        desc="degradation",
                    )
                )
        else:
            r = species[label_species[0]]
            for i_gene in range(species.get_n_genes()):
                i_rxn = reactions.register_reaction(Reaction([], [r[i_gene]], None, desc="synthesis"))
                syn_rxns.append(i_rxn)
                reactions.register_reaction(
                    Reaction(
                        [r[i_gene]],
                        [],
                        lambda C, r=r[i_gene], gamma=param_dict["gamma"][i_gene]: gamma * C[r],
                        desc="degradation",
                    )
                )
        return reactions, syn_rxns

    def simulate(self, label_time):
        if np.isscalar(label_time):
            label_time = np.ones(self.get_n_cells()) * label_time
        self._label_time = label_time

        self.Tl, self.Cl = [], None
        for i in range(self.get_n_cells()):
            tau = label_time[i]
            # set alpha for each synthesis reaction
            for i_gene, i_rxn in enumerate(self.syn_rxns):
                self.reactions[i_rxn].rate_func = lambda C, alpha=self.alpha[i, i_gene]: alpha
            T, C = self.reactions.simulate([0, tau], np.zeros(len(self.species)))
            self.Tl = np.hstack((self.Tl, T[-1]))
            self.Cl = C[:, -1] if self.Cl is None else np.vstack((self.Cl, C[:, -1]))

    def write_to_anndata(self, adata: anndata):
        if adata.n_vars != self.species.get_n_genes():
            raise Exception(
                f"The input anndata has {adata.n_vars} genes while there are {self.species.get_n_genes()} registered."
            )

        if adata.n_obs != self.get_n_cells():
            raise Exception(f"The input anndata has {adata.n_obs} cells while there are {self.get_n_cells()} labeled.")

        if self.Tl is not None and self.Cl is not None:
            adata.obs["actual_label_time"] = self.Tl
            adata.obs["label_time"] = self._label_time
            # gene species go here
            for species, indices in self.species.iter_gene_species():
                S = self.Cl[:, indices]
                adata.layers[species] = S
                main_info("A layer is created for the labeled species %s." % species)
        else:
            raise Exception("No simulated data has been generated; Run simulation first.")

        return adata


class BifurcationTwoGenes(CellularModelSimulator):
    def __init__(
        self,
        param_dict: Dict,
        C0s: Optional[np.ndarray] = None,
        r_aug: float = 20,
        tau: float = 3,
        n_C0s: int = 10,
        gene_names: Optional[List] = None,
        report_stoich: bool = False,
    ) -> None:
        """
        Two gene toggle switch model anndata simulator.

        Args:
            param_dict: The parameter dictionary containing "a", "b", "S", "K", "m", "n", "beta" (optional), "gamma"
                if `param_dict` has the key "beta", the simulation includes the splicing process and therefore has 4 species (u and s for each gene).
            C0s: Initial conditions (# init cond. by # species). If None, the simulator automatically generates `n_C0s` initial conditions based on the steady states.
            r_aug: Parameter which augments steady state copy number for gene 1 (r1) and gene 2 (r2). At steady state, r1_s ~ r*(a1+b1)/ga1; r2_s ~ r*(a2+b2)/ga2
            tau: Time scale parameter which does not affect steady state solutions.
            n_C0s: Number of augmented initial conditions, if C0s is `None`.
            gene_names: List of gene names. If `None`, "gene_1", "gene_2", etc., are used.
            report_stoich: Whether to report the Stoichiometry Matrix.
        """
        if gene_names is None:
            gene_names = ["gene_1", "gene_2"]

        super().__init__(
            gene_names,
            synthesis_param_names=["a", "b", "S", "K", "m", "n"],
            param_dict=param_dict,
            molecular_param_names=["a", "b", "S", "K"],
            kinetic_param_names=["a", "b"],
            C0s=C0s,
            r_aug=r_aug,
            tau=tau,
            n_C0s=n_C0s,
            velocity_func=ode_bifur2genes,
            report_stoich=report_stoich,
        )

    def auto_C0(self, r_aug, tau):
        a, b = self.param_dict["a"], self.param_dict["b"]
        ga = self.param_dict["gamma"]

        x1_s = (a[0] + b[0]) / ga[0]
        x2_s = (a[1] + b[1]) / ga[1]
        if self.splicing:
            be = self.param_dict["beta"]
            C0s = np.zeros(len(self.species))
            C0s[self.species["unspliced", 0]] = ga[0] / be[0] * x1_s
            C0s[self.species["spliced", 0]] = x1_s
            C0s[self.species["unspliced", 1]] = ga[1] / be[1] * x2_s
            C0s[self.species["spliced", 1]] = x2_s
        else:
            C0s = np.array([x1_s, x2_s])
        return C0s

    def register_reactions(self, reactions):
        def rate_syn(x, y, gene):
            activation = hill_act_func(
                x, self.param_dict["a"][gene], self.param_dict["S"][gene], self.param_dict["m"][gene]
            )
            inhibition = hill_inh_func(
                y, self.param_dict["b"][gene], self.param_dict["K"][gene], self.param_dict["n"][gene]
            )
            return activation + inhibition

        if self.splicing:
            u1, u2 = self.species["unspliced", 0], self.species["unspliced", 1]
            s1, s2 = self.species["spliced", 0], self.species["spliced", 1]
            # 0 -> u1
            reactions.register_reaction(
                Reaction([], [u1], lambda C, s1=s1, s2=s2: rate_syn(C[s1], C[s2], 0), desc="synthesis")
            )
            # 0 -> u2
            reactions.register_reaction(
                Reaction([], [u2], lambda C, s1=s1, s2=s2: rate_syn(C[s2], C[s1], 1), desc="synthesis")
            )
        else:
            x1, x2 = self.species["total", 0], self.species["total", 1]
            # 0 -> x1
            reactions.register_reaction(
                Reaction([], [x1], lambda C, x1=x1, x2=x2: rate_syn(C[x1], C[x2], 0), desc="synthesis")
            )
            # 0 -> x2
            reactions.register_reaction(
                Reaction([], [x2], lambda C, x1=x1, x2=x2: rate_syn(C[x2], C[x1], 1), desc="synthesis")
            )

        super().register_reactions(reactions)


class OscillationTwoGenes(CellularModelSimulator):
    def __init__(
        self,
        param_dict: Dict,
        C0s: Optional[np.ndarray] = None,
        r_aug: float = 20,
        tau: float = 3,
        n_C0s: int = 10,
        gene_names: Optional[List] = None,
        report_stoich: bool = False,
    ) -> None:
        """
        Two gene oscillation model anndata simulator. This is essentially a predator-prey model, where gene 1 (predator) inhibits gene 2 (prey) and gene 2 activates gene 1.

        Args:
            param_dict: The parameter dictionary containing "a", "b", "S", "K", "m", "n", "beta" (optional), "gamma"
                if `param_dict` has the key "beta", the simulation includes the splicing process and therefore has 4 species (u and s for each gene).
            C0s: Initial conditions (# init cond. by # species). If None, the simulator automatically generates `n_C0s` initial conditions based on the steady states.
            r_aug: Parameter which augments copy numbers for the two genes.
            tau: Time scale parameter which does not affect steady state solutions.
            n_C0s: Number of augmented initial conditions, if C0s is `None`.
            gene_names: List of gene names. If `None`, "gene_1", "gene_2", etc., are used.
            report_stoich: Whether to report the Stoichiometry Matrix.
        """
        if gene_names is None:
            gene_names = ["gene_1", "gene_2"]

        super().__init__(
            gene_names,
            synthesis_param_names=["a", "b", "S", "K", "m", "n"],
            param_dict=param_dict,
            molecular_param_names=["a", "b", "S", "K"],
            kinetic_param_names=["a", "b"],
            C0s=C0s,
            r_aug=r_aug,
            tau=tau,
            n_C0s=n_C0s,
            velocity_func=ode_osc2genes,
            report_stoich=report_stoich,
        )

    def auto_C0(self, r_aug, tau):
        # TODO: derive solutions for auto C0
        if self.splicing:
            ga, be = self.param_dict["gamma"], self.param_dict["beta"]
            C0s = np.zeros(len(self.species))
            C0s[self.species["unspliced", 0]] = ga[0] / be[0] * 70
            C0s[self.species["spliced", 0]] = 70
            C0s[self.species["unspliced", 1]] = ga[1] / be[1] * 70
            C0s[self.species["spliced", 1]] = 70
        else:
            C0s = 70 * np.ones(len(self.species))
        return C0s

    def register_reactions(self, reactions):
        def rate_syn_1(x, y, gene):
            activation = hill_act_func(
                x, self.param_dict["a"][gene], self.param_dict["S"][gene], self.param_dict["m"][gene]
            )
            inhibition = hill_inh_func(
                y, self.param_dict["b"][gene], self.param_dict["K"][gene], self.param_dict["n"][gene]
            )
            return activation + inhibition

        def rate_syn_2(x, y, gene):
            activation1 = hill_act_func(
                x, self.param_dict["a"][gene], self.param_dict["S"][gene], self.param_dict["m"][gene]
            )
            activation2 = hill_act_func(
                y, self.param_dict["b"][gene], self.param_dict["K"][gene], self.param_dict["n"][gene]
            )
            return activation1 + activation2

        if self.splicing:
            u1, u2 = self.species["unspliced", 0], self.species["unspliced", 1]
            s1, s2 = self.species["spliced", 0], self.species["spliced", 1]
            # 0 -> u1
            reactions.register_reaction(
                Reaction([], [u1], lambda C, s1=s1, s2=s2: rate_syn_1(C[s1], C[s2], 0), desc="synthesis")
            )
            # 0 -> u2
            reactions.register_reaction(
                Reaction([], [u2], lambda C, s1=s1, s2=s2: rate_syn_2(C[s2], C[s1], 1), desc="synthesis")
            )
        else:
            x1, x2 = self.species["total", 0], self.species["total", 1]
            # 0 -> x1
            reactions.register_reaction(
                Reaction([], [x1], lambda C, x1=x1, x2=x2: rate_syn_1(C[x1], C[x2], 0), desc="synthesis")
            )
            # 0 -> x2
            reactions.register_reaction(
                Reaction([], [x2], lambda C, x1=x1, x2=x2: rate_syn_2(C[x2], C[x1], 1), desc="synthesis")
            )

        super().register_reactions(reactions)


class Neurongenesis(CellularModelSimulator):
    def __init__(
        self,
        param_dict: Dict,
        C0s: Optional[np.ndarray] = None,
        r_aug: float = 20,
        tau: float = 3,
        n_C0s: int = 10,
        report_stoich: bool = False,
    ) -> None:
        """
        Neurongenesis model from Xiaojie Qiu, et. al, 2012. anndata simulator.

        Args:
            param_dict: The parameter dictionary.
                if `param_dict` has the key "beta", the simulation includes the splicing process and therefore has 4 species (u and s for each gene).
            C0s: Initial conditions (# init cond. by # species). If None, the simulator automatically generates `n_C0s` initial conditions based on the steady states.
            r_aug: Parameter which augments steady state copy number for gene 1 (r1) and gene 2 (r2). At steady state, r1_s ~ r*(a1+b1)/ga1; r2_s ~ r*(a2+b2)/ga2
            tau: Time scale parameter which does not affect steady state solutions.
            n_C0s: Number of augmented initial conditions, if C0s is `None`.
            report_stoich: Whether to report the Stoichiometry Matrix.
        """

        gene_names = [
            "Pax6",
            "Mash1",
            "Zic1",
            "Brn2",
            "Tuj1",
            "Hes5",
            "Scl",
            "Olig2",
            "Stat3",
            "A1dh1L",
            "Myt1L",
            "Sox8",
        ]

        super().__init__(
            gene_names,
            synthesis_param_names=["a", "K", "n"],
            param_dict=param_dict,
            molecular_param_names=["a", "K"],
            kinetic_param_names=["a"],
            C0s=C0s,
            r_aug=r_aug,
            tau=tau,
            n_C0s=n_C0s,
            velocity_func=ode_neurongenesis,
            report_stoich=report_stoich,
        )

    def auto_C0(self, r_aug, tau):
        # C0 = np.ones(self.get_n_species()) * r_aug
        C0 = np.zeros(self.get_n_species())
        # TODO: splicing case
        C0[self.species["total", "Pax6"]] = 2.0 * r_aug
        return C0

    def register_reactions(self, reactions):
        def rate_pax6(x, y, z, gene):
            a = self.param_dict["a"][gene]
            K = self.param_dict["K"][gene]
            n = self.param_dict["n"][gene]
            rate = a * K**n / (K**n + x**n + y**n + z**n)
            return rate

        def rate_act(x, gene):
            a = self.param_dict["a"][gene]
            K = self.param_dict["K"][gene]
            n = self.param_dict["n"][gene]
            rate = a * x**n / (K**n + x**n)
            return rate

        def rate_toggle(x, y, gene):
            a = self.param_dict["a"][gene]
            K = self.param_dict["K"][gene]
            n = self.param_dict["n"][gene]
            rate = a * x**n / (K**n + x**n + y**n)
            return rate

        def rate_tuj1(x, y, z, gene):
            a = self.param_dict["a"][gene]
            K = self.param_dict["K"][gene]
            n = self.param_dict["n"][gene]
            rate = a * (x**n + y**n + z**n) / (K**n + x**n + y**n + z**n)
            return rate

        def rate_stat3(x, y, gene):
            a = self.param_dict["a"][gene]
            K = self.param_dict["K"][gene]
            n = self.param_dict["n"][gene]
            rate = a * (x**n * y**n) / (K**n + x**n * y**n)
            return rate

        if self.splicing:
            # TODO: develop the splicing model
            raise NotImplementedError("The splicing model has not been developed.")
        else:
            pax6 = self.species["total", "Pax6"]
            mash1 = self.species["total", "Mash1"]
            hes5 = self.species["total", "Hes5"]
            scl = self.species["total", "Scl"]
            olig2 = self.species["total", "Olig2"]
            zic1 = self.species["total", "Zic1"]
            brn2 = self.species["total", "Brn2"]
            tuj1 = self.species["total", "Tuj1"]
            a1dh1l = self.species["total", "A1dh1L"]
            sox8 = self.species["total", "Sox8"]
            stat3 = self.species["total", "Stat3"]
            myt1l = self.species["total", "Myt1L"]

            # 0 -> pax6
            reactions.register_reaction(
                Reaction(
                    [],
                    [pax6],
                    lambda C, x=tuj1, y=a1dh1l, z=sox8, g=pax6: rate_pax6(C[x], C[y], C[z], g),
                    desc="synthesis",
                )
            )
            # 0 -> mash1
            reactions.register_reaction(
                Reaction([], [mash1], lambda C, x=pax6, y=hes5, g=mash1: rate_toggle(C[x], C[y], g), desc="synthesis")
            )
            # 0 -> zic1
            reactions.register_reaction(
                Reaction([], [zic1], lambda C, x=mash1, g=zic1: rate_act(C[x], g), desc="synthesis")
            )
            # 0 -> brn2
            reactions.register_reaction(
                Reaction([], [brn2], lambda C, x=mash1, y=olig2, g=brn2: rate_toggle(C[x], C[y], g), desc="synthesis")
            )
            # 0 -> tuj1
            reactions.register_reaction(
                Reaction(
                    [],
                    [tuj1],
                    lambda C, x=zic1, y=brn2, z=myt1l, g=tuj1: rate_tuj1(C[x], C[y], C[z], g),
                    desc="synthesis",
                )
            )
            # 0 -> hes5
            reactions.register_reaction(
                Reaction([], [hes5], lambda C, x=pax6, y=mash1, g=hes5: rate_toggle(C[x], C[y], g), desc="synthesis")
            )
            # 0 -> scl
            reactions.register_reaction(
                Reaction([], [scl], lambda C, x=hes5, y=olig2, g=scl: rate_toggle(C[x], C[y], g), desc="synthesis")
            )
            # 0 -> olig2
            reactions.register_reaction(
                Reaction([], [olig2], lambda C, x=hes5, y=scl, g=olig2: rate_toggle(C[x], C[y], g), desc="synthesis")
            )
            # 0 -> stat3
            reactions.register_reaction(
                Reaction([], [stat3], lambda C, x=hes5, y=scl, g=stat3: rate_stat3(C[x], C[y], g), desc="synthesis")
            )
            # 0 -> a1dh1l
            reactions.register_reaction(
                Reaction([], [a1dh1l], lambda C, x=stat3, g=a1dh1l: rate_act(C[x], g), desc="synthesis")
            )
            # 0 -> myt1l
            reactions.register_reaction(
                Reaction([], [myt1l], lambda C, x=olig2, g=myt1l: rate_act(C[x], g), desc="synthesis")
            )
            # 0 -> sox8
            reactions.register_reaction(
                Reaction([], [sox8], lambda C, x=olig2, g=sox8: rate_act(C[x], g), desc="synthesis")
            )

        super().register_reactions(reactions)
