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

diff2genes_params = {
                'gamma': 0.2,
                'a1': 1.5,
                'b1': 1,
                'a2': 0.5,
                'b2': 2.5,
                'K': 2.5,
                'n': 10
            }

diff2genes_splicing_params = {
                'beta': 0.5,
                'gamma': 0.2,
                'a1': 1.5,
                'b1': 1,
                'a2': 0.5,
                'b2': 2.5,
                'K': 2.5,
                'n': 10
            }

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

    def augment_C0_gaussian(self, n, sigma=5, inplace=True):
        C0s = np.array(self.C0s, copy=True)
        for C0 in self.C0s:
            c = np.random.normal(scale=sigma, size=(n, self.n_species))
            c += C0
            c = np.clip(c, 0)
            C0s = np.vstack((C0s, c))
        if inplace:
            self.C0s = C0s
        return C0s

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
    def __init__(self, param_dict, C0s=None, r=20, tau=3, n_C0s=10, gene_names=None, gene_param_names=None) -> None:
        '''
            2 gene toggle switch model anndata simulator. 
            r: controls steady state copy number for x1 and x2. At steady state, x1_s ~ r*(a1+b1)/ga1; x2_s ~ r*(a2+b2)/ga2
        '''

        param_dict = self.fix_param_dict(param_dict)

        main_info('Adjusting parameters based on r and tau...')

        if 'be1' in param_dict.keys():
            param_dict['be1'] /= tau
        if 'be2' in param_dict.keys():
            param_dict['be2'] /= tau

        param_dict['ga1'] /= tau
        param_dict['ga2'] /= tau
        param_dict['a1'] *= r/tau
        param_dict['a2'] *= r/tau
        param_dict['b1'] *= r/tau
        param_dict['b2'] *= r/tau
        param_dict['K'] *= r

        if 'be1' in param_dict.keys() and 'be2' in param_dict.keys():
            self.splicing = True
        else:
            self.splicing = False

        # calculate C0 if not specified, C0 ~ [x1_s, x2_s]
        if C0s is None:
            a1, a2, b1, b2 = param_dict['a1'], param_dict['a2'], param_dict['b1'], param_dict['b2']
            ga1, ga2 = param_dict['ga1'], param_dict['ga2']

            x1_s = r*(a1+b1)/ga1
            x2_s = r*(a2+b2)/ga2
            if self.splicing:
                be1, be2 = param_dict['be1'], param_dict['be2']
                C0s = np.array([ga1/be1*x1_s, x1_s, ga2/be2*x2_s, x2_s])
            else:
                C0s = np.array([x1_s, x2_s])

            C0s = self.augment_C0_gaussian(n_C0s, sigma=5, inplace=False)

        super().__init__(simulate_2bifurgenes, C0s, param_dict, gene_names, gene_param_names)

    def fix_param_dict(self, param_dict):
        param_dict = param_dict.copy()
        # gene specific required parameters
        if 'a' in param_dict.keys():
            if 'a1' not in param_dict.keys():
                param_dict['a1'] = param_dict['a']
            if 'a2' not in param_dict.keys():
                param_dict['a2'] = param_dict['a']
            main_info(f"universal values for a1 and a2: {param_dict['a']}")
            del param_dict['a']
        else:
            if 'a1' not in param_dict.keys() and 'a2' not in param_dict.keys():
                raise Exception('a1 or a2 not defined.')

        if 'b' in param_dict.keys():
            if 'b1' not in param_dict.keys():
                param_dict['b1'] = param_dict['b']
            if 'b2' not in param_dict.keys():
                param_dict['b2'] = param_dict['b']
            main_info(f"universal values for b1 and b2: {param_dict['b']}")
            del param_dict['b']
        else:
            if 'b1' not in param_dict.keys() and 'b2' not in param_dict.keys():
                raise Exception('b1 or b2 not defined.')

        if 'gamma' in param_dict.keys() and 'ga' not in param_dict.keys():
            param_dict['ga'] = param_dict['gamma']
            del param_dict['gamma']
        if 'ga' in param_dict.keys():
            if 'ga1' not in param_dict.keys():
                param_dict['ga1'] = param_dict['ga']
            if 'b2' not in param_dict.keys():
                param_dict['ga2'] = param_dict['ga']
            main_info(f"universal values for ga1 and ga2: {param_dict['ga']}")
            del param_dict['ga']
        else:
            if 'ga1' not in param_dict.keys() and 'ga2' not in param_dict.keys():
                raise Exception('ga1 or ga2 not defined.')
        
        # gene specific optional parameters
        if 'beta' in param_dict.keys() and 'be' not in param_dict.keys():
            param_dict['be'] = param_dict['beta']
            del param_dict['beta']

        if 'be' in param_dict.keys():
            if 'be1' not in param_dict.keys():
                param_dict['be1'] = param_dict['be']
            if 'be2' not in param_dict.keys():
                param_dict['be2'] = param_dict['be']
            main_info(f"universal values for be1 and be2: {param_dict['be']}")
            del param_dict['be']

        # other parameters
        if 'K' not in param_dict.keys():
            raise Exception('K not defined.')

        if 'n' not in param_dict.keys():
            raise Exception('n not defined.')

        return param_dict