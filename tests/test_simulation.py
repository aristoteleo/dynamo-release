import dynamo as dyn


def test_simulate_anndata():
    import numpy as np

    bifur2genes_params = {
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

    neurongenesis_params = {
        "gamma": np.ones(12),
        "a": [2.2, 4, 3, 3, 3, 4, 5, 5, 3, 3, 3, 3],
        "K": [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "n": 4 * np.ones(12),
    }

    simulator = dyn.sim.BifurcationTwoGenes(param_dict=bifur2genes_params)
    simulator.simulate(t_span=[0, 10], n_cells=1000)
    adata = simulator.generate_anndata()
    assert adata.n_vars == 2 and adata.n_obs == 1000

    simulator = dyn.sim.Neurongenesis(param_dict=neurongenesis_params)
    simulator.simulate(t_span=[0, 10], n_cells=1000)
    adata = simulator.generate_anndata()
    assert adata.n_vars == 12 and adata.n_obs == 1000

    simulator = dyn.sim.OscillationTwoGenes(param_dict=osc2genes_params)
    simulator.simulate(t_span=[0, 10], n_cells=1000)
    adata = simulator.generate_anndata()
    assert adata.n_vars == 2 and adata.n_obs == 1000

    kin_simulator = dyn.sim.KinLabelingSimulator(simulator=simulator)
    kin_simulator.simulate(label_time=5)
    adata2 = kin_simulator.write_to_anndata(adata)
    assert adata2.n_vars == 2 and adata2.n_obs == 1000


def test_Gillespie():
    adata, adata2 = dyn.sim.Gillespie(
        method="basic",
        a=[0.8, 0.8],
        b=[0.8, 0.8],
        la=[0.8, 0.8],
        aa=[0.8, 0.8],
        ai=[0.8, 0.8],
        si=[0.8, 0.8],
        be=[1, 1],
        ga=[1, 1],
    )
    assert adata.n_vars == 2

    adata, adata2 = dyn.sim.Gillespie(
        method="simulate_2bifurgenes",
    )
    assert adata.n_vars == 2

    adata, adata2 = dyn.sim.Gillespie(
        method="differentiation",
    )
    assert adata.n_vars == 2

    adata, adata2 = dyn.sim.Gillespie(
        method="oscillation",
        a=[0.8, 0.8],
        b=[0.8, 0.8],
        la=[0.8, 0.8],
        aa=[0.8, 0.8],
        ai=[0.8, 0.8],
        si=[0.8, 0.8],
        be=[1, 1],
        ga=[1, 1],
    )
    assert adata.n_vars == 2
