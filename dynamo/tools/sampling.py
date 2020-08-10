import numpy as np 
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from .utils import timeit

class TRNET:
    def __init__(self, n_nodes, X):
        self.n_nodes = n_nodes
        self.n_dims = X.shape[1]
        self.X = X
        self.W = self.draw_sample(self.n_nodes)     # initialize the positions of nodes
    
    def draw_sample(self, n_samples):
        idx = np.random.randint(0, self.X.shape[0], n_samples)
        return self.X[idx]
        
    def runOnce(self, p, l, ep, c):
        # calc the squared distances ||w - p||^2
        D = p - self.W
        sD = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            sD[i] = D[i].dot(D[i])
        
        # calc the closeness rank k's
        I = np.argsort(sD)
        K = np.empty_like(I)
        K[I] = np.arange(len(I))       
        
        # move the nodes
        if c == 0:
            self.W += ep * np.exp(-K[:, None]/l) * D
        else:
            # move nodes whose displacements are larger than the cutoff to accelerate the calculation
            kc = - l * np.log(c/ep)
            idx = K < kc
            K = K[:, None]
            self.W[idx, :] += ep * np.exp(-K[idx]/l) * D[idx, :]
                
    def run(self, tmax=200, li=0.2, lf=0.01, ei=0.3, ef=0.05, c=0):
        tmax = int(tmax * self.n_nodes)
        li = li * self.n_nodes
        P = self.draw_sample(tmax)
        for t in tqdm(range(1, tmax + 1), desc='Running TRN'):
            # calc the parameters
            tt = t / tmax
            l = li * np.power(lf / li, tt)
            ep = ei * np.power(ef / ei, tt)
            # run once
            self.runOnce(P[t-1], l, ep, c)
    
    def run_n_pause(self, k0, k, tmax=200, li=0.2, lf=0.01, ei=0.3, ef = 0.05, c=0):
        tmax = int(tmax * self.n_nodes)
        li = li * self.n_nodes
        P = self.draw_sample(tmax)
        for t in range(k0, k + 1):
            # calc the parameters
            tt = t / tmax
            l = li * np.power(lf / li, tt)
            ep = ei * np.power(ef / ei, tt)           
            # run once
            self.runOnce(P[t-1], l, ep, c)
            
            if t % 1000 == 0:
                print (str(t) + " steps have been run")


@timeit
def trn(X, n, return_index=True, **kwargs):
    trnet = TRNET(n, X)
    trnet.run(**kwargs)
    if not return_index:
        return trnet.W
    else:
        if X.shape[0] > 200000 and X.shape[1] > 2: 
            from pynndescent import NNDescent

            nbrs = NNDescent(X, metric='euclidean', n_neighbors=1, n_jobs=-1, random_state=19491001)
            idx, _ = nbrs.query(trnet.W, k=1)
        else:
            alg = 'ball_tree' if X.shape[1] > 10 else 'kd_tree'
            nbrs = NearestNeighbors(n_neighbors=1, algorithm=alg, n_jobs=-1).fit(X)
            _, idx = nbrs.kneighbors(trnet.W)

        return idx[:, 0]

def sample_by_velocity(V, n):
    tmp_V = np.linalg.norm(V, axis=1)
    p = tmp_V / np.sum(tmp_V)
    idx = np.random.choice(np.arange(len(V)), size=n, p=p, replace=False)
    return idx

def lhsclassic(n_samples, n_dim):
    # From PyDOE
    # Generate the intervals
    cut = np.linspace(0, 1, n_samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(n_samples, n_dim)
    a = cut[:n_samples]
    b = cut[1: n_samples + 1]
    rdpoints = np.zeros(u.shape)
    for j in range(n_dim):
        rdpoints[:, j] = u[:, j] * (b - a) + a

    # Make the random pairings
    H = np.zeros(rdpoints.shape)
    for j in range(n_dim):
        order = np.random.permutation(range(n_samples))
        H[:, j] = rdpoints[order, j]

    return H

def sample(arr, n, method, X=None, V=None, **kwargs):
    if method == 'random':
        cell_idx = np.random.choice(arr, size=n, replace=False)
    elif method == 'velocity' and V is not None:
        cell_idx = sample_by_velocity(V, n, **kwargs)
    elif method == 'trn' and X is not None:
        cell_idx = trn(X, n, **kwargs)
    else:
        raise NotImplementedError(f"The sampling method {method} is not implemented or relevant data are not provided.")
    return cell_idx
