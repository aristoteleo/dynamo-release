import os
import warnings

if "NVCC" not in os.environ:
    os.environ["NVCC"] = "/usr/local/cuda-11.5/bin/nvcc"
    warnings.warn(
        "NVCC Path not found, set to  : /usr/local/cuda-11.5/bin/nvcc . \nPlease set NVCC as appropitate to your environment"
    )

import cupy as cp
from numba import cuda
import math

##  Cuda JIT
code = """
#include <thrust/count.h>
extern "C" __global__
void sort_sparse_array(double *data, int*indices, int *indptr, int n_rows)
{
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if(tid >= n_rows) return;
      thrust::sort_by_key(thrust::seq, data+ indptr[tid], data + indptr[tid+1], indices + indptr[tid]);     
}
"""

kernel = cp.RawModule(code=code, backend="nvcc")
sort_f = kernel.get_function("sort_sparse_array")

## Numba function
@cuda.jit
def find_top_k_values(
    data, indices, indptr, output_values_ar, output_idx_ar, k, n_rows
):
    gid = cuda.grid(1)

    if gid >= n_rows:
        return

    row_st_ind = indptr[gid]
    row_end_ind = indptr[gid + 1] - 1

    k = min(k, 1 + row_end_ind - row_st_ind)
    for i in range(0, k):
        index = row_st_ind + i
        if data[index] != 0:
            output_values_ar[gid][i] = data[index]
            output_idx_ar[gid][i] = indices[index]


def find_top_k_values_sparse_matrix(X, k):

    X = X.copy()

    ### Output arrays to save the top k values
    values_ar = cp.full(fill_value=0, shape=(X.shape[0], k), dtype=cp.float64)
    idx_ar = cp.full(fill_value=-1, shape=(X.shape[0], k), dtype=cp.int32)

    ### sort in decreasing order
    X.data = X.data * -1
    sort_f(
        (math.ceil(X.shape[0] / 32),), (32,), (X.data, X.indices, X.indptr, X.shape[0])
    )
    X.data = X.data * -1

    ## configure kernel based on number of tasks
    find_top_k_values_k = find_top_k_values.forall(X.shape[0])

    find_top_k_values_k(X.data, X.indices, X.indptr, values_ar, idx_ar, k, X.shape[0])

    return idx_ar, values_ar


def top_n_sparse(X, n):
    """Return indices,values of top n values in each row of a sparse matrix
    Args:
        X: The sparse matrix from which to get the
        top n indices and values per row
        n: The number of highest values to extract from each row
    Returns:
        indices: The top n indices per row
        values: The top n values per row
    """
    value_ls, idx_ls = [], []
    batch_size = 500
    for s in range(0, X.shape[0], batch_size):
        e = min(s + batch_size, X.shape[0])
        idx_ar, value_ar = find_top_k_values_sparse_matrix(X[s:e], n)
        value_ls.append(value_ar)
        idx_ls.append(idx_ar)

    indices = cp.concatenate(idx_ls)
    values = cp.concatenate(value_ls)

    return indices, values