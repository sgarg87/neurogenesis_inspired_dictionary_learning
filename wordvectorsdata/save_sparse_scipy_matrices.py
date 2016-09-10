import numpy as np
import scipy.sparse as ss


def save_sparse_csr(filename, array):
    np.savez_compressed(filename,
             data=array.data,
             indices=array.indices,
             indptr=array.indptr,
             shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return ss.csr_matrix(
        (
            loader['data'],
            loader['indices'],
            loader['indptr']
        ),
        shape=loader['shape']
    )
