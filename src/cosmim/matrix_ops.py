"""Symmetric matrices stored as lower triangles, and the maps from a covariance matrix to a feature (the matrix square root by default)."""
import functools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

THREAD_CAP = 8          # more threads did not make it faster
THREAD_MIN_ROWS = 512   # below this, threads cost more than they save


@functools.lru_cache(maxsize=None)
def tril_indices(g):
    """Indices of the lower triangle of a g x g matrix."""
    return np.tril_indices(g)


def batches(n, batch_size):
    """Slices of at most `batch_size` covering range(n)."""
    if batch_size is None or batch_size >= n:
        return [slice(0, n)]
    return [slice(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def to_tril(mat):
    """(..., g, g) -> (..., g(g+1)/2)."""
    rows, cols = tril_indices(mat.shape[-1])
    return mat[..., rows, cols]


def from_tril(tril_flat, g):
    """(..., g(g+1)/2) -> (..., g, g)."""
    rows, cols = tril_indices(g)
    mat = np.zeros(tril_flat.shape[:-1] + (g, g), dtype=tril_flat.dtype)
    mat[..., rows, cols] = tril_flat
    mat[..., cols, rows] = tril_flat
    return mat


def resolve_n_jobs(n_jobs, n_rows):
    """Number of threads: None means up to 8."""
    if n_rows < THREAD_MIN_ROWS:
        return 1
    if n_jobs is None:
        return max(1, min(THREAD_CAP, os.cpu_count() or 1))
    return max(1, int(n_jobs))


def map_over_stack(fn, stack, out_shape, n_jobs=None):
    """Apply `fn` to chunks of a stack of matrices in threads. Each matrix is computed on its own, so the result does not depend on the number of threads."""
    n = len(stack)
    workers = resolve_n_jobs(n_jobs, n)
    if workers == 1:
        return fn(stack)
    bounds = np.linspace(0, n, workers + 1).astype(int)
    out = np.empty((n,) + tuple(out_shape), dtype=stack.dtype)

    def work(lo, hi):
        out[lo:hi] = fn(stack[lo:hi])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, bounds[:-1], bounds[1:]))     # list() re-raises errors from the threads
    return out


def eigh_map(tril_flat, g, fn, tril=True, n_jobs=None):
    """V diag(fn(lambda)) V^T of each symmetric matrix, from its eigendecomposition."""
    def kernel(chunk):
        vals, vecs = np.linalg.eigh(from_tril(chunk, g))
        mat = np.einsum("...ij,...j,...kj->...ik", vecs, fn(vals), vecs)
        return to_tril(mat) if tril else mat

    if tril_flat.ndim != 2:
        return kernel(tril_flat)
    out_shape = (tril_flat.shape[-1],) if tril else (g, g)
    return map_over_stack(kernel, tril_flat, out_shape, n_jobs)


def eigh_sqrt(tril_flat, g, tril=True, n_jobs=None):
    """Symmetric matrix square root, V diag(sqrt(lambda)) V^T."""
    return eigh_map(tril_flat, g, lambda vals: np.sqrt(np.clip(vals, 0, None)), tril, n_jobs)   # rounding can make tiny eigenvalues negative


def eigh_logm(tril_flat, g, tril=True, n_jobs=None):
    """Matrix logarithm, V diag(log(lambda)) V^T, eigenvalues floored at 1e-10."""
    return eigh_map(tril_flat, g, lambda vals: np.log(np.clip(vals, 1e-10, None)), tril, n_jobs)


def feature(tril_flat, g, transform="sqrt", n_jobs=None):
    """The feature of each covariance matrix: 'sqrt', 'logm' or 'raw' (the matrix itself)."""
    if transform == "sqrt":
        return eigh_sqrt(tril_flat, g, n_jobs=n_jobs)
    if transform == "logm":
        return eigh_logm(tril_flat, g, n_jobs=n_jobs)
    if transform == "raw":
        return tril_flat
    raise ValueError(f"transform must be 'sqrt', 'logm' or 'raw', not {transform!r}")
