"""The CoSMIM feature: multi-scale niche covariance, scale weights, shrinkage and the matrix square root."""
import warnings

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .matrix_ops import batches, feature, from_tril, tril_indices

WEIGHT_EPS = 1e-10       # inside the logarithm of the entropy term
PENALTY_VALUE = 1e9      # when the fused covariance is not positive definite
OPTIMIZATION_FTOL = 1e-6


def build_knn_index(coords, k):
    """The k nearest cells of every cell; column 0 is the cell itself."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1).fit(coords)
    return nn.kneighbors(coords)[1]


def compute_multiscale_covariance(expr, global_mean, knn, k, cov_type="shift", exclude_target=False):
    """Niche covariance of a batch of cells at each scale k, as lower triangles: (batch, len(k), g(g+1)/2)."""
    g = expr.shape[1]
    rows, cols = tril_indices(g)
    identity = np.eye(g, dtype=expr.dtype)
    out = np.zeros((len(knn), len(k), len(rows)), dtype=expr.dtype)
    for j, kk in enumerate(k):
        niche = knn[:, 1:kk + 1] if exclude_target else knn[:, :kk + 1]
        x = expr[niche]
        if cov_type == "traditional":
            centred = x - np.mean(x, axis=1, keepdims=True)
        else:
            centred = x - global_mean[None, None, :]
        denominator = max(niche.shape[1] - 1, 1)
        cov = np.matmul(centred.transpose(0, 2, 1), centred) / denominator
        cov += cov.mean() * 1e-5 * identity                    # a small ridge keeps it invertible
        out[:, j] = cov[..., rows, cols]
    return out


def apply_regularization(covs, g, alpha=0.1):
    """Shrinkage: (1 - alpha) * C + alpha * mean(diag(C)) * I."""
    rows, cols = tril_indices(g)
    diag = rows == cols
    eye = np.eye(g, dtype=covs.dtype)[rows, cols]
    target = np.mean(covs[..., diag], axis=-1)[..., np.newaxis] * eye
    return (1 - alpha) * covs + alpha * target


def optimize_weights(mean_covs, entropy_reg=10, max_iter=100, reg_strength=1e-6):
    """Scale weights minimising -log det(sum_i w_i C_i) + entropy_reg * sum_i w_i log w_i, with w_i >= 0 and sum_i w_i = 1 (SLSQP). Uniform if the optimiser fails."""
    m, p = mean_covs.shape
    if m == 1:
        return np.array([1.0])
    g = int((np.sqrt(8 * p + 1) - 1) / 2)
    C = from_tril(np.asarray(mean_covs, dtype=np.float64), g) + reg_strength * np.eye(g)[None, :, :]

    def objective_and_grad(w):
        fused = np.einsum("i,ijk->jk", w, C)
        sign, log_det = np.linalg.slogdet(fused)
        if sign <= 0 or not np.isfinite(log_det):
            return PENALTY_VALUE, np.zeros_like(w)
        inv = scipy.linalg.inv(fused)
        w0 = np.clip(w, 0, None)
        loss = -log_det + entropy_reg * np.sum(w0 * np.log(w0 + WEIGHT_EPS))
        grad = -np.einsum("ij,kji->k", inv, C) + entropy_reg * (1 + np.log(w0 + WEIGHT_EPS))
        return loss, grad

    w_init = np.ones(m) / m
    result = scipy.optimize.minimize(objective_and_grad, w_init, method="SLSQP", jac=True, bounds=[(0, 1)] * m, constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}], options={"maxiter": max_iter, "ftol": OPTIMIZATION_FTOL, "disp": False})
    return result.x / result.x.sum() if result.success else w_init


def compute_cosmim(adata, k=(5, 15, 45), layer="log", spatial_key="spatial", cov_type="shift", weight_strategy="sampled", sample_size=1000, entropy_reg=10, batch_size=1000, alpha=0.1, exclude_target=False, weights=None, feature_transform="sqrt", n_jobs=None, dtype=np.float64, knn=None):
    """The CoSMIM feature of every cell. Returns (features, weights).

    The covariances are computed twice instead of being kept in memory: once to learn the weights, once for the feature. `knn` gives the neighbours instead of searching for them: (n_cells, max(k) + 1) cell indices, nearest first, column 0 the cell itself.
    """
    k = [k] if np.isscalar(k) else list(k)
    keep = adata.var["highly_variable"].values if "highly_variable" in adata.var else np.ones(adata.n_vars, dtype=bool)
    # the copy stores the layer column by column, which sets the order of the sums in the mean below
    sub = adata[:, keep].copy()
    expr = sub.X if layer is None else sub.layers[layer]
    if sparse.issparse(expr):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float64)
    n, g = expr.shape
    p = g * (g + 1) // 2
    global_mean = np.mean(expr, axis=0)
    if knn is None:
        knn = build_knn_index(sub.obsm[spatial_key], max(k) + 1)
    else:
        knn = np.asarray(knn)
        if knn.shape[0] != n or knn.shape[1] < max(k) + 1:
            raise ValueError(f"knn has shape {knn.shape}, expected ({n}, at least {max(k) + 1})")

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(k),):
            raise ValueError(f"weights has shape {weights.shape}, expected ({len(k)},)")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if not np.allclose(weights.sum(), 1.0, atol=1e-6):
            warnings.warn(f"weights sum to {weights.sum():.6f}; normalising them to 1")
            weights = weights / weights.sum()
    elif len(k) == 1:
        weights = np.array([1.0])
    else:
        if weight_strategy == "sampled":
            pick = np.random.RandomState(42).choice(n, size=min(sample_size, n), replace=False)
            picked = np.zeros((len(pick), len(k), p))
            for sl in batches(n, batch_size):
                inside = (pick >= sl.start) & (pick < sl.stop)
                if inside.any():
                    covs = compute_multiscale_covariance(expr, global_mean, knn[sl], k, cov_type, exclude_target)
                    picked[inside] = covs[pick[inside] - sl.start]
            mean_covs = picked.mean(axis=0)                    # in the order of the sample
        elif weight_strategy == "global":
            total = np.zeros((len(k), p))
            for sl in batches(n, batch_size):
                for cell in compute_multiscale_covariance(expr, global_mean, knn[sl], k, cov_type, exclude_target):
                    total += cell              # cell by cell, the order numpy's mean uses
            mean_covs = total / n
        else:
            raise ValueError(f"weight_strategy must be 'sampled' or 'global', not {weight_strategy!r}")
        weights = optimize_weights(mean_covs, entropy_reg)

    features = np.zeros((n, p), dtype=dtype)
    for sl in batches(n, batch_size):
        covs = compute_multiscale_covariance(expr, global_mean, knn[sl], k, cov_type, exclude_target)
        fused = covs[:, 0, :] if len(k) == 1 else np.einsum("i,nij->nj", weights, covs)
        features[sl] = feature(apply_regularization(fused, g, alpha), g, feature_transform, n_jobs)
    return features, weights
