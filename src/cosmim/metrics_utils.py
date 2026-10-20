"""The nine evaluation metrics of SDMBench (Yuan et al. 2024)."""
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

UNLABELED = {"nan", "none", "na", "unknown", ""}
MAX_ENTRIES = 4_000_000   # size of one block of the distance matrix


def labelled_cells(adata, key):
    """Boolean mask of the cells with an annotation in obs[key] (not NaN, 'nan', 'none', 'na', 'unknown' or empty)."""
    gt = adata.obs[key]
    return (~gt.isna() & ~gt.astype(str).str.strip().str.lower().isin(UNLABELED)).values


def calc_ari(adata, gt_key, pred_key):
    """Adjusted Rand index."""
    return adjusted_rand_score(adata.obs[gt_key].astype("category").cat.codes, adata.obs[pred_key].astype("category").cat.codes)


def calc_nmi(adata, gt_key, pred_key):
    """Normalised mutual information."""
    return normalized_mutual_info_score(adata.obs[gt_key].astype("category").cat.codes, adata.obs[pred_key].astype("category").cat.codes)


def calc_hom(adata, gt_key, pred_key):
    """Homogeneity."""
    return float(homogeneity_score(adata.obs[gt_key], adata.obs[pred_key]))


def calc_com(adata, gt_key, pred_key):
    """Completeness."""
    return float(completeness_score(adata.obs[gt_key], adata.obs[pred_key]))


def calc_chaos(adata, pred_key, spatial_key="spatial"):
    """CHAOS (lower is better): mean distance from a cell to the nearest cell of its cluster, on standardised coordinates. Clusters of at most two cells are skipped."""
    labels = np.array(adata.obs[pred_key])
    location = StandardScaler().fit_transform(np.array(adata.obsm[spatial_key]))
    clusters = np.unique(labels)
    dist_val = np.zeros(len(clusters))
    count = 0
    for label in clusters:
        points = location[labels == label, :]
        n = len(points)
        if n <= 2:
            continue
        nearest = []
        step = max(1, MAX_ENTRIES // n)
        for start in range(0, n, step):
            rows = np.arange(start, min(start + step, n))
            d = distance_matrix(points[rows], points)
            d[np.arange(len(rows)), rows] = np.inf
            nearest.append(d.min(axis=1))
        dist_val[count] = np.sum(np.concatenate(nearest))
        count += 1
    return np.sum(dist_val) / len(labels)


def calc_pas(adata, pred_key, spatial_key="spatial", k=10):
    """PAS (lower is better): fraction of cells with more than half of their k nearest cells in another cluster."""
    labels = np.array(adata.obs[pred_key])
    location = np.array(adata.obsm[spatial_key])
    n = len(location)
    abnormal = []
    step = max(1, MAX_ENTRIES // n)
    for start in range(0, n, step):
        rows = np.arange(start, min(start + step, n))
        d = distance_matrix(location[rows], location)
        d[np.arange(len(rows)), rows] = np.inf
        nearest = np.argsort(d, axis=1)[:, :k]
        abnormal.append((labels[nearest] != labels[rows][:, None]).sum(axis=1) > k / 2)
    return np.sum(np.concatenate(abnormal)) / len(labels)


def calc_asw(adata, pred_key, spatial_key="spatial", sample_size=None, random_state=0):
    """Average silhouette width in space, in [-1, 1]; `sample_size` cells for large sections."""
    location = np.asarray(adata.obsm[spatial_key], dtype=float)
    labels = np.asarray(adata.obs[pred_key])
    if sample_size is not None and sample_size < len(labels):
        idx = np.random.default_rng(random_state).choice(len(labels), size=sample_size, replace=False)
        location, labels = location[idx], labels[idx]
    return float(silhouette_score(X=squareform(pdist(location)), labels=labels, metric="precomputed"))


def calc_moranI_gearyC(adata_counts, pred_key, top_n=5, n_perms=100, n_jobs=1, spatial_key="spatial"):
    """Median Moran's I and Geary's C of the clusters' marker genes (squidpy)."""
    try:
        import squidpy as sq
    except ImportError as err:
        raise ImportError("Moran's I and Geary's C need squidpy: pip install cosmim[metrics]") from err
    adata = adata_counts.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        adata.uns.pop("log1p", None)
    if not isinstance(adata.obs[pred_key].dtype, pd.CategoricalDtype):
        adata.obs[pred_key] = adata.obs[pred_key].astype("category")
    sizes = adata.obs[pred_key].value_counts()
    adata = adata[adata.obs[pred_key].isin(sizes.index[sizes > 3].tolist())].copy()
    adata.X = adata.X.astype(np.float64) if sparse.issparse(adata.X) else np.asarray(adata.X, dtype=np.float64)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby=pred_key)
    genes = []
    for i in range(top_n):
        genes.extend(list(adata.uns["rank_genes_groups"]["names"][i]))
    genes = np.unique(genes).tolist()
    perms = None if (n_perms is None or n_perms <= 0) else n_perms
    sq.gr.spatial_neighbors(adata, spatial_key=spatial_key)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=genes, n_perms=perms, n_jobs=n_jobs)
    sq.gr.spatial_autocorr(adata, mode="geary", genes=genes, n_perms=perms, n_jobs=n_jobs)
    return float(np.median(adata.uns["moranI"]["I"])), float(np.median(adata.uns["gearyC"]["C"]))


def calc_all_metrics(adata, gt_key, pred_key, adata_counts=None, spatial_key="spatial", asw_sample_size=None, moranI_top_n=5, moranI_n_perms=100, n_jobs=1, skip_moran=False):
    """ARI, NMI, HOM, COM, CHAOS, PAS, ASW, MoranI, GearyC as a dict.

    ARI, NMI, HOM and COM use only the annotated cells. ASW is rescaled to (ASW + 1) / 2. Moran's I and Geary's C use the counts of `adata_counts` (default: `adata`).
    """
    labelled = labelled_cells(adata, gt_key)
    adata_sup = adata if labelled.all() else adata[labelled]
    if not labelled.all():
        warnings.warn(f"{(~labelled).sum()} of {adata.n_obs} cells have no {gt_key!r} label; ARI, NMI, HOM and COM use the labelled cells only")
    metrics = {"ARI": calc_ari(adata_sup, gt_key, pred_key), "NMI": calc_nmi(adata_sup, gt_key, pred_key), "HOM": calc_hom(adata_sup, gt_key, pred_key), "COM": calc_com(adata_sup, gt_key, pred_key), "CHAOS": calc_chaos(adata, pred_key, spatial_key), "PAS": calc_pas(adata, pred_key, spatial_key), "ASW": (calc_asw(adata, pred_key, spatial_key, asw_sample_size) + 1) / 2}
    if skip_moran:
        metrics["MoranI"], metrics["GearyC"] = float("nan"), float("nan")
    else:
        metrics["MoranI"], metrics["GearyC"] = calc_moranI_gearyC(adata if adata_counts is None else adata_counts, pred_key, top_n=moranI_top_n, n_perms=moranI_n_perms, n_jobs=n_jobs, spatial_key=spatial_key)
    return metrics
