"""PCA, a neighbour graph and Leiden clustering."""
import anndata as ad
import numpy as np
import scanpy as sc


def find_resolution(adata, target_labels, key_added="leiden", starting_resolution=0.6, max_iter=60, random_state=1234):
    """The Leiden resolution that gives `target_labels` clusters, or the closest one found."""
    resolution = starting_resolution
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, random_state=random_state, flavor="igraph", n_iterations=2)
    num_labels = adata.obs[key_added].nunique()
    if num_labels == target_labels:
        return resolution
    best_resolution, best_diff = resolution, abs(num_labels - target_labels)

    iteration = 0
    for step in (0.1, 0.05, 0.01):
        if iteration >= max_iter:
            break
        direction = 1 if num_labels < target_labels else -1
        while iteration < max_iter:
            resolution = max(round(resolution + direction * step, 4), 1e-4)
            iteration += 1
            sc.tl.leiden(adata, resolution=resolution, key_added=key_added, random_state=random_state, flavor="igraph", n_iterations=2)
            num_labels = adata.obs[key_added].nunique()
            if abs(num_labels - target_labels) < best_diff:
                best_resolution, best_diff = resolution, abs(num_labels - target_labels)
            if num_labels == target_labels:
                return resolution
            # past the target: go back with a smaller step
            if (1 if num_labels < target_labels else -1) != direction:
                break

    if best_diff > 0:
        print(f"find_resolution: no resolution gives {target_labels} clusters; using {best_resolution:.4f}, which is {best_diff} away")
    return best_resolution


def run_leiden_partition(features, max_labels=None, resolution=None, n_pcs=30, n_neighbors=50, random_state=1234, return_resolution=False):
    """PCA, a kNN graph and Leiden, with `max_labels` clusters or at `resolution`. Returns the labels (and the resolution if `return_resolution`)."""
    if (max_labels is None) == (resolution is None):
        raise ValueError("give exactly one of max_labels and resolution")
    adata = ad.AnnData(X=np.asarray(features))
    n_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep="X_pca", random_state=random_state)
    if resolution is None:
        resolution = find_resolution(adata, max_labels, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, flavor="igraph", n_iterations=2)
    labels = adata.obs["leiden"].values
    return (labels, resolution) if return_resolution else labels


def get_num_groups(adata, annotation_key):
    """Number of annotated domains."""
    if annotation_key not in adata.obs:
        raise KeyError(f"{annotation_key!r} not found in adata.obs")
    return adata.obs[annotation_key].nunique()
