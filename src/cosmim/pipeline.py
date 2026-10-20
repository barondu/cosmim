"""CoSMIM on one section in one call."""
from .cluster_utils import get_num_groups, run_leiden_partition
from .metrics_utils import calc_ari, calc_nmi, labelled_cells
from .niche_covariance import compute_cosmim
from .preprocess import spatial_preprocess_pipeline


def run(adata, annotation_key=None, max_labels=None, strategy="svg_hvg", n_top_hvg=64, k=(5, 15, 45), entropy_reg=10, alpha=0.1, n_pcs=30, n_neighbors=50, random_state=1234, **cosmim_options):
    """Preprocess, compute the CoSMIM feature and cluster into `max_labels` clusters (default: the number of domains in obs[annotation_key]). Other keyword arguments go to compute_cosmim.

    Returns the preprocessed AnnData with obs['cosmim'] (clusters), obsm['cosmim'] (feature) and uns['cosmim'] (genes, weights, resolution, and NMI and ARI if annotated).
    """
    adata_raw, adata_hvg = spatial_preprocess_pipeline(adata, strategy=strategy, n_top_hvg=n_top_hvg)
    features, weights = compute_cosmim(adata_hvg, k=k, entropy_reg=entropy_reg, alpha=alpha, **cosmim_options)
    if max_labels is None:
        if annotation_key is None:
            raise ValueError("give max_labels or annotation_key")
        max_labels = get_num_groups(adata_raw, annotation_key)
    labels, resolution = run_leiden_partition(features, max_labels=max_labels, n_pcs=n_pcs, n_neighbors=n_neighbors, random_state=random_state, return_resolution=True)
    adata_raw.obs["cosmim"] = labels
    adata_raw.obsm["cosmim"] = features
    adata_raw.uns["cosmim"] = {"genes": adata_hvg.var_names.tolist(), "weights": weights, "resolution": resolution}
    if annotation_key is not None:
        labelled = labelled_cells(adata_raw, annotation_key)
        adata_raw.uns["cosmim"]["NMI"] = calc_nmi(adata_raw[labelled], annotation_key, "cosmim")
        adata_raw.uns["cosmim"]["ARI"] = calc_ari(adata_raw[labelled], annotation_key, "cosmim")
    return adata_raw
