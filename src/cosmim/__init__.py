"""CoSMIM: Covariance of Spatial Multi-scale Integrated Microenvironment.

Steps: spatial_preprocess_pipeline -> compute_cosmim -> run_leiden_partition -> calc_all_metrics; run() does all of them for one section.
"""
from .cluster_utils import find_resolution, get_num_groups, run_leiden_partition
from .matrix_ops import eigh_logm, eigh_sqrt, feature, from_tril, to_tril
from .metrics_utils import calc_all_metrics, calc_ari, calc_asw, calc_chaos, calc_com, calc_hom, calc_moranI_gearyC, calc_nmi, calc_pas
from .niche_covariance import apply_regularization, build_knn_index, compute_cosmim, compute_multiscale_covariance, optimize_weights
from .pipeline import run
from .preprocess import compute_svg, preprocess_sample_spatial, select_hvg_spatial, spatial_preprocess_pipeline

__version__ = "1.0.1"

__all__ = ["preprocess_sample_spatial", "compute_svg", "select_hvg_spatial", "spatial_preprocess_pipeline", "build_knn_index", "compute_multiscale_covariance", "apply_regularization", "optimize_weights", "compute_cosmim", "eigh_sqrt", "eigh_logm", "feature", "to_tril", "from_tril", "find_resolution", "run_leiden_partition", "get_num_groups", "calc_ari", "calc_nmi", "calc_hom", "calc_com", "calc_chaos", "calc_pas", "calc_asw", "calc_moranI_gearyC", "calc_all_metrics", "run"]
