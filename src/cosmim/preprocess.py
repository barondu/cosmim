"""Quality control, normalisation and gene selection."""
import scanpy as sc


def preprocess_sample_spatial(adata, min_counts=5, min_cells=3, target_sum=1e4):
    """QC, normalisation and log1p, on a copy. The counts are kept in layers['counts'] and the log values in layers['log']."""
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers["log"] = adata.X.copy()
    return adata


def compute_svg(adata, threshold=0.05, n_neighbors=15):
    """Spatially variable genes: Moran's I above `threshold`."""
    sc.pp.neighbors(adata, use_rep="spatial", n_neighbors=n_neighbors)
    adata.var["morans_i"] = sc.metrics.morans_i(adata)
    return adata.var_names[adata.var["morans_i"] > threshold].tolist()


def select_hvg_spatial(adata, n_top_genes, layer=None, flavor="seurat"):
    """The `n_top_genes` most highly variable genes, as a copy."""
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, layer=layer, flavor=flavor)
    return adata[:, adata.var["highly_variable"]].copy()


def spatial_preprocess_pipeline(adata, strategy="svg_hvg", n_top_hvg=64, morans_i_threshold=0.05, svg_n_neighbors=15, min_counts=5, min_cells=3, target_sum=1e4):
    """Preprocess a section and choose the genes of the feature. Returns (adata_raw, adata_hvg).

    strategy: 'svg_hvg' (spatially variable genes, then the most highly variable of them), 'svg_only' (the largest Moran's I) or 'hvg_only'.
    """
    if strategy not in ("svg_hvg", "svg_only", "hvg_only"):
        raise ValueError(f"strategy must be 'svg_hvg', 'svg_only' or 'hvg_only', not {strategy!r}")
    adata_raw = preprocess_sample_spatial(adata, min_counts=min_counts, min_cells=min_cells, target_sum=target_sum)
    if strategy == "hvg_only":
        return adata_raw, select_hvg_spatial(adata_raw, n_top_genes=n_top_hvg)
    svg = compute_svg(adata_raw, threshold=morans_i_threshold, n_neighbors=svg_n_neighbors)
    if strategy == "svg_hvg":
        return adata_raw, select_hvg_spatial(adata_raw[:, svg].copy(), n_top_genes=n_top_hvg)
    top = adata_raw.var.loc[svg, "morans_i"].nlargest(n_top_hvg).index.tolist()
    adata_hvg = adata_raw[:, top].copy()
    adata_hvg.var["highly_variable"] = True
    return adata_raw, adata_hvg
