# CoSMIM

**Co**variance of **S**patial **M**ulti-scale **I**ntegrated **M**icroenvironment: a training-free representation for spatial domain identification.

Many spatial domain methods describe the neighbourhood of a cell by its mean gene expression. CoSMIM describes it by the gene–gene covariance of the neighbourhood at several spatial scales (the cell and its 5, 15 and 45 nearest neighbours). The scales are fused with weights learned for each section, the fused matrix is shrunk towards a scaled identity, and the matrix square root is used as a Euclidean feature. Standard PCA and Leiden clustering of this feature give the spatial domains; nothing is trained.

![Overview of CoSMIM](https://raw.githubusercontent.com/barondu/cosmim/main/figures/fig1/fig1.png)

Fig. 1: Overview of CoSMIM.

## Install

```bash
pip install "cosmim[metrics]"   # [metrics] adds squidpy, needed for Moran's I and Geary's C
```

To use the package versions the paper's results were computed with:

```bash
git clone https://github.com/barondu/cosmim.git
cd cosmim
conda env create -f environment.yml
conda activate cosmim
pip install -e .
```

CoSMIM needs no GPU.

## Quick start

```python
import scanpy as sc
import cosmim

adata = sc.read_h5ad("data/D4_STARmap_VisualCortex/STARmap_20180505_BY3_1k.h5ad")
result = cosmim.run(adata, annotation_key="Region", strategy="svg_only", n_top_hvg=64,
                    n_pcs=30, n_neighbors=200)
result.obs["cosmim"]                                           # the spatial domains
cosmim.calc_all_metrics(result, gt_key="Region", pred_key="cosmim")   # the nine metrics
```

`run` clusters a section into as many domains as `annotation_key` has annotated domains. On a section without an annotation, give the number of domains (`max_labels=...`), or call the steps directly and give a Leiden resolution:

```python
adata_raw, adata_hvg = cosmim.spatial_preprocess_pipeline(adata, strategy="svg_hvg", n_top_hvg=64)
features, weights = cosmim.compute_cosmim(adata_hvg, k=[5, 15, 45], entropy_reg=10, alpha=0.1)
adata_raw.obs["cosmim"] = cosmim.run_leiden_partition(features, resolution=1.0, n_pcs=30,
                                                      n_neighbors=50)
```

## Parameters

| Parameter              | Default (the paper)                                                  | Options                                               |
| ---------------------- | -------------------------------------------------------------------- | ----------------------------------------------------- |
| `strategy`             | `"svg_hvg"`: spatially variable genes, then the most highly variable | `"svg_only"` (largest Moran's I), `"hvg_only"`        |
| `n_top_hvg`            | 64                                                                   | number of genes of the feature                        |
| `k`                    | `[5, 15, 45]`                                                        | the scales (numbers of neighbours)                    |
| `cov_type`             | `"shift"`: centred on the mean of the section                        | `"traditional"`: on the mean of the niche             |
| `exclude_target`       | `False`                                                              | leave the cell out of its own niche                   |
| `entropy_reg`          | 10                                                                   | λ, the entropy term of the scale weights              |
| `weight_strategy`      | `"sampled"`: weights learned on 1,000 cells                          | `"global"`: on all cells                              |
| `weights`              | learned                                                              | fixed weights, for example `[1/3, 1/3, 1/3]`          |
| `alpha`                | 0.1                                                                  | shrinkage towards a scaled identity                   |
| `feature_transform`    | `"sqrt"`: the matrix square root                                     | `"logm"` (log-Euclidean), `"raw"` (no map)            |
| `dtype`                | `float64`                                                            | `float32` halves the memory of the returned feature   |
| `n_pcs`, `n_neighbors` | 30, 50                                                               | PCA components and graph neighbours of the clustering |

## Tutorials

| Notebook                                                                     | Data                                                                             | What it shows                                                   |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [1_dataset4_STARmap](https://github.com/barondu/cosmim/blob/main/notebooks/1_dataset4_STARmap.ipynb)                     | dataset4, STARmap mouse visual cortex, 1 section             | CoSMIM step by step and the nine metrics                        |
| [2_dataset1_MERFISH_benchmark](https://github.com/barondu/cosmim/blob/main/notebooks/2_dataset1_MERFISH_benchmark.ipynb) | dataset1, MERFISH mouse hypothalamus, 5 sections          | CoSMIM on the five sections, the benchmark of sixteen methods and its statistical tests, gene–gene correlations across domains |
| [3_dataset6_StereoSeq](https://github.com/barondu/cosmim/blob/main/notebooks/3_dataset6_StereoSeq.ipynb)                     | dataset6, Stereo-seq mouse embryo E10.5_E1S1 and E10.5_E2S1 | the liver and the brain found by each method, and the niche covariance of the liver |

The notebooks were run with the versions in `environment.yml` on Linux (aarch64). On other machines and library builds a few cells can change cluster, so the scores can differ from the paper.

## Benchmark

![Benchmark of 16 methods](https://raw.githubusercontent.com/barondu/cosmim/main/figures/fig2/fig2.png)

Fig. 2: Benchmarking of 16 spatial domain identification methods on six datasets (22 sections). Sixteen methods on dataset1–6, scored with eight metrics. CoSMIM has the highest overall score (summed rank 121 of 128; the second method has 106). Tutorial 2 computes the ranks and the statistical tests from the per-section scores.

## Data

The seven datasets of the paper. Datasets 1–6 are those distributed with SDMBench (Yuan *et al.*, *Nat. Methods* 21, 712–722 (2024)), whose published scores of eleven methods are part of the benchmark table.

|          | Platform   | Tissue                                         | Reference                                                                                                                                              |
| -------- | ---------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| dataset1 | MERFISH    | mouse hypothalamic preoptic region, 5 sections | Moffitt *et al.*, *Science* 362, eaau5324 (2018); domain labels of Li & Zhou, *Genome Biol.* 23, 168 (2022)                                            |
| dataset2 | osmFISH    | mouse somatosensory cortex, 1 section          | Codeluppi *et al.*, *Nat. Methods* 15, 932–935 (2018)                                                                                                  |
| dataset3 | BaristaSeq | mouse visual cortex, 3 sections                | Chen *et al.*, *Nucleic Acids Res.* 46, e22 (2018); Long *et al.*, arXiv:2301.08436 (2023)                                                             |
| dataset4 | STARmap    | mouse visual cortex, 1 section                 | Wang *et al.*, *Science* 361, eaat5691 (2018); second annotation (`smoothed_manual`, tutorial 1) of Singhal *et al.*, *Nat. Genet.* 56, 431–441 (2024) |
| dataset5 | STARmap    | mouse medial prefrontal cortex, 3 sections     | Wang *et al.*, *Science* 361, eaat5691 (2018)                                                                                                          |
| dataset6 | Stereo-seq | mouse embryo E9.5 and E10.5, 9 sections        | Chen *et al.*, *Cell* 185, 1777–1792 (2022)                                                                                                            |
| dataset7 | Xenium     | mouse brain, 3 sections                        | 10x Genomics; region annotation of Bhuva *et al.*, *Genome Biol.* 25, 99 (2024)                                                                        |

## Citation

The preprint is on bioRxiv: <https://doi.org/10.64898/2026.10.20.760373>

```bibtex
@article{du2026cosmim,
  title   = {Multi-scale microenvironment covariance for unsupervised spatial domain identification},
  author  = {Du, Bairui and Kiryu, Hisanori},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.20.760373}
}
```

## License

The code is under the BSD 3-Clause License (see [LICENSE](https://github.com/barondu/cosmim/blob/main/LICENSE)). The data files keep the terms of their sources listed above; the figures are those of the paper.