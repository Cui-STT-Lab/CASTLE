# CASTLE

CASTLE: Cell-type Aware SpaTial domain detection via contrastive Learning Embedding

# Overview

<img width="2996" height="1583" alt="overview" src="https://github.com/user-attachments/assets/8a058ccf-8b04-4fca-8979-bfa8e32c2f7d" />

CASTLE integrate spatial proximity and cell-type information directly during graph construction, using deconvolved compositions for spot-resolution data or cell-level annotations for single-cell platforms, with a fallback to expression-based similarity when labels are uncertain or unavailable. A self-supervised, local-context contrastive objective learns embeddings aligned to each spot’s microenvironment, while a lightweight encoder-decoder reconstructs expression to regularize the representation.

# Installation

CASTLE has been implemented and tested with Python 3.11.13 and torch 2.7.1.

```
git clone https://github.com/Cui-STT-Lab/CASTLE.git
cd CASTLE/
pip install -r requirements.txt
```

The default mclust clustering option also requires the R package `mclust`, which is called from Python through `rpy2`.

```
R -e "install.packages('mclust', repos='https://cloud.r-project.org')"
```

# Datasets

All datasets used in our paper can be found in:

- The DLPFC dataset is accessible within the spatialLIBD package (http://spatial.libd.org/spatialLIBD).
- Data and H&E images for MOB are available for download at https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/.
- The human breast cancer 10x Visium dataset is accessible at https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0.
- The MERFISH dataset of the mouse hypothalamic preoptic area (MPOA) and mouse medial prefrontal cortex data from STARmap are available at http://sdmbench.drai.cn/.
- The Slide-seq dataset is available at https://portals.broadinstitute.org/single_cell/study/slide-seq-study.
- The processed Stereo-seq data from mouse olfactory bulb tissue is accessible at https://github.com/JinmiaoChenLab/SEDR_analyses.
- The human breast cancer Xenium dataset is available from 10x Genomics at https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip. The corresponding supervised cell type/state annotations are available from https://github.com/10XGenomics/janesick_nature_comms_2023_companion.
- The human colorectal cancer Visium HD dataset is available from 10x Genomics at https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer/Visium_HD_Human_Colon_Cancer_binned_outputs.tar.gz. The corresponding pathologist-provided spatial domain annotations are available from Zenodo at https://zenodo.org/records/11402686.

# Usage

## DLPFC

The DLPFC example uses a Visium count matrix, a cell-type proportion table, and layer labels. In the code below, `csv_file` should point to a CSV file whose rows are spot barcodes and whose columns are cell types. `df_meta_layer` is the layer annotation for the same spots, for example the `layer_guess_reordered_short` column from the spatialLIBD metadata.

```
import torch
import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from CASTLE.CASTLE import CASTLE

index = 8

adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()
adata

# Example metadata loading. Update metadata_file and layer_col for your DLPFC slice.
metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)
layer_col = 'layer_guess_reordered_short'
adata.obs['ground_truth'] = metadata.loc[adata.obs_names, layer_col]
df_meta_layer = adata.obs['ground_truth']

norm_weights = pd.read_csv(csv_file, index_col=0)
print(norm_weights)
adata = adata[adata.obs_names.isin(norm_weights.index)].copy()
norm_weights = norm_weights.loc[adata.obs_names]
adata.obsm['cell_type'] = norm_weights.values
df_meta_layer = adata.obs['ground_truth']
n_clusters = df_meta_layer.dropna().unique().shape[0]

model = CASTLE(adata, device=device,datatype = '10X', mode='sim', k_celltype=5, sim_threshold=0.9, cell_proportions='cell_type', weight1=True, weight2=True)
adata = model.train()

radius = 30
tool = 'mclust' # mclust, leiden, and louvain
# clustering
from CASTLE.utils import clustering
clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)

import matplotlib.pyplot as plt

adata = adata[~pd.isnull(adata.obs['ground_truth'])]
sc.pl.spatial(adata,
              img_key="hires",
              color=["ground_truth", "domain"],
              title=["Ground truth", "ARI=%.4f"%ARI(adata.obs['domain'], adata.obs['ground_truth'])],
              show=True)

```

## Merfish data

For MERFISH `.h5ad` files, CASTLE expects spatial coordinates in `adata.obsm['spatial']` and the cell-type labels used for graph construction in `adata.obs['cell_class']`. If ground-truth labels are available, store them in `adata.obs['ground_truth']` before plotting or computing ARI.

```
import torch
import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from CASTLE.CASTLE import CASTLE

index = 4

adata = sc.read_h5ad(file_fold)
adata.var_names_make_unique()
adata

required_obs = 'cell_class'
required_obsm = 'spatial'
if required_obs not in adata.obs:
    raise KeyError(f"MERFISH input must contain adata.obs['{required_obs}'].")
if required_obsm not in adata.obsm:
    raise KeyError(f"MERFISH input must contain adata.obsm['{required_obsm}'].")

model = CASTLE(adata, device=device, datatype = 'cell', mode='celltype', k_celltype = 10, weight1=False, weight2=False, cell_type='cell_class')
adata = model.train()

radius = 50
n_clusters = 8 # the number of clusters
tool = 'mclust' # mclust, leiden, and louvain
# clustering
from CASTLE.utils import clustering
clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)

import matplotlib.pyplot as plt
sc.pl.spatial(adata,
              color=["ground_truth","domain"],
              title=["Ground truth", "ARI=%.4f"%ARI(adata.obs['domain'], adata.obs['ground_truth'])],
              spot_size=20,
              basis='spatial')

```
