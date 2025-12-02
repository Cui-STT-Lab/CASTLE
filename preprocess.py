#This code has been partly adapted from https://github.com/JinmiaoChenLab/GraphST

import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.preprocessing import normalize

def filter_with_overlap_gene(adata, adata_sc):
    # remove all-zero-valued genes
    #sc.pp.filter_genes(adata, min_cells=1)
    #sc.pp.filter_genes(adata_sc, min_cells=1)
    
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    
    return adata, adata_sc

def permutation(feature, meta=None):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    if meta is not None:
        meta_permutated = meta[ids]
        return feature_permutated, meta_permutated
    return feature_permutated


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj


def construct_interaction_celltype(
    adata,
    mode='pca_cosine_knn',          # 'spatial' | 'celltype' | 'pca_cosine_knn' 0.4-0.6 | 'snn_jaccard' 0.01 | 'adaptive_rbf' 0.2-0.3
    cell_type=None,                 # required for 'celltype'
    # spatial candidates
    k_spatial=6,                    # used by spatial + all similarity modes
    k_celltype=6,                   # only for 'celltype'
    # similarity backends (work on adata.obsm['feat'] / ['feat_a'])
    expr_k=15,                      # k for expression-space kNN (SNN / sigma for RBF)
    metric='cosine',                # 'cosine' or 'euclidean' for expression-space kNN
    sim_threshold=0.5,              # keep edge if sim >= threshold
    weight1=True,                   # graph_neigh(*): True=keep weights; False=binary
    weight2=True,                   # adj(*):        True=keep weights; False=binary
    symmetrize='max'                # 'max' | 'mean'
):
    """
    Writes (all as CSR into obsm):
      - graph_neigh , adj
      - graph_neigh_a, adj_a  (only if the needed inputs exist)
    """
    # -------- helpers --------
    def _to_binary_csr(M):
        M = M if sp.issparse(M) else sp.csr_matrix(M)
        B = M.tocsr(copy=True)
        if B.nnz:
            B.data[:] = 1.0
        return B
    def _symmetrize(M, how='max'):
        M = M if sp.issparse(M) else sp.csr_matrix(M)
        if how == 'mean':
            S = (M + M.T) * 0.5
        else:
            S = M.maximum(M.T)
        return S.tocsr()
    def _spatial_candidates(coords, k):
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
        _, knn = nbrs.kneighbors(coords, return_distance=True)  # knn[:,0] is self
        return knn
    def _build_celltype_graph(labels, knn_idxs, topk):
        n = len(labels)
        rows, cols = [], []
        for i in range(n):
            js = knn_idxs[i, 1:topk+1]
            keep = [j for j in js if labels[i] == labels[j]]
            rows.extend([i]*len(keep)); cols.extend(keep)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    # --- similarity workers (score only the k_spatial spatial neighbors) ---
    def _sim_cosine_on_feats(F, knn_spatial, k_spatial, thresh):
        # cosine via L2-normalized dot
        Fn = normalize(F, norm='l2', axis=1, copy=True)
        n = Fn.shape[0]
        rows, cols, vals = [], [], []
        for i in range(n):
            js = knn_spatial[i, 1:k_spatial+1]
            sim = Fn[i, None] @ Fn[js].T        # shape (1, k_spatial)
            sim = np.asarray(sim).ravel()
            keep = sim >= thresh
            if np.any(keep):
                rows.extend([i]*keep.sum())
                cols.extend(js[keep].tolist())
                vals.extend(sim[keep].tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    def _sim_snn_jaccard_on_feats(F, knn_spatial, k_spatial, expr_k, metric, thresh):
        n = F.shape[0]
        # build kNN in feature space
        nn = NearestNeighbors(n_neighbors=expr_k, metric=metric).fit(F)
        _, knn_expr = nn.kneighbors(F, return_distance=True)
        # binary kNN matrix
        r, c = [], []
        for i in range(n):
            js = knn_expr[i]
            r.extend([i]*len(js)); c.extend(js.tolist())
        knn_bin = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))
        inter = knn_bin.dot(knn_bin.T).tocsr()        # |N(i) ∩ N(j)|
        deg = np.asarray(knn_bin.sum(axis=1)).ravel()
        rows, cols, vals = [], [], []
        for i in range(n):
            js = knn_spatial[i, 1:k_spatial+1]
            row_i = inter.getrow(i)
            inter_map = {col: val for col, val in zip(row_i.indices, row_i.data)}
            sims = []
            for j in js:
                ij_inter = inter_map.get(j, 0.0)
                ij_union = deg[i] + deg[j] - ij_inter
                s = (ij_inter / ij_union) if ij_union > 0 else 0.0
                sims.append(s)
            sims = np.asarray(sims)
            keep = sims >= thresh
            if np.any(keep):
                rows.extend([i]*keep.sum())
                cols.extend(js[keep].tolist())
                vals.extend(sims[keep].tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    def _sim_adaptive_rbf_on_feats(F, knn_spatial, k_spatial, expr_k, thresh):
        # local scales from expr-space kNN (euclidean)
        nn = NearestNeighbors(n_neighbors=expr_k, metric='euclidean').fit(F)
        dists_expr, _ = nn.kneighbors(F, return_distance=True)
        sigma = dists_expr[:, -1] + 1e-8
        n = F.shape[0]
        rows, cols, vals = [], [], []
        for i in range(n):
            js = knn_spatial[i, 1:k_spatial+1]
            diffs = F[js] - F[i]                   # (k_spatial, d)
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            ws = np.exp(-(d2) / (sigma[i] * sigma[js]))
            keep = ws >= thresh
            if np.any(keep):
                rows.extend([i]*keep.sum())
                cols.extend(js[keep].tolist())
                vals.extend(ws[keep].tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # -------- inputs & spatial candidates --------
    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required.")
    coords = np.asarray(adata.obsm['spatial'], dtype=float)
    n = coords.shape[0]
    knn_sp = _spatial_candidates(coords, max(k_spatial, k_celltype))
    # -------- per-mode build --------
    if mode == 'spatial':
        rows, cols, vals = [], [], []
        for i in range(n):
            js = knn_sp[i, 1:k_spatial+1]
            rows.extend([i]*len(js)); cols.extend(js.tolist()); vals.extend([1.0]*len(js))
        M = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        graph_neigh = (M if weight1 else _to_binary_csr(M))
        adata.obsm['graph_neigh'] = graph_neigh
        adj = _symmetrize(M, how=symmetrize)
        adata.obsm['adj'] = (adj if weight2 else _to_binary_csr(adj))
        return
    if mode == 'celltype':
        if (cell_type is None) or (cell_type not in adata.obs.columns):
            raise ValueError("mode='celltype' requires a valid 'cell_type' in adata.obs.")
        # main labels
        ct = np.asarray(adata.obs[cell_type])
        M = _build_celltype_graph(ct, knn_sp, k_celltype)
        graph_neigh = (M if weight1 else _to_binary_csr(M))
        adata.obsm['graph_neigh'] = graph_neigh
        adj = _symmetrize(M, how=symmetrize)
        adata.obsm['adj'] = (adj if weight2 else _to_binary_csr(adj))
        # augmented labels (cell_type_a), if present
        ct_a_key = f"{cell_type}_a"
        if ct_a_key in adata.obs.columns:
            ct_a = np.asarray(adata.obs[ct_a_key])
            M_a = _build_celltype_graph(ct_a, knn_sp, k_celltype)
            graph_neigh_a = (M_a if weight1 else _to_binary_csr(M_a))
            adata.obsm['graph_neigh_a'] = graph_neigh_a
            adj_a = _symmetrize(M_a, how=symmetrize)
            adata.obsm['adj_a'] = (adj_a if weight2 else _to_binary_csr(adj_a))
        return
    # similarity modes: use adata.obsm['feat'] / ['feat_a']
    if 'feat' not in adata.obsm:
        raise ValueError("For similarity modes, adata.obsm['feat'] is required.")
    F = np.asarray(adata.obsm['feat'], dtype=float)
    if mode == 'pca_cosine_knn':
        M = _sim_cosine_on_feats(F, knn_sp, k_spatial, sim_threshold)
    elif mode == 'snn_jaccard':
        M = _sim_snn_jaccard_on_feats(F, knn_sp, k_spatial, expr_k, metric, sim_threshold)
    elif mode == 'adaptive_rbf':
        M = _sim_adaptive_rbf_on_feats(F, knn_sp, k_spatial, expr_k, sim_threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # write main
    graph_neigh = (M if weight1 else _to_binary_csr(M))
    adata.obsm['graph_neigh'] = graph_neigh
    adj = _symmetrize(M, how=symmetrize)
    adata.obsm['adj'] = (adj if weight2 else _to_binary_csr(adj))
    # write _a if available
    if 'feat_a' in adata.obsm:
        Fa = np.asarray(adata.obsm['feat_a'], dtype=float)
        if mode == 'pca_cosine_knn':
            M_a = _sim_cosine_on_feats(Fa, knn_sp, k_spatial, sim_threshold)
        elif mode == 'snn_jaccard':
            M_a = _sim_snn_jaccard_on_feats(Fa, knn_sp, k_spatial, expr_k, metric, sim_threshold)
        elif mode == 'adaptive_rbf':
            M_a = _sim_adaptive_rbf_on_feats(Fa, knn_sp, k_spatial, expr_k, sim_threshold)
        graph_neigh_a = (M_a if weight1 else _to_binary_csr(M_a))
        adata.obsm['graph_neigh_a'] = graph_neigh_a
        adj_a = _symmetrize(M_a, how=symmetrize)
        adata.obsm['adj_a'] = (adj_a if weight2 else _to_binary_csr(adj_a))


def construct_interaction_celltype_similarity(
    adata,
    k_spatial=3,
    k_celltype=6,
    sim_threshold=0.8,
    mode='sim',                 # 'spatial' | 'sim' | 'hybrid'
    cell_proportions=None,      # key from obs/obsm
    weight1=False,              # graph_neigh if keep weight edge(False=binary)
    weight2=False,              # adj if keep weight edge(False=binary)
    sim_use_weight=True,        
    symmetrize='max'            # 'max' or 'mean'
):
    # ---------- Utility functions ----------
    def _to_binary_csr(M):
        """Convert any sparse/dense matrix to binary CSR."""
        if not sp.issparse(M):
            M = sp.csr_matrix(M)
        B = M.tocsr(copy=True)
        if B.nnz > 0:
            B.data[:] = 1.0
        return B
    def _symmetrize(M, how='max'):
        """Symmetrize to CSR."""
        if not sp.issparse(M):
            M = sp.csr_matrix(M)
        if how == 'mean':
            S = (M + M.T) * 0.5
        else:  # 'max'
            S = M.maximum(M.T)
        return S.tocsr()
    def _build_interaction(CP, distance_matrix):
        """
        Given a cell proportions matrix CP (n_spots x n_celltypes),
        construct interaction (sparse CSR) according to the current mode.
        """
        n_spot = CP.shape[0]
        sim_mat = cosine_similarity(CP) if CP is not None else None
        rows, cols, vals = [], [], []
        for i in range(n_spot):
            sorted_idx = np.argsort(distance_matrix[i])
            if mode == 'spatial':
                js = sorted_idx[1:k_spatial+1]
                rows.extend([i]*len(js)); cols.extend(js.tolist()); vals.extend([1.0]*len(js))
            elif mode == 'sim':
                if sim_mat is None:
                    raise ValueError("mode='sim' 需要 cell_proportions（或其 key/矩阵）。")
                js = sorted_idx[1:k_celltype+1]
                for j in js:
                    if i == j:
                        continue
                    s = sim_mat[i, j]
                    if s > sim_threshold:
                        w = float(s) if sim_use_weight else 1.0
                        rows.append(i); cols.append(j); vals.append(w)
            elif mode == 'hybrid':
                if sim_mat is None:
                    raise ValueError("mode='hybrid' requires cell_proportions (or its key/matrix).")
                # 1) Nearest k_spatial: must connect, weight=1
                js1 = sorted_idx[1:k_spatial+1]
                rows.extend([i]*len(js1)); cols.extend(js1.tolist()); vals.extend([1.0]*len(js1))
                # 2) (k_spatial, k_celltype]: connect if sim ≥ threshold
                js2 = sorted_idx[k_spatial+1:k_celltype+1]
                for j in js2:
                    if i == j:
                        continue
                    s = sim_mat[i, j]
                    if s > sim_threshold:
                        w = float(s) if sim_use_weight else 1.0
                        rows.append(i); cols.append(j); vals.append(w)
            else:
                raise ValueError("mode 必须是 'spatial'、'sim' 或 'hybrid'")
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_spot, n_spot))
    # ---------- Coordinates and distances ----------
    if 'spatial' not in adata.obsm_keys():
        raise ValueError("需要 adata.obsm['spatial']。")
    position = np.asarray(adata.obsm['spatial'], dtype=float)
    n_spot = position.shape[0]
    distance_matrix = ot.dist(position, position, metric='euclidean')
    adata.obsm['distance_matrix'] = distance_matrix 
    # ---------- cell_proportions ----------
    # Main CP
    CP = None
    if cell_proportions is not None:
        if isinstance(cell_proportions, str):
            if cell_proportions in adata.obsm_keys():
                CP = np.asarray(adata.obsm[cell_proportions], dtype=float)
            elif cell_proportions in adata.obs.columns:
                CP = np.asarray(adata.obs[cell_proportions], dtype=float)
            else:
                raise KeyError(f"找不到 '{cell_proportions}' 于 obsm/obs。")
        else:
            CP = np.asarray(cell_proportions, dtype=float)
    # Augmented CP_a (if exists)
    CP_a = None
    if 'cell_proportions_a' in adata.obsm_keys():
        CP_a = np.asarray(adata.obsm['cell_proportions_a'], dtype=float)
    # ---------- Main graph: interaction → graph_neigh / adj ----------
    interaction = _build_interaction(CP if mode != 'spatial' else np.zeros((n_spot,1)),
                                     distance_matrix)
    graph_neigh = (interaction if weight1 else _to_binary_csr(interaction))
    adata.obsm['graph_neigh'] = graph_neigh
    adj_sym = _symmetrize(interaction, how=symmetrize)
    adj = (adj_sym if weight2 else _to_binary_csr(adj_sym))
    adata.obsm['adj'] = adj
    # ---------- Augmented graph (if CP_a exists): interaction_a → graph_neigh_a / adj_a ----------
    if CP_a is not None:
        interaction_a = _build_interaction(CP_a if mode != 'spatial' else np.zeros((n_spot,1)),
                                           distance_matrix)
        graph_neigh_a = (interaction_a if weight1 else _to_binary_csr(interaction_a))
        adata.obsm['graph_neigh_a'] = graph_neigh_a
        adj_a_sym = _symmetrize(interaction_a, how=symmetrize)
        adj_a = (adj_a_sym if weight2 else _to_binary_csr(adj_a_sym))
        adata.obsm['adj_a'] = adj_a


def construct_interaction_celltype_similarity_sparse(
    adata,
    k_spatial=3,
    k_celltype=6,
    sim_threshold=0.8,
    mode='sim',                 # 'spatial' | 'sim' | 'hybrid'
    cell_proportions=None,      # obsm/obs key or ndarray (n_spots x n_celltypes)
    weight1=False,              # graph_neigh: False=binary, True=keep weights
    weight2=False,              # adj        : False=binary, True=keep weights
    sim_use_weight=True,        # in sim/hybrid, use similarity as weight or just 1
    symmetrize='max',           # 'max' | 'mean'
    use_float32=True            # store weights in float32 to save RAM
):
    """
    Sparse/large-data version:
      - No n×n cosine matrix; similarities are computed only for spatial candidates.
      - No n×n distance matrix; spatial kNN provides ranked candidates.
      - Writes CSR matrices into:
          adata.obsm['graph_neigh'], adata.obsm['adj']
        and, if adata.obsm['cell_proportions_a'] exists:
          adata.obsm['graph_neigh_a'], adata.obsm['adj_a']
    """
    # ---------- helpers ----------
    def _as_array_or_sparse(X):
        # Return ndarray or sparse matrix as-is; ensure dtype float
        if sp.issparse(X):
            return X.astype(np.float32 if use_float32 else np.float64)
        X = np.asarray(X)
        return X.astype(np.float32 if use_float32 else np.float64, copy=False)
    def _to_binary_csr(M):
        M = M if sp.issparse(M) else sp.csr_matrix(M)
        B = M.tocsr(copy=True)
        if B.nnz:
            B.data[:] = 1.0
        return B
    def _symmetrize(M, how='max'):
        M = M if sp.issparse(M) else sp.csr_matrix(M)
        if how == 'mean':
            S = (M + M.T) * 0.5
        else:  # 'max'
            S = M.maximum(M.T)
        return S.tocsr()
    def _spatial_knn_indices(coords, k):
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
        _, idx = nn.kneighbors(coords, return_distance=True)  # idx[:,0] = self
        return idx
    def _get_CP(adata, key_or_array):
        if key_or_array is None:
            return None
        if isinstance(key_or_array, str):
            if key_or_array in adata.obsm_keys():
                return _as_array_or_sparse(adata.obsm[key_or_array])
            if key_or_array in adata.obs.columns:
                return _as_array_or_sparse(adata.obs[key_or_array].values)
            raise KeyError(f"'{key_or_array}' not found in adata.obsm or adata.obs.")
        return _as_array_or_sparse(key_or_array)
    def _cosine_for_candidates(F, knn_idx, first_k, second_range=None):
        """
        Compute cosine similarities only for candidate sets.
        - F: (n,d) array/sparse
        - knn_idx: (n, K) indices with self in column 0
        - first_k: use neighbors [1:first_k+1] as the first block
        - second_range: optional (start, end) to add more candidates
        Returns rows, cols, vals lists for CSR construction.
        """
        # L2 normalize rows (works for dense or sparse)
        Fn = normalize(F, norm='l2', axis=1, copy=True)
        n = Fn.shape[0]
        rows, cols, vals = [], [], []
        def _score_block(i, js, thr):
            if len(js) == 0:
                return
            # row-vector x matrix^T
            if sp.issparse(Fn):
                sim = (Fn[i] @ Fn[js].T).A.ravel()
            else:
                sim = Fn[i, None] @ Fn[js].T
                sim = np.asarray(sim).ravel()
            keep = sim >= sim_threshold
            if np.any(keep):
                rows.extend([i]*keep.sum())
                cols.extend(js[keep].tolist())
                vals.extend(sim[keep].tolist())
        for i in range(n):
            js1 = knn_idx[i, 1:first_k+1]
            _score_block(i, js1, sim_threshold)
            if second_range is not None:
                start, end = second_range
                js2 = knn_idx[i, start:end]
                _score_block(i, js2, sim_threshold)
        return rows, cols, vals
    # ---------- inputs ----------
    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required.")
    coords = _as_array_or_sparse(adata.obsm['spatial'])
    if sp.issparse(coords):
        coords = coords.A
    n_spot = coords.shape[0]
    # spatial candidate indices (ranked by spatial distance)
    K_needed = max(k_spatial, k_celltype)
    knn_sp = _spatial_knn_indices(coords, K_needed)
    # main CP
    CP = _get_CP(adata, cell_proportions)
    # augmented CP_a (optional)
    CP_a = adata.obsm.get('cell_proportions_a', None)
    if CP_a is not None:
        CP_a = _as_array_or_sparse(CP_a)
    # ---------- build main interaction ----------
    rows, cols, vals = [], [], []
    if mode == 'spatial':
        for i in range(n_spot):
            js = knn_sp[i, 1:k_spatial+1]
            rows.extend([i]*len(js)); cols.extend(js.tolist()); vals.extend([1.0]*len(js))
    elif mode == 'sim':
        if CP is None:
            raise ValueError("mode='sim' requires cell_proportions.")
        # score ONLY the first k_celltype spatial neighbors
        r, c, v = _cosine_for_candidates(CP, knn_sp, first_k=k_celltype)
        rows += r; cols += c; vals += v
        if not sim_use_weight:
            vals = [1.0]*len(vals)
    elif mode == 'hybrid':
        if CP is None:
            raise ValueError("mode='hybrid' requires cell_proportions.")
        # 1) first k_spatial: always connect with weight=1
        for i in range(n_spot):
            js = knn_sp[i, 1:k_spatial+1]
            rows.extend([i]*len(js)); cols.extend(js.tolist()); vals.extend([1.0]*len(js))
        # 2) (k_spatial, k_celltype]: cosine ≥ threshold
        #    we compute similarities only for that block
        r, c, v = _cosine_for_candidates(CP, knn_sp, first_k=0, second_range=(k_spatial+1, k_celltype+1))
        if not sim_use_weight:
            v = [1.0]*len(v)
        rows += r; cols += c; vals += v
    else:
        raise ValueError("mode must be 'spatial', 'sim', or 'hybrid'")
    interaction = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float32 if use_float32 else np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n_spot, n_spot)
    )
    # graph_neigh (main)
    graph_neigh = interaction if weight1 else _to_binary_csr(interaction)
    adata.obsm['graph_neigh'] = graph_neigh
    # symmetric adj (main)
    adj_sym = _symmetrize(interaction, how=symmetrize)
    adata.obsm['adj'] = (adj_sym if weight2 else _to_binary_csr(adj_sym))
    # ---------- build augmented interaction (if CP_a exists) ----------
    if CP_a is not None and mode in ('sim', 'hybrid'):
        rows, cols, vals = [], [], []
        if mode == 'sim':
            r, c, v = _cosine_for_candidates(CP_a, knn_sp, first_k=k_celltype)
            if not sim_use_weight:
                v = [1.0]*len(v)
            rows += r; cols += c; vals += v
        else:  # hybrid
            for i in range(n_spot):
                js = knn_sp[i, 1:k_spatial+1]
                rows.extend([i]*len(js)); cols.extend(js.tolist()); vals.extend([1.0]*len(js))
            r, c, v = _cosine_for_candidates(CP_a, knn_sp, first_k=0, second_range=(k_spatial+1, k_celltype+1))
            if not sim_use_weight:
                v = [1.0]*len(v)
            rows += r; cols += c; vals += v
        interaction_a = sp.csr_matrix(
            (np.asarray(vals, dtype=np.float32 if use_float32 else np.float64),
             (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(n_spot, n_spot)
        )
        graph_neigh_a = interaction_a if weight1 else _to_binary_csr(interaction_a)
        adata.obsm['graph_neigh_a'] = graph_neigh_a

        adj_a_sym = _symmetrize(interaction_a, how=symmetrize)
        adata.obsm['adj_a'] = (adj_a_sym if weight2 else _to_binary_csr(adj_a_sym))   

def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata, cell_type=None, cell_proportions = None):  
    adata_Vars =  adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    # data augmentation
    if (cell_type is not None) and (cell_type in adata.obs.columns):
        feat_a, cell_type_a = permutation(feat, adata.obs[cell_type])
        if not pd.api.types.is_categorical_dtype(adata.obs[cell_type]):
            adata.obs[cell_type] = pd.Categorical(adata.obs[cell_type])
        cats = adata.obs[cell_type].cat.categories  # 假设原列是分类
        adata.obs[cell_type + '_a'] = pd.Categorical(cell_type_a.values, categories=cats)
        #adata.obs[cell_type + '_a'] = cell_type_a
    elif (cell_proportions is not None) and (cell_proportions in adata.obsm.keys()):
        feat_a, cell_proportions_a = permutation(feat, adata.obsm[cell_proportions])
        adata.obsm['cell_proportions_a'] = cell_proportions_a
    else:
        feat_a = permutation(feat)  
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    

def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    
