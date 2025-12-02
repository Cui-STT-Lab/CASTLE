#This code has been partly adapted from https://github.com/JinmiaoChenLab/GraphST

import torch
# import time
# import random
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
# from scipy.sparse.csc import csc_matrix
# from scipy.sparse.csr import csr_matrix
import pandas as pd
from .model import Encoder, Encoder_sparse
from .preprocess import construct_interaction_celltype_similarity_sparse, construct_interaction_celltype_similarity, preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction_celltype, add_contrastive_label, get_feature, permutation, fix_seed

class CASTLE():
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        datatype = '10X',
        mode='spatial',  # 'spatial' | 'celltype' | 'pca_cosine_knn' | 'snn_jaccard' | 'adaptive_rbf' | 'hybrid' | 'knn'
        k_spatial=5, k_celltype=10,
        cell_type=None,
        sim_threshold = 0.9, #0.6 for cell similarity
        cell_proportions=None,
        weight1=False,
        weight2=False
        ):
        '''
        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning. 
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning. 
            The default is 1.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional    
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data. 
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.datatype = datatype
        self.adj_a = None
        self.graph_neigh_a = None
        self.k_spatial = k_spatial
        self.k_celltype = k_celltype
        self.mode = mode
        self.cell_type = cell_type
        self.sim_threshold = sim_threshold
        self.cell_proportions = cell_proportions
        self.weight1 = weight1  
        self.weight2 = weight2

        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)
        
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata, cell_type=cell_type, cell_proportions=cell_proportions)

        if 'adj' not in adata.obsm.keys():
            if self.datatype == 'Slide':
              construct_interaction_celltype_similarity_sparse(self.adata, k_spatial=k_spatial, k_celltype=k_celltype, sim_threshold=sim_threshold, mode=mode, cell_proportions=cell_proportions, weight1=weight1, weight2=weight2)
            elif self.datatype == 'Stereo':
              construct_interaction_celltype(self.adata, mode=mode, cell_type=cell_type, k_spatial=k_spatial, k_celltype=k_celltype, sim_threshold=sim_threshold, weight1=weight1, weight2=weight2)
            elif self.datatype == 'cell':    
              construct_interaction_celltype(self.adata, mode=mode, cell_type=cell_type, k_spatial=k_spatial, k_celltype=k_celltype, sim_threshold=sim_threshold, weight1=weight1, weight2=weight2)
            else:
              construct_interaction_celltype_similarity(self.adata, k_spatial=k_spatial, k_celltype=k_celltype, sim_threshold=sim_threshold, mode=mode, cell_proportions=cell_proportions, weight1=weight1, weight2=weight2)

        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)

        if 'adj' in self.adata.obsm:
            self.adj = self.adata.obsm['adj']            
        else:
            raise KeyError("No adjacency found.")
        if 'adj_a' in self.adata.obsm:
            self.adj_a = self.adata.obsm['adj_a']  

        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        if ('graph_neigh_a' in self.adata.obsm) and (self.adj_a is not None):
            self.graph_neigh_a = torch.FloatTensor(self.adata.obsm['graph_neigh_a'].copy() + np.eye(self.adj_a.shape[0])).to(self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide']:
           #using sparse
           print('Building sparse matrix ...')
           self.adj = preprocess_adj_sparse(self.adj).to(self.device)
           if self.adj_a is not None:
                self.adj_a = preprocess_adj_sparse(self.adj_a).to(self.device)
        else: 
           # standard version
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)
            if self.adj_a is not None:
                self.adj_a = preprocess_adj(self.adj_a)
                self.adj_a = torch.FloatTensor(self.adj_a).to(self.device)
            
    def train(self):
        if self.datatype in ['Stereo', 'Slide']:
           self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh, self.graph_neigh_a).to(self.device)
        else:
           self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh, self.graph_neigh_a).to(self.device)
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        print('Begin to train ST data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            
            self.features_a = permutation(self.features)
            if self.datatype == 'Slide':
              construct_interaction_celltype_similarity_sparse(self.adata, k_spatial=self.k_spatial, k_celltype=self.k_celltype, sim_threshold=self.sim_threshold, mode=self.mode, cell_proportions=self.cell_proportions, weight1=self.weight1, weight2=self.weight2)
            elif self.datatype == 'Stereo':
              construct_interaction_celltype(self.adata, mode=self.mode, cell_type=self.cell_type, k_spatial=self.k_spatial, k_celltype=self.k_celltype, sim_threshold=self.sim_threshold, weight1=self.weight1, weight2=self.weight2)
            elif self.datatype == 'cell':    
              construct_interaction_celltype(self.adata, mode=self.mode, cell_type=self.cell_type, k_spatial=self.k_spatial, k_celltype=self.k_celltype, sim_threshold=self.sim_threshold, weight1=self.weight1, weight2=self.weight2)
            else:
              construct_interaction_celltype_similarity(self.adata, k_spatial=self.k_spatial, k_celltype=self.k_celltype, sim_threshold=self.sim_threshold, mode=self.mode, cell_proportions=self.cell_proportions, weight1=self.weight1, weight2=self.weight2)
            if 'adj' in self.adata.obsm:
                self.adj = self.adata.obsm['adj']            
            if 'adj_a' in self.adata.obsm:
                self.adj_a = self.adata.obsm['adj_a']  
            
            if self.datatype in ['Stereo', 'Slide']:
                #using sparse
                print('Building sparse matrix ...')
                self.adj = preprocess_adj_sparse(self.adj).to(self.device)
                if self.adj_a is not None:
                        self.adj_a = preprocess_adj_sparse(self.adj_a).to(self.device)
            else: 
                # standard version
                self.adj = preprocess_adj(self.adj)
                self.adj = torch.FloatTensor(self.adj).to(self.device)
                if self.adj_a is not None:
                    self.adj_a = preprocess_adj(self.adj_a)
                    self.adj_a = torch.FloatTensor(self.adj_a).to(self.device)
                        
            self.hidden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj, self.adj_a)
            
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)
            
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(self.features, self.features_a, self.adj, self.adj_a)[1]
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy() 
                self.emb = self.model(self.features, self.features_a, self.adj, self.adj_a)[0]
                self.emb = F.normalize(self.emb, p=2, dim=1).detach().cpu().numpy()
            else:
                self.emb_rec = self.model(self.features, self.features_a, self.adj, self.adj_a)[1].detach().cpu().numpy()
                self.emb = self.model(self.features, self.features_a, self.adj, self.adj_a)[0].detach().cpu().numpy()

            self.adata.obsm['emb'] = self.emb_rec
            self.adata.obsm['GraphST_newG'] = self.emb
                
            return self.adata

