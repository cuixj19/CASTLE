#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import episcanpy.api as epi
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering


def kmeans_m(adata, num_clusters, use_rep='latent'):
    """
    Compute kmeans clustering using latent embeddings fits.
    random_state = 2023
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=2023).fit(adata.obsm[use_rep]) 
    adata.obs['kmeans'] = pd.Series(kmeans.labels_,index=adata.obs.index).astype('category')

    
def hc_m(adata, num_clusters, use_rep='latent'):
    """
    Compute hierarchical clustering using latent embeddings fits.
    """
    hc = AgglomerativeClustering(n_clusters=num_clusters).fit(adata.obsm[use_rep])
    adata.obs['hc'] = pd.Series(hc.labels_,index=adata.obs.index).astype('category')

    
def cluster_eval(adata, label_key, cluster_key, res_name, res_table_pd):
    print(cluster_key)
    ARI = epi.tl.ARI(adata, label_key, cluster_key)
    AMI = epi.tl.AMI(adata, label_key, cluster_key)
    NMI = metrics.normalized_mutual_info_score(adata.obs[label_key], adata.obs[cluster_key])
    FMI = metrics.fowlkes_mallows_score(adata.obs[label_key], adata.obs[cluster_key])
    print('ARI:%.3f\tAMI:%.3f\tNMI:%.3f\tFMI:%.3f'%(ARI,AMI,NMI,FMI))
    res_table_pd = res_table_pd.append(pd.DataFrame({'method':[res_name]*4, 'cluster_key':[cluster_key]*4, 'metric':['ARI','AMI','NMI','FMI'], 
                                  'value':[ARI,AMI,NMI,FMI]}), ignore_index=True)

    return res_table_pd


def eval_cls(adata, res_name, use_rep='latent', test=False):
    epi.tl.louvain(adata, key_added='Dlouvain')
    epi.tl.getNClusters(adata, n_cluster=adata.obs['cell_type'].nunique(), method='louvain', key_added='Clouvain')
    epi.tl.leiden(adata, key_added='Dleiden')
    epi.tl.getNClusters(adata, n_cluster=adata.obs['cell_type'].nunique(), method='leiden', key_added='Cleiden')

    kmeans_m(adata, num_clusters=adata.obs['cell_type'].nunique(), use_rep=use_rep)
    adata.obs['Ckmeans'] = adata.obs['kmeans'].copy()

    hc_m(adata, num_clusters=adata.obs['cell_type'].nunique(), use_rep=use_rep)
    adata.obs['Chc'] = adata.obs['hc'].copy()

    if test:
        adata = adata[adata.obs.test_type=='TEST'].copy()
    print(adata.shape)
    
    res_table_pd = pd.DataFrame(columns=['method', 'cluster_key', 'metric', 'value'])
    res_table_pd = cluster_eval(adata, 'cell_type', 'Dlouvain', res_name, res_table_pd)
    res_table_pd = cluster_eval(adata, 'cell_type', 'Clouvain', res_name, res_table_pd)
    res_table_pd = cluster_eval(adata, 'cell_type', 'Dleiden', res_name, res_table_pd)
    res_table_pd = cluster_eval(adata, 'cell_type', 'Cleiden', res_name, res_table_pd)
    res_table_pd = cluster_eval(adata, 'cell_type', 'Ckmeans', res_name, res_table_pd)
    res_table_pd = cluster_eval(adata, 'cell_type', 'Chc', res_name, res_table_pd)
    
    return res_table_pd, adata
    
    


