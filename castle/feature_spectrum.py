#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import episcanpy.api as epi
from collections import Counter


def cal_index(indexs, shapes=400):
    ids = []
    for id1 in indexs:
        id3 = np.zeros(shapes, dtype=int)
        id2 = Counter(id1.reshape(-1,))
        for key in id2.keys():
            id3[key] = id2[key]
        ids.append(id3)
    return ids

    
def cal_tfidf(ys):
    N, M = ys.shape[0], ys.shape[1]
    Ns = np.sum(ys, axis=1).reshape(-1,1)
    df = np.array([sum(ys[:, i] > 0) for i in range(M)])
    df = np.log(1 + N / df)
    tfidf = (ys / Ns) * df.reshape(1,-1)
    return tfidf

    
def cal_feat_spe(adata):
    indices = adata.obsm['feature_index']
    n_codebook = np.max(indices)+1
    emb_ind = np.array(cal_index(indices, shapes=n_codebook))
    labels = adata.obs['cell_type'].cat.codes
    n_sample = len(labels)
    label_unique = np.unique(labels)
    emb_inds0 = np.zeros((len(label_unique), n_codebook))
    for i in label_unique:
        emb_inds0[i] = np.sum(emb_ind[labels == i], axis=0)
    emb_inds = cal_tfidf(emb_inds0)
    print("Feature spectrum shape: ", emb_inds.shape)
    
    ids = np.argmax(emb_inds, axis=0)
    reodered_ind2 = np.argsort(ids)
    emb_reinds = emb_inds[:, reodered_ind2]

    id_dict = Counter(ids)
    reodered_inds2 = sorted(id_dict.items(), key=lambda x:x[0], reverse=False)
    i0 = np.arange(len(reodered_inds2))
    i1 = 0
    tfidfs = []
    varss = []
    pvalues = []
    for i in range(len(reodered_inds2)):
        i2 = i1+reodered_inds2[i][1]
#         print(i,i1,i2)
        i3 = i0[i0 != i]
        vs = emb_reinds[i,i1:i2]
        reodered_ind2[i1:i2] = reodered_ind2[i1:i2][np.argsort(-vs)]
        i1 = i1+reodered_inds2[i][1]
    emb_reinds = emb_inds[:, reodered_ind2]
    adata.uns['feature_spectrum'] = emb_reinds

    return adata
    
    


