#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse
import torch
from torch.utils.data import Dataset, DataLoader
from anndata import AnnData
import scanpy as sc
import episcanpy.api as epi
np.warnings.filterwarnings('ignore')


def load_data(
        data_list, 
        join='inner', 
        batch_name='batch', 
        cell_type_name='cell_type', 
        reference=0, 
        target_name='batch', 
        min_features=0, 
        min_cells=0.01, 
        batch_size=None, 
        outdir='output/', 
        log=None, 
    ):
    """
    Load dataset with preprocessing
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    cell_type_name
        Use this annotation in obs as cell types for training model. Default: 'cell_type'.
    reference
        If equal to 0, do not incorporate the reference dataset. If equal to 1, incorporate unlabeled reference dataset. If equal to 2, incorporate labeled reference dataset. Default: 0.
    target_name
        If and only if reference is True, this annotation is valid and indicates the batch as the target. Default: 'batch'.
    min_features
        Minimum number (integer) or ratio (float) of features required for a cell to pass filtering. Default: 0.
    min_cells
        Minimum number (integer) or ratio (float) of cells required for a feature to pass filtering. Default: 0.01.
    batch_size
        Number of samples per batch to load. Default: None.
    log
        If log, record each operation in the log file. Default: None.
    
    Returns
    -------
    adata
        The AnnData object after combination and preprocessing.
    trainloader
        An iterable over the given dataset for training.
    testloader
        An iterable over the given dataset for testing.
    """
    if os.path.exists(outdir+'adata_preprocessed_aaa.h5ad'):
        adata = sc.read_h5ad(outdir+'adata_preprocessed.h5ad')
        if log: log.info('Loaded: {}'.format(outdir+'adata_preprocessed.h5ad'))
        if log: log.info('Processed dataset shape: {}'.format(adata.shape))
    else:
        adata = concat_data(data_list, join=join, batch_name=batch_name)
        if log: log.info('Raw dataset shape: {}'.format(adata.shape))
        if batch_name!='batch':
            adata.obs['batch'] = adata.obs[batch_name].copy()
        if 'batch' not in adata.obs:
            adata.obs['batch'] = 'batch'
        adata.obs['batch'] = adata.obs['batch'].astype('category')

        if cell_type_name!='cell_type':
            adata.obs['cell_type'] = adata.obs[cell_type_name].copy()
        if 'cell_type' not in adata.obs:
            adata.obs['cell_type'] = 'cell_type'
        adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
        
        if reference == 2:
            adata.obs['test_type'] = adata.shape[0]*['TEST']
            adata.obs['test_type'][adata.obs['batch'] != target_name] = adata.obs['cell_type'][adata.obs['batch'] != target_name].copy()
            adata.obs['test_type'] = adata.obs['test_type'].astype('category')
            cats = adata.obs['test_type'].cat.categories.tolist()
            cats.remove('TEST')
            cats.append('TEST')
            adata.obs.test_type.cat.reorder_categories(cats, inplace=True)
        else:
            adata.obs['test_type'] = adata.obs['cell_type'].copy()

        adata = preprocessing(
            adata, 
            min_features=min_features, 
            min_cells=min_cells, 
            log=log, 
        )
        # adata.write(outdir+'adata_preprocessed.h5ad', compression='gzip')
    
    if log: log.info('Number of cell types: {}'.format(len(adata.obs['cell_type'].cat.categories)))
    if log: log.info('Number of batches: {}'.format(len(adata.obs['batch'].cat.categories)))
        
    if batch_size == None:
        if adata.shape[0] >= 50000:
            batch_size = 64
        elif adata.shape[0] >= 10000:
            batch_size = 32
        else:
            batch_size = 16
    
    scdata = SingleCellDataset(adata)
    trainloader = DataLoader(
        scdata, 
        batch_size=batch_size, 
        drop_last=True, 
        shuffle=True, 
        num_workers=8
    )
    testloader = DataLoader(
        scdata, 
        batch_size=batch_size
    )
    return adata, trainloader, testloader


def concat_data(
        data_list, 
        join='inner', 
        batch_name='batch', 
        save=None
    ):
    """
    Concatenate multiple datasets along the observations axis with name ``batch_name``.
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_name
        Add the batch annotation to obs using this key. Default: 'batch'.
    save
        Path to save the new merged AnnData. Default: None.
        
    Returns
    -------
    New merged AnnData.
    """
    if len(data_list) == 1:
        return load_file(data_list[0])
    elif isinstance(data_list, str):
        return load_file(data_list)
    adata_list = []
    for root in data_list:
        adata = load_file(root)
        adata_list.append(adata)
    
    concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_name)  
    if save:
        concat.write(save, compression='gzip')
    return concat


def load_file(path):  
    """
    Load single cell dataset from file
    
    Parameters
    ----------
    path
        the path store the file
        
    Return
    ------
    AnnData
    """
    if os.path.isfile(path):
        if path.endswith(('.csv', '.csv.gz')):
            adata = sc.read_csv(path).T
        elif path.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz')):
            df = pd.read_csv(path, sep='\t', index_col=0).T
            adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        elif path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
    else:
        raise ValueError("File {} not exists".format(path))
        
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata


def preprocessing(
        adata: AnnData, 
        min_features: int = 0, 
        min_cells: float = 0.01, 
        log=None
    ):
    """
    Preprocessing scCAS data
    
    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs × n_vars. Rows correspond to cells and columns to features.
    min_features
        Minimum number (integer) or ratio (float) of features required for a cell to pass filtering. Default: 0.
    min_cells
        Minimum number (integer) or ratio (float) of cells required for a feature to pass filtering. Default: 0.01.
    log
        If log, record each operation in the log file. Default: None.
        
    Return
    -------
    The AnnData object after preprocessing.
    """
    if log: log.info('Preprocessing')
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    
    if log: log.info('Binarized')
    epi.pp.binarize(adata)
    
    if log: log.info('Filtering cells')
    min_features_ = np.ceil(min_features*adata.shape[1]) if type(min_features) == float else min_features
    sc.pp.filter_cells(adata, min_genes=min_features_)
    
    if log: log.info('Filtering features')
    min_cells_ = np.ceil(min_cells*adata.shape[0]) if type(min_cells) == float else min_cells
    sc.pp.filter_genes(adata, min_cells=min_cells_)

    if log: log.info('Calculating TF-IDF')
    adata = cal_tfidf(adata)
    
    if log: log.info('Normalizing total per cell')
    sc.pp.normalize_total(adata)
    
    if log: log.info('Max-min scaling')
    adata = maxmin_scale(adata)
    
    if log: log.info('Processed dataset shape: {}'.format(adata.shape))
    return adata


# Perform Signac TF-IDF (count_mat: cell*peak)
def cal_tfidf(adata, chunk_size=10000):
    a = np.sum(adata.X,axis=0)
    b = np.sum(adata.X,axis=1)
    adata.X = adata.X / b
    adata.X = adata.X / a
    c = 1e4 * adata.X.shape[0]
    c1, c2 = 1 / c, np.log(c)
    ct = adata.X.shape[0]/chunk_size
    if ct > 1:
        ct = int(ct)
        for i in range(ct):
            adata.X[i*chunk_size:(i+1)*chunk_size] = np.log(c1 + adata.X[i*chunk_size:(i+1)*chunk_size])
        adata.X[ct*chunk_size:] = np.log(c1 + adata.X[ct*chunk_size:])
    else:
        adata.X = np.log(c1 + adata.X)
    adata.X = adata.X + c2
    adata.X = scipy.sparse.csr_matrix(adata.X)
    return adata


def maxmin_scale(adata):
    if issparse(adata.X):
        adata.X = adata.X.todense()
    a_max = np.max(adata.X)
    a_min = np.min(adata.X)
    adata.X = (adata.X - a_min) / (a_max - a_min)
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    return adata
        
    
class SingleCellDataset(Dataset):
    """
    Dataloader of single-cell data
    """
    def __init__(self, adata):
        """
        create a SingleCellDataset object
            
        Parameters
        ----------
        adata
            AnnData object wrapping the single-cell data matrix
        """
        self.adata = adata
        self.shape = adata.shape
        
    def __len__(self):
        return self.adata.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.adata.X[idx].toarray().squeeze()
        batch_id = self.adata.obs['batch'].cat.codes[idx]
        celltype_id = self.adata.obs['test_type'].cat.codes[idx]
        return x, batch_id, celltype_id, idx


