#!/usr/bin/env python
import os
import numpy as np
import torch
import scanpy as sc
from anndata import AnnData
import episcanpy.api as epi

from .data import load_data
from .model.castle_train import CASTLE_train
from .model.utils import EarlyStopping
from .metrics import eval_cls
from .logger import create_logger
# from .feature_spectrum import cal_feat_spe


def CASTLE(
        data_list, 
        join='inner', 
        batch_name='batch', 
        cell_type_name='cell_type', 
        reference=0, 
        target_name='batch', 
        min_features=0, 
        min_cells=0.01, 
        enc_dims=[1024, 256], 
        latent_dim=50, 
        n_embed=400, 
        split=10, 
        ema=True, 
        commitment_cost=0.25, 
        decay=0.99, 
        batch_loss_weight=0.001, 
        batch_loss_ratio=0.1, 
        celltype_loss_weight=0.001, 
        celltype_loss_ratio=0.1, 
        clf_loss_weight=1.0, 
        batch_size=None, 
        lr=1e-4, 
        max_iteration=100000, 
        max_epoch=500, 
        seed=124, 
        gpu=0, 
        outdir='output/', 
        ignore_umap=False, 
        verbose=False, 
        show=True, 
    ):
    """
    single-cell Chromatin Accessibility Sequencing data analysis via discreTe Latent Embedding
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    cell_type_name
        Use this annotation in obs as cell types for training model. Default: 'cell_type'.
    reference
        If equal to 0, do not incorporate the reference dataset. If equal to 1, incorporate unlabeled reference dataset. If equal to 2, incorporate labeled reference dataset. Default: 0.
    target_name
        If and only if reference is not 0, this annotation is valid and indicates the batch as the target. Default: 'batch'.
    min_features
        Minimum number (integer) or ratio (float) of features required for a cell to pass filtering. Default: 0.
    min_cells
        Minimum number (integer) or ratio (float) of cells required for a feature to pass filtering. Default: 0.01.
    enc_dims
        The number of nodes in the linear layers of encoder. Default: [1024, 256].
    latent_dim
        The dimension of latent embeddings. Default: 50.
    n_embed
        The size of codebook. Default: 400.
    split
        The number of split quantizations. Default: 10.
    ema
        If True, adopt the exponential moving average (EMA) to update the codebook instead of the codebook loss. Default: True.
    commitment_cost
        The weight of commitment loss designed for encoder. Default: 0.25.
    decay
        The decay ratio when ema is True. Default: 0.99.
    batch_loss_weight
        The weight of batch loss. Default: 0.001.
    batch_loss_ratio
        The proportion of pairs of quantized features when compute the batch loss. Default: 0.1.
    celltype_loss_weight
        The weight of celltype loss. Default: 0.001.
    celltype_loss_ratio
        The proportion of pairs of quantized features when compute the celltype loss. Default: 0.1.
    clf_loss_weight
        The weight of clf loss. Default: 1.0.
    batch_size
        Number of samples per batch to load. Default: None.
    lr
        Learning rate. Default: 1e-4.
    max_iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 100000.
    seed
        Random seed for torch and numpy. Default: 124.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: 'output/'.
    ignore_umap
        If True, do not perform UMAP for visualization, louvain for clustering and clustering evaluation. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    
    Returns
    -------
    The output folder contains:
    adata.h5ad
        The AnnData matrice after batch effects removal. The low-dimensional representation of the data is stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records raw data information, filter conditions, model parameters etc.
    umap.pdf 
        UMAP plot for visualization.
    """
    
    outdir = outdir+'/'
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'log.txt')

    np.random.seed(seed) # seed
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
        log.info('Using GPU: {}'.format(gpu))
    else:
        device='cpu'
        log.info('Using CPU')
    
    adata, trainloader, testloader = load_data(
        data_list, 
        join=join, 
        batch_name=batch_name, 
        cell_type_name=cell_type_name, 
        reference=reference, 
        target_name=target_name, 
        batch_size=batch_size, 
        min_features=min_features, 
        min_cells=min_cells,
        outdir=outdir,
        log=log
    )

    early_stopping = EarlyStopping(patience=10, checkpoint_file=outdir+'/checkpoint/model.pt', verbose=False)
    x_dim, n_batch = adata.shape[1], len(adata.obs['batch'].cat.categories)
    n_celltype = len(adata.obs['test_type'].cat.categories)-1 if reference==2 else len(adata.obs['cell_type'].cat.categories)

    # model config
    enc = [['fc', enc_dim, 1, 'tanh', 0] for enc_dim in enc_dims] + [['fc', latent_dim, 0, '', 0]]
    dec = [['fc', x_dim, 1, 'sigmoid', 0]]

    model = CASTLE_train(enc, dec, n_batch=n_batch, n_celltype=n_celltype, reference=reference, n_embed=n_embed, split=split, ema=ema, commitment_cost=commitment_cost, decay=decay, batch_loss_weight=batch_loss_weight, batch_loss_ratio=batch_loss_ratio, celltype_loss_weight=celltype_loss_weight, celltype_loss_ratio=celltype_loss_ratio, clf_loss_weight=clf_loss_weight)

    test_type_id = np.where(adata.obs['test_type'].cat.categories == 'TEST')[0][0] if reference == 2 else None

    log.info('model\n'+model.__repr__())
    model.fit(
        trainloader, 
        test_type_id,
        lr=lr, 
        max_iteration=max_iteration, 
        max_epoch=max_epoch,
        device=device, 
        early_stopping=early_stopping, 
        verbose=verbose
    )
    torch.save({'n_top_features':adata.var.index, 'enc':enc, 'dec':dec, 'n_batch':n_batch, 'n_celltype':n_celltype}, outdir+'/checkpoint/config.pt')
    
    model = CASTLE_train(enc, dec, n_batch=n_batch, n_celltype=n_celltype, reference=reference, n_embed=n_embed, split=split, ema=ema, commitment_cost=commitment_cost, decay=decay, batch_loss_weight=batch_loss_weight, batch_loss_ratio=batch_loss_ratio, celltype_loss_weight=celltype_loss_weight, celltype_loss_ratio=celltype_loss_ratio, clf_loss_weight=clf_loss_weight)
    model.load_model(outdir+'/checkpoint/model.pt')
    model.to(device)
    
    adata.obsm['latent'], adata.obsm['feature_index'] = model.encodeBatch(testloader, device=device)
    log.info('Output dir: {}'.format(outdir))
    # adata.write(outdir+'adata.h5ad')
    
    if not ignore_umap:
        if reference != 0:
            adata = adata[adata.obs['batch'] == target_name].copy()
        adata1 = adata[:, :latent_dim].copy()
        adata1.X = adata1.obsm['latent'].copy()
        log.info('Plot umap')
        sc.pp.neighbors(adata1, n_neighbors=15, method='umap', use_rep='latent', random_state=0)
        sc.tl.umap(adata1, min_dist=0.1)
        
        # Clustering evaluation
        results, adata1 = eval_cls(adata1, res_name='CASTLE', use_rep='latent', test=False)
        results.to_csv(outdir+'results.csv')
        
        # UMAP visualization
        sc.settings.figdir = outdir
        sc.set_figure_params(dpi=80, figsize=(8,8), fontsize=20)
        cols = ['batch', 'cell_type'] if n_batch>1 else ['cell_type']
        color = [c for c in cols if c in adata1.obs]
        sc.pl.umap(adata1, color=color, wspace=0.5, ncols=2, show=show, return_fig=True)
        
        adata1.write(outdir+'adata.h5ad')
        return adata1
    
    return adata

