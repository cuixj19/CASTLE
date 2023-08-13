#!/usr/bin/env python
import sys
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .layer import *


class CASTLE_train(nn.Module):
    """
    CASTLE framework
    """
    def __init__(self, enc, dec, n_batch=1, n_celltype=1, reference=0, n_embed=400, split=10, ema=True, commitment_cost=0.25, decay=0.99, 
                 batch_loss_weight=0.001, batch_loss_ratio=0.1, celltype_loss_weight=0.001, celltype_loss_ratio=0.1, clf_loss_weight=1.0):
        """
        Parameters
        ----------
        enc
            Encoder structure config.
        dec
            Decoder structure config.
        n_batch
            The number of different cell batchs. Default: 1.
        n_celltype
            The number of different cell types. Default: 1.
        reference
            If equal to 0, do not incorporate the reference dataset. If equal to 1, incorporate unlabeled reference dataset. If equal to 2, incorporate labeled reference dataset. Default: 0.
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
        """
        super().__init__()
        self.x_dim = dec[-1][1]
        self.z_dim = enc[-1][1]
        self.n_batch = n_batch
        self.n_celltype = n_celltype
        self.reference = reference
        
        self.encoder = Encoder(self.x_dim, enc, n_embed=n_embed, split=split, ema=ema, commitment_cost=commitment_cost, decay=decay)
        self.decoder = NN(self.z_dim, dec)
        if self.reference == 2:
            self.classifier = Classifier(self.z_dim, self.n_celltype)
        
        self.split = split
        self.batch_loss_weight = batch_loss_weight
        self.batch_loss_ratio = batch_loss_ratio
        self.celltype_loss_weight = celltype_loss_weight
        self.celltype_loss_ratio = celltype_loss_ratio
        self.clf_loss_weight = clf_loss_weight
    
    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        
    def encodeBatch(
            self, 
            dataloader, 
            device='cuda', 
            eval=False
        ):
        """
        Inference
        
        Parameters
        ----------
        dataloader
            An iterable over the given dataset for inference.
        device
            'cuda' or 'cpu' for . Default: 'cuda'.
        eval
            If True, set the model to evaluation mode. Default: False.
        
        Returns
        -------
        Inference layer and sample index (if return_idx=True).
        """
        self.to(device)
        if eval:
            self.eval()
        else:
            self.train()
        
        indices = np.zeros((dataloader.dataset.shape[0], self.split))
        output = np.zeros((dataloader.dataset.shape[0], self.z_dim))
        for x,_,_,idx in dataloader:
            x = x.float().to(device)
            z,_,_,index,_ = self.encoder(x)
            output[idx] = z.detach().cpu().numpy()
            indices[idx] = index.detach().cpu().numpy()
        indices = indices.astype(int)
        return output, indices
        
    def batch_loss(self, embedding, y=None):
        loss_dim = embedding.shape[-1] // self.split
        batch_loss = 0.0 * torch.mean(embedding[0])
        embeddings = []
        for i in range(self.n_batch):
            if sum(y==i) == 0:
                continue
            embeddings.append(embedding[y==i].reshape(-1, loss_dim).clone())
        
        if len(embeddings) > 1:
            d = None
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    if d == None:
                        d = torch.abs(torch.cdist(embeddings[i], embeddings[j], p=2)).view(-1)
                    else:
                        d = torch.cat((d, torch.abs(torch.cdist(embeddings[i], embeddings[j], p=2)).view(-1)))
            batch_loss = batch_loss + self.batch_loss_weight * torch.mean(torch.sort(d)[0][:int(self.batch_loss_ratio*len(d))])
        return batch_loss
    
    def celltype_loss(self, embedding, l=None):
        loss_dim = embedding.shape[-1] // self.split
        celltype_loss = 0.0 * torch.mean(embedding[0])
        d = None
        for i in range(self.n_celltype):
            if sum(l==i) <= 1 or sum((l!=i)&(l!=self.n_celltype)) <= 1:
                continue
            embedding0 = embedding[l==i].reshape(-1, loss_dim).clone()
            embedding1 = embedding[(l!=i)&(l!=self.n_celltype)].reshape(-1, loss_dim).clone()

            if d == None:
                d = torch.abs(torch.cdist(embedding0, embedding1, p=2)).view(-1)
            else:
                d = torch.cat((d, torch.abs(torch.cdist(embedding0, embedding1, p=2)).view(-1)))
        
        if d != None:
            celltype_loss = celltype_loss - self.celltype_loss_weight * torch.mean(torch.sort(d, descending=False)[0][:int(self.celltype_loss_ratio*len(d))])
        return celltype_loss
        
    def clf_loss(self, pred_l, l, test_type_id):
        idx = torch.reshape((l != test_type_id).nonzero(as_tuple=False), (-1,))
        pred_l = torch.index_select(pred_l, 0, idx)
        l = torch.index_select(l, 0, idx)
        return F.cross_entropy(pred_l, l) *len(l) / idx.size(dim=0)
    
    def fit(
            self, 
            dataloader, 
            test_type_id=None, 
            lr=1e-4, 
            max_iteration=100000, 
            max_epoch=500, 
            early_stopping=None, 
            device='cuda', 
            verbose=False
        ):
        """
        Fit model
        
        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        test_type_id
            An indicator for whether introduce labeled reference dataset. Default: None.
        lr
            Learning rate. Default: 1e-4.
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 100000.
        max_epoch
            Max epochs for training. Default: 500.
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None.
        device
            'cuda' or 'cpu' for training. Default: 'cuda'.
        verbose
            Verbosity, True or False. Default: False.
        """
        self.to(device)
        test_type_id = torch.as_tensor(test_type_id).long().to(device) if test_type_id else None
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-6)
        n_epoch = min(max_epoch, int(np.ceil(max_iteration/len(dataloader))))
        
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iter', disable=(not verbose))
                epoch_loss = defaultdict(float)
                for i, (x, y, l, idx) in tk0:
                    x, y, l = x.float().to(device), y.long().to(device), l.long().to(device)
                    z_q, z_e, latent_loss, index, perplexity = self.encoder(x)
                    recon_x = self.decoder(z_q)
                    
                    recon_loss = F.binary_cross_entropy(recon_x, x) * x.size(-1)
                    if self.reference == 2:
                        pred_l = self.classifier(z_q)
                        batch_loss = self.batch_loss(z_e, y=y)
                        celltype_loss = self.celltype_loss(z_q, l=l)
                        clf_loss = self.clf_loss_weight * self.clf_loss(pred_l, l, test_type_id)
                        loss = {'recon':recon_loss, 'latent':latent_loss, 'batch':batch_loss, 'celltype':celltype_loss, 'clf':clf_loss, 
                             'perplexity':perplexity}
                        loss_ = loss['recon'] + loss['latent'] + loss['batch'] + loss['celltype'] + loss['clf']
                    elif self.n_batch > 1:
                        batch_loss = self.batch_loss(z_e, y=y)
                        loss = {'recon':recon_loss, 'latent':latent_loss, 'batch':batch_loss, 'perplexity':perplexity}
                        loss_ = loss['recon'] + loss['latent'] + loss['batch']
                    else:
                        loss = {'recon':recon_loss, 'latent':latent_loss, 'perplexity':perplexity}
                        loss_ = loss['recon'] + loss['latent']
                    optim.zero_grad()
                    loss_.backward()
                    optim.step()
                    
                    for k,v in loss.items():
                        epoch_loss[k] += loss[k].item()
                    info = ','.join(['{}={:.3f}'.format(k, v) for k,v in loss.items()])
                    tk0.set_postfix_str(info)  

                epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
                tq.set_postfix_str(epoch_info)
                    
                early_stopping(sum(list(epoch_loss.values())[:-1]), self)
#                 if early_stopping.early_stop:
                if early_stopping.early_stop and epoch+1 > 200:
                    print('EarlyStopping: run {} epoch'.format(epoch+1))
                    break
    
    
    