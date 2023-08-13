#!/usr/bin/env python
import math
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F


activation = {
    'relu':nn.ReLU(),
    'rrelu':nn.RReLU(),
    'sigmoid':nn.Sigmoid(),
    'leaky_relu':nn.LeakyReLU(),
    'tanh':nn.Tanh(),
    '':None
}       

class Block(nn.Module):
    """
    Basic block consist of:
        fc -> bn -> act -> dropout
    """
    def __init__(
            self,
            input_dim, 
            output_dim, 
            norm=0, 
            act='', 
            dropout=0
        ):
        """
        Parameters
        ----------
        input_dim
            dimension of input
        output_dim
            dimension of output
        norm
            batch normalization
        act
            activation function
        dropout
            dropout rate
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
        if norm == 1:
            self.norm = nn.BatchNorm1d(output_dim)
        else:
            self.norm = None
            
        self.act = activation[act]
            
        if dropout >0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
    def forward(self, x):
        h = self.fc(x)
        if self.norm:
            if len(x) == 1:
                pass
            else:
                h = self.norm(h)
        if self.act:
            h = self.act(h)
        if self.dropout:
            h = self.dropout(h)
        return h


class NN(nn.Module):
    """
    Neural network consist of multi Blocks
    """
    def __init__(self, input_dim, cfg):
        """
        Parameters
        ----------
        input_dim
            input dimension
        cfg
            model structure configuration, 'fc' -> fully connected layer
        """
        super().__init__()
        net = []
        for i, layer in enumerate(cfg):
            if i==0:
                d_in = input_dim
            if layer[0] == 'fc':
                net.append(Block(d_in, *layer[1:]))
            d_in = layer[1]
        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, input_dim, cfg, n_embed=400, split=10, ema=True, commitment_cost=0.25, decay=0.99):
        super().__init__()
        h_dim = cfg[-2][1]
        latent_dim = cfg[-1][1]
        self.latent_dim = latent_dim
        self.enc = NN(input_dim, cfg[:-1])
        self.quantize_linear = NN(h_dim, cfg[-1:])
        self.quantize = split_quant(n_embed=n_embed, embed_dim=latent_dim, split=split, ema=ema, commitment_cost=commitment_cost, decay=decay)

    def forward(self, x):
        encs = self.enc(x)
        z_e = self.quantize_linear(encs)
        z_q, latent_loss, index, perplexity = self.quantize(z_e)
        return z_q, z_e, latent_loss, index, perplexity

    
class split_quant(nn.Module):
    def __init__(self, embed_dim=50, n_embed=400, split=10, ema=True, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.part = embed_dim // split
        self.split = split
        
        if ema:
            self.vq = VectorQuantizerEMA(num_embeddings=n_embed, embedding_dim=self.part, commitment_cost=commitment_cost, decay=decay)
        else:
            self.vq = VectorQuantizer(num_embeddings=n_embed, embedding_dim=self.part, commitment_cost=commitment_cost)

    def forward(self, z_e):
        quant, latent_loss, index, perplexity = self.vq(z_e.reshape(-1, self.part))
        return quant.reshape(z_e.shape[0], -1), latent_loss, index.reshape(z_e.shape[0], -1), perplexity
    
    
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        encoding_indices = encoding_indices.reshape(inputs.shape[0], -1)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)
            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, encoding_indices, perplexity

    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost
    
    def forward(self, inputs):
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        encoding_indices = encoding_indices.reshape(inputs.shape[0], -1)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, encoding_indices, perplexity
   
    
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim, momentum=0.1),
            nn.Dropout(0.2),
        )

    def forward(self, x):     
        z = self.linear1(x)
        return z

