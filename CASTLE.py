#!/usr/bin/env python
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single-cell Chromatin Accessibility Sequencing data analysis via discreTe Latent Embedding')
    
    parser.add_argument('--data_list', '-d', type=str, nargs='+', default=[])
    parser.add_argument('--join', type=str, default='inner')
    parser.add_argument('--batch_name', type=str, default='batch')
    parser.add_argument('--cell_type_name', type=str, default='cell_type')
    parser.add_argument('--reference', '-r', type=int, default=0)
    parser.add_argument('--target_name', type=str, default='batch')
    
    parser.add_argument('--min_features', type=int, default=0)
    parser.add_argument('--min_cells', type=float, default=0.01)
    parser.add_argument('--enc_dims', type=int, default=[1024, 256])
    parser.add_argument('--latent_dim', '-l', type=int, default=50)
    parser.add_argument('--n_embed', type=int, default=400)
    parser.add_argument('--split', type=int, default=10)
    parser.add_argument('--no_ema', action='store_false')
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--decay', type=float, default=0.99)
    
    parser.add_argument('--batch_loss_weight', type=float, default=0.001)
    parser.add_argument('--batch_loss_ratio', type=float, default=0.1)
    parser.add_argument('--celltype_loss_weight', type=float, default=0.001)
    parser.add_argument('--celltype_loss_ratio', type=float, default=0.1)
    parser.add_argument('--clf_loss_weight', type=float, default=1.0)
    
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_iteration', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=124)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--outdir', '-o', type=str, default='output/')
    parser.add_argument('--ignore_umap', action='store_true')
    parser.add_argument('--verbose', action='store_true')


    args = parser.parse_args()
    
    import castle
    adata = castle.main.CASTLE(
        data_list=args.data_list, 
        join=args.join, 
        batch_name=args.batch_name, 
        cell_type_name=args.cell_type_name, 
        reference=args.reference, 
        target_name=args.target_name, 
        min_features=args.min_features, 
        min_cells=args.min_cells, 
        enc_dims=args.enc_dims, 
        latent_dim=args.latent_dim, 
        n_embed=args.n_embed, 
        split=args.split, 
        ema=(not args.no_ema), 
        commitment_cost=args.commitment_cost, 
        decay=args.decay, 
        batch_loss_weight=args.batch_loss_weight, 
        batch_loss_ratio=args.batch_loss_ratio, 
        celltype_loss_weight=args.celltype_loss_weight, 
        celltype_loss_ratio=args.celltype_loss_ratio, 
        clf_loss_weight=args.clf_loss_weight, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        max_iteration=args.max_iteration, 
        seed=args.seed, 
        gpu=args.gpu, 
        outdir=args.outdir, 
        ignore_umap=args.ignore_umap, 
        verbose=args.verbose, 
    )
        
