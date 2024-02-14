#!/usr/bin/env python
import castle

# Reproduce the results of CASTLE on 16 benchmark datasets
data_lists = ['InSilico','Stimulated Droplet','Resting Droplet','Fetal Lung','Fetal Liver','Immune','Splenocyte','Bone Marrow A',
              'Bone Marrow B','Lung A','Lung B','Whole Brain A','Whole Brain B','Cerebellum','Testes','Brain']
for data_list in data_lists:
    adata = castle.main.CASTLE(
        data_list=data_list+'.h5ad', 
        outdir='./'+data_list, 
    )

# Reproduce the results of CASTLE on 8 benchmark datasets with unlabeled reference datasets
data_lists = ['Stimulated Droplet','Resting Droplet','Bone Marrow A','Bone Marrow B','Lung A','Lung B','Whole Brain A','Whole Brain B']
for data_list in data_lists:
    adata = castle.main.CASTLE(
        data_list=data_list+'_with_unlabeled_reference.h5ad', 
        outdir='./'+data_list, 
        reference=1, 
        target_name='batch1', 
    )

# Reproduce the results of CASTLE on 8 benchmark datasets with labeled reference datasets
data_lists = ['Stimulated Droplet','Resting Droplet','Bone Marrow A','Bone Marrow B','Lung A','Lung B','Whole Brain A','Whole Brain B']
for data_list in data_lists:
    adata = castle.main.CASTLE(
        data_list=data_list+'_with_labeled_reference.h5ad', 
        outdir='./'+data_list, 
        reference=2, 
        target_name='batch1', 
    )
        
