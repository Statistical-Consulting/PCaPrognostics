import os
import sys
import pandas as pd
import logging

# Setup paths for Windows
PROJECT_ROOT =  os.getcwd()

# Set up path for Mac
# PROJECT_ROOT = str(Path(os.getcwd()).parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Imports
from models.modelling_process import ModellingProcess

DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.95,
    'gene_type': 'all_genes',
    'use_imputed': True,
    'use_cohorts': False, 
    'select_random' : False, 
    'requires_ohenc' : True, 
    'only_pData': False,
    'clinical_covs' : ["AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA']
}

def create_block_data(pData, X, threshold_cohorts = 2): 
    """
    Groups genes based on cohort overlap/missingness pattern and creates data blocks and corresponsing indices.

    Args:
        pData (pd.DataFrame): Clinical covariates data (if available).
        X (pd.DataFrame): Gene data.
        threshold_cohorts (int, optional): Minimum number of cohorts required to form a block. Default is 2.

    Returns:
        tuple: 
            - df_block_data (pd.DataFrame): Block-wise gene data.
            - df_block_indices (pd.DataFrame): Block indices per block.
    """
    ov_coh_per_gene = list()
    if 'cohort' in X.columns: 
        X_tmp = X.drop(labels='cohort', axis = 1)
    else: 
        X_tmp = X
    for name, values in X_tmp.items():
        genes = X['cohort'][values.notna() == True].unique().tolist()
        #if len(genes) < 9:
        res = {'cohort' : '-'.join(genes), 'gene' : name, 'len' : len(genes)}
        ov_coh_per_gene.append(res)
        
    ov_coh_per_gene_df = pd.DataFrame(ov_coh_per_gene)
    
    block_data = [] 
    block_indcs = []
    if pData is not None: 
        nmb_pdata = len(pData.columns)
        print(nmb_pdata)
        block_data.append(pData)
    else: 
        nmb_pdata = 0
    i_start = 0
    for index, cohs in enumerate(ov_coh_per_gene_df['cohort'].unique()):
        cohs_list = cohs.split('-')
        nmb_cohs = len(cohs_list)
        genes = ov_coh_per_gene_df['gene'][ov_coh_per_gene_df['cohort'] == cohs]
        nmb_genes = len(genes)
        if nmb_cohs >= 7 and nmb_genes > 100 or nmb_cohs > threshold_cohorts and nmb_genes > 300:
            sel_genes = X.loc[:, genes]
            block_data.append(sel_genes)
            indx_dict = {'nmb_cohs':nmb_cohs, 'nmb_genes': nmb_genes, 'i_start': i_start, 'i_end' : i_start + nmb_genes + nmb_pdata -1}
            i_start = i_start + nmb_genes + nmb_pdata 
            block_indcs.append(indx_dict)   
            nmb_pdata = 0 
    
    df_block_data = pd.concat(block_data, axis = 1)
    df_block_data['cohort'] = X['cohort']
    df_block_indices = pd.DataFrame(block_indcs).sort_values(by = ['nmb_cohs', 'nmb_genes'], ascending= False)
    return df_block_data, df_block_indices


mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT)

pData = mp.X.iloc[:, 0:6]
X = mp.X.iloc[:, 6:]
X['cohort'] = mp.groups

df_block_data, df_block_indices = create_block_data(pData=None, X = X, threshold_cohorts=2)
df_block_data.reset_index(inplace=True)

df_block_indices.to_csv('df_block_indices_100_300_2.csv', index=False)
df_block_data.to_csv('df_block_data_small_100_300_2.csv', index=False)
