import pandas as pd

def create_block_data(pData, X, threshold_cohorts = 2): 
    ov_coh_per_gene = list()
    if X['cohort'] is not None: 
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
    # create block Dataframe
    nmb_pdata = len(pData.columns)
    block_data.append(pData)
    i_start = 0
    for index, cohs in enumerate(ov_coh_per_gene_df['cohort'].unique()):
        cohs_list = cohs.split('-')
        nmb_cohs = len(cohs_list)
        genes = ov_coh_per_gene_df['gene'][ov_coh_per_gene_df['cohort'] == cohs]
        nmb_genes = len(genes)
        if nmb_genes > threshold_cohorts:
            sel_genes = X.loc[:, genes]
            block_data.append(sel_genes)
            indx_dict = {'nmb_cohs':nmb_cohs, 'nmb_genes': nmb_genes, 'i_start': i_start, 'i_end' : i_start + nmb_genes + nmb_pdata -1}
            i_start = i_start + nmb_genes
            block_indcs.append(indx_dict)   
            nmb_pdata = 0 
    
    df_block_data = pd.concat(block_data, axis = 1)
    df_block_data['cohort'] = X['cohort']
    df_block_indices = pd.DataFrame(block_indcs).sort_values(by = ['nmb_cohs', 'nmb_genes'], ascending= False)
    return df_block_data, df_block_indices