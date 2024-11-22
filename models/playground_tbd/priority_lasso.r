library(dplyr)
library(prioritylasso)
library(survival)

df_blockwise_data = read.csv2('models/playground_tbd/df_block_data.csv', sep = ',')
df_blockwise_indcs = read.csv('models/playground_tbd/df_block_indices.csv')
df_pData = read.csv2('data/merged_data/pData/original/merged_original_pData.csv', sep = ',')

construct_indcs_bp <- function(df){
    range_list <- lapply(1:nrow(df), function(i) {
        (df$i_start[i]+1):(df$i_end[i]+1)
        })
    range_list
}

remove_small_blocks <- function(indcs, df_data){
    indcs = indcs[indcs$nmb_genes == 1,]
    for(i in 1:nrow(indcs)){
        i_start = indcs[i, 'i_start']+1
        i_end = indcs[i, 'i_end']+1
        df_data = df_data[, -c(i_start:i_end)]
        print(ncol(df_data))
    }       
    return(df_data) 
}

#df_blockwise_indcs = df_blockwise_indcs[df_blockwise_indcs$nmb_genes > 1,]
index_list = construct_indcs_bp(df_blockwise_indcs)
df_data = df_blockwise_data[, -1]
#df_data = remove_small_blocks(df_blockwise_indcs, df_blockwise_data)
df_data
y = Surv(as.numeric(as.character(df_pData$MONTH_TO_BCR)), df_pData$BCR_STATUS)

prio_lasso = prioritylasso(
df_data,
y,
family = "cox",
blocks = index_list, 
block1.penalization = TRUE,
lambda.type = "lambda.1se",
type.measure = "deviance"
)
summary(y)

