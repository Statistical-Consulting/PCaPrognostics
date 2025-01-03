library(dplyr)

db_data <- readRDS('models/playground_tbd/msigdbr_H.Rds')
load('models/playground_tbd/PIONEER_genesets.Rdata')
#str(PIONEER_PCa_genesets)
#str(db_data)

db_data_grouped <- db_data %>% group_by(gs_id) %>% reframe(genes = paste0(ensembl_id, collapse = ","))
db_data_list = strsplit(db_data_grouped$genes, ',')
names(db_data_list) <- db_data_grouped$gs_id
genes_db = do.call(c, db_data_list)
genes_db_uniq= unique(genes_db)
str(pathways_db)
str(genes_db_uniq)
pathways_db <- unique(db_data$gs_id)

pathways_pioneer <- names(PIONEER_PCa_genesets)
unnamed_pioneer <- unname(PIONEER_PCa_genesets)
genes_pioneer <- do.call(c, unnamed_pioneer)
genes_pioneer_uniq <- unique(genes_pioneer)
str(genes_pioneer_uniq)

list_cmplt <- c(db_data_list, PIONEER_PCa_genesets)

pathways_cmplt = names(list_cmplt)
genes_cmplt = unique(c(genes_db_uniq, genes_pioneer_uniq))

str(pathways_cpmlt)
str(genes_cmplt)

mask <- matrix(0, ncol = length(genes_cmplt), nrow = length(pathways_cpmlt))
rownames(mask) <- pathways_cmplt
colnames(mask) <- genes_cmplt

for (pathway in pathways_cmplt){
    pathway_genes <- list_cmplt[[pathway]]
    mask[pathway, colnames(mask) %in% pathway_genes] <- 1
}

#print(mask_pioneer)
str(mask)
write.csv(mask, "pathway_mask.csv", row.names = TRUE)