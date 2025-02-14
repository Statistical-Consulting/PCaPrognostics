library(dplyr)
# creates sparse pathway mask based on existing pathway information
db_data <- readRDS('data/msigdbr_H.Rds')
load('data/PIONEER_genesets.Rdata')

db_data_grouped <- db_data %>% group_by(gs_id) %>% reframe(genes = paste0(ensembl_id, collapse = ","))
db_data_list = strsplit(db_data_grouped$genes, ',')
names(db_data_list) <- db_data_grouped$gs_id
genes_db = do.call(c, db_data_list)
genes_db_uniq= unique(genes_db)
pathways_db <- unique(db_data$gs_id)

pathways_pioneer <- names(PIONEER_PCa_genesets)
unnamed_pioneer <- unname(PIONEER_PCa_genesets)
genes_pioneer <- do.call(c, unnamed_pioneer)
genes_pioneer_uniq <- unique(genes_pioneer)

list_cmplt <- c(db_data_list, PIONEER_PCa_genesets)

pathways_cmplt = names(list_cmplt)
genes_cmplt = unique(c(genes_db_uniq, genes_pioneer_uniq))

mask <- matrix(0, ncol = length(genes_cmplt), nrow = length(pathways_cmplt))
rownames(mask) <- pathways_cmplt
colnames(mask) <- genes_cmplt

for (pathway in pathways_cmplt){
    pathway_genes <- list_cmplt[[pathway]]
    mask[pathway, colnames(mask) %in% pathway_genes] <- 1
}

write.csv(mask, "data/pathway_mask.csv", row.names = TRUE)