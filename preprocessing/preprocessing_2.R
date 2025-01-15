pData <- as.data.frame(read_csv('data/merged_data/pData/imputed/test_pData_imputed.csv', lazy = TRUE))
common_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "common_genes", "common_genes_test_imputed.csv"), row.names=1)
intersect_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "intersection", "intersect_test_genes_imputed.csv"), row.names=1)
all_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "all_genes", "all_genes_test.csv"), row.names=1)
# Split pData
pData_cohort1 <- pData[grep("test_cohort_1", rownames(pData)), ]
pData_cohort2 <- pData[grep("test_cohort_2", rownames(pData)), ]

# Split common_genes
common_genes_cohort1 <- common_genes[grep("test_cohort_1", rownames(common_genes)), ]
common_genes_cohort2 <- common_genes[grep("test_cohort_2", rownames(common_genes)), ]

# Split intersect_genes
intersect_genes_cohort1 <- intersect_genes[grep("test_cohort_1", rownames(intersect_genes)), ]
intersect_genes_cohort2 <- intersect_genes[grep("test_cohort_2", rownames(intersect_genes)), ]

# Split all_genes
all_genes_cohort1 <- all_genes[grep("test_cohort_1", rownames(all_genes)), ]
all_genes_cohort2 <- all_genes[grep("test_cohort_2", rownames(all_genes)), ]

# Save pData splits
write.csv(pData_cohort1, file.path(".", "data", "cohort_data", "pData", "imputed", "test_pData_cohort1_imputed.csv"))
write.csv(pData_cohort2, file.path(".", "data", "cohort_data", "pData", "imputed", "test_pData_cohort2_imputed.csv"))

# Save common_genes splits
write.csv(common_genes_cohort1, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort1.csv"))
write.csv(common_genes_cohort2, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort2.csv"))

# Save intersect_genes splits
write.csv(intersect_genes_cohort1, file.path(".", "data", "cohort_data", "exprs", "intersect_test_genes_imputed_cohort1.csv"))
write.csv(intersect_genes_cohort2, file.path(".", "data", "cohort_data", "exprs", "intersect_test_genes_imputed_cohort2.csv"))

# Save all_genes splits
write.csv(all_genes_cohort1, file.path(".", "data", "cohort_data", "exprs", "all_genes_test_cohort1.csv"))
write.csv(all_genes_cohort2, file.path(".", "data", "cohort_data", "exprs", "all_genes_test_cohort2.csv"))
