"
This second preprocessing file covers mainly preprocessing regarding the scores 
and test cohorts. It is seperate due to the already extensive run time of the 
first general preprocessing file and because its not necessary for model training.
All rows can simply be executed to generate all .csv files.
"

library(readr)

pData <- as.data.frame(read_csv('data/merged_data/pData/imputed/test_pData_imputed.csv', lazy = TRUE))
common_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "common_genes", "common_genes_test_imputed.csv"), row.names=1)
intersect_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "intersection", "intersect_test_genes_imputed.csv"), row.names=1)
all_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "all_genes", "all_genes_test.csv"), row.names=1)
scores <- read.csv(file.path(".", "data", "scores", "test_scores.csv"), row.names=1)
test_pData_with_scores <- cbind(pData, scores)

# Split pData
pData_cohort1 <- pData[grep("test_cohort_1", pData$...1), ]
pData_cohort2 <- pData[grep("test_cohort_2", pData$...1), ]
pData_cohort2_1_example <- pData_cohort2[1,]

# split pData with risks cores
test_pData_with_scores_cohort1 <- test_pData_with_scores[grep("test_cohort_1", test_pData_with_scores$...1), ]
test_pData_with_scores_cohort2 <- test_pData_with_scores[grep("test_cohort_2", test_pData_with_scores$...1), ]

# For cohort 1
low_risk_cohort1 <- subset(test_pData_with_scores_cohort1, risk_score < 0)
high_risk_cohort1 <- subset(test_pData_with_scores_cohort1, risk_score >= 0)

# For cohort 2
low_risk_cohort2 <- subset(test_pData_with_scores_cohort2, risk_score < 0)
high_risk_cohort2 <- subset(test_pData_with_scores_cohort2, risk_score >= 0)




# Split common_genes
common_genes_cohort1 <- common_genes[grep("test_cohort_1", rownames(common_genes)), ]
common_genes_cohort2 <- common_genes[grep("test_cohort_2", rownames(common_genes)), ]
common_genes_cohort2_1_example <-common_genes_cohort2[1,]

# Split common_genes risk based
common_genes_cohort1_low_risk <- common_genes_cohort1[rownames(common_genes_cohort1) %in% low_risk_cohort1$...1, ]
common_genes_cohort1_high_risk <- common_genes_cohort1[rownames(common_genes_cohort1) %in% high_risk_cohort1$...1, ]
common_genes_cohort2_low_risk <- common_genes_cohort2[rownames(common_genes_cohort2) %in% low_risk_cohort2$...1, ]
common_genes_cohort2_high_risk <- common_genes_cohort2[rownames(common_genes_cohort2) %in% high_risk_cohort2$...1, ]


# Split intersect_genes
intersect_genes_cohort1 <- intersect_genes[grep("test_cohort_1", rownames(intersect_genes)), ]
intersect_genes_cohort2 <- intersect_genes[grep("test_cohort_2", rownames(intersect_genes)), ]

# Split intersect_genes risk based
intersect_genes_cohort1_low_risk <- intersect_genes_cohort1[rownames(intersect_genes_cohort1) %in% low_risk_cohort1$...1, ]
intersect_genes_cohort1_high_risk <- intersect_genes_cohort1[rownames(intersect_genes_cohort1) %in% high_risk_cohort1$...1, ]
intersect_genes_cohort2_low_risk <- intersect_genes_cohort2[rownames(intersect_genes_cohort2) %in% low_risk_cohort2$...1, ]
intersect_genes_cohort2_high_risk <- intersect_genes_cohort2[rownames(intersect_genes_cohort2) %in% high_risk_cohort2$...1, ]




# Split all_genes
all_genes_cohort1 <- all_genes[grep("test_cohort_1", rownames(all_genes)), ]
all_genes_cohort2 <- all_genes[grep("test_cohort_2", rownames(all_genes)), ]

rownames(scores) <- pData[,1]
scores_cohort1 <- scores[grep("test_cohort_1", rownames(scores)), ]
scores_cohort2 <- scores[grep("test_cohort_2", rownames(scores)), ]


# Save pData splits
write.csv(pData_cohort1, file.path(".", "data", "cohort_data", "pData", "imputed", "test_pData_cohort1_imputed.csv"))
write.csv(pData_cohort2, file.path(".", "data", "cohort_data", "pData", "imputed", "test_pData_cohort2_imputed.csv"))
write.csv(pData_cohort2_1_example, file.path(".", "data", "cohort_data", "pData", "imputed", "test_pData_cohort2_imputed_1_example.csv"))

# Save pData + risk score splits
write.csv(low_risk_cohort1, file.path(".", "data", "cohort_data", "pData", "imputed", "low_risk_pData_test_cohort1.csv"))
write.csv(high_risk_cohort1, file.path(".", "data", "cohort_data", "pData", "imputed", "high_risk_pData_test_cohort1.csv"))
write.csv(low_risk_cohort2, file.path(".", "data", "cohort_data", "pData", "imputed", "low_risk_pData_test_cohort2.csv"))
write.csv(high_risk_cohort2, file.path(".", "data", "cohort_data", "pData", "imputed", "high_risk_pData_test_cohort2.csv"))


# Save common_genes splits
write.csv(common_genes_cohort1, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort1.csv"))
write.csv(common_genes_cohort2, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort2.csv"))
write.csv(common_genes_cohort2_1_example, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort2_1_example.csv"))

# Save common_genes risk based splits
write.csv(common_genes_cohort1_low_risk, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort1_low_risk.csv"))
write.csv(common_genes_cohort1_high_risk, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort1_high_risk.csv"))
write.csv(common_genes_cohort2_low_risk, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort2_low_risk.csv"))
write.csv(common_genes_cohort2_high_risk, file.path(".", "data", "cohort_data", "exprs", "common_genes_test_imputed_cohort2_high_risk.csv"))


# Save intersect_genes splits
write.csv(intersect_genes_cohort1, file.path(".", "data", "cohort_data", "exprs", "intersect_test_genes_imputed_cohort1.csv"))
write.csv(intersect_genes_cohort2, file.path(".", "data", "cohort_data", "exprs", "intersect_test_genes_imputed_cohort2.csv"))

# Save common_genes risk based splits
write.csv(intersect_genes_cohort1_low_risk, file.path(".", "data", "cohort_data", "exprs", "intersect_genes_test_cohort1_low_risk.csv"))
write.csv(intersect_genes_cohort1_high_risk, file.path(".", "data", "cohort_data", "exprs", "intersect_genes_test_cohort1_high_risk.csv"))
write.csv(intersect_genes_cohort2_low_risk, file.path(".", "data", "cohort_data", "exprs", "intersect_genes_test_cohort2_low_risk.csv"))
write.csv(intersect_genes_cohort2_high_risk, file.path(".", "data", "cohort_data", "exprs", "intersect_genes_test_cohort2_high_risk.csv"))



# Save all_genes splits
write.csv(all_genes_cohort1, file.path(".", "data", "cohort_data", "exprs", "all_genes_test_cohort1.csv"))
write.csv(all_genes_cohort2, file.path(".", "data", "cohort_data", "exprs", "all_genes_test_cohort2.csv"))

# Save score splits
write.csv(scores_cohort1, file.path(".", "data", "scores", "test_scores_cohort1.csv"))
write.csv(scores_cohort2, file.path(".", "data", "scores", "test_scores_cohort2.csv"))




