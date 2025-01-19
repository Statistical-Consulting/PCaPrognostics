
library(glmnet)
library(dplyr)
# Load necessary libraries
library(randomForestSRC)
library(caret)
library(dplyr)
library(survival)
library(dplyr)
library(prioritylasso)
library(survival)
library(readr)
library(rsample)
library(purrr)
library(SurvMetrics)
library(glmnet)

load_all_results <- function(results_path) {
    csv_files <- list.files(results_path, pattern = "\\.csv$", full.names = TRUE)
    combined_data <- lapply(csv_files, function(file) {
        data <- read.csv(file)
        data$model <- gsub(".csv", "", basename(file))
        data$dataset <- "gene_blocks"
        return(data)
  }) %>% bind_rows()
  combined_data[, 1] <- NULL

  return(combined_data)
}

aggregate_results <- function(results) {
    results_aggr <- results %>% group_by(model) %>% summarise(mean = mean(ci), sd = sd(ci))
    return(results_aggr)
}

combine_results <- function(results_nstd, results_test){
    df <- merge(results_nstd, results_test)
    df[, 1] <- NULL

    return(df)
}
# -------------------- functions to load feat. imp from model
load_feat_imp <- function(model_path){
    #model = load("pen_lasso_exprs_imputed_pData.Rdata")
    load(model_path)
    coefs <- coef(final_model)
    coefs_vals <- coefs[[1]]
    coefs_df <- data.frame(feature = names(coefs_vals), value = as.numeric(coefs_vals))
    non_zero_coefs <- coefs_df %>% filter(value > 0)  # Extract non-zero coefficients
    #print(final_model.nzero)
    return(non_zero_coefs)
}

# ------------------- Get perfroamnce across all models for that model class
test_perf_all_models <- function(model_path){
    files <- list.files(model_path)
    perf = setNames(data.frame(matrix(ncol = 3, nrow = length(files))), c("model", "ci_cohort1", "ci_cohort2"))
    for (i  in seq_along(files)) {
        #file = "rsf_pData.Rdata"
        file = files[[i]]
        print(file)
        
        load(paste0(model_path, "\\", file))


        relevant_vars <- c('MONTH_TO_BCR', 'BCR_STATUS', "AGE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
        TISSUE_FFPE <- 1
        TISSUE_Fresh_frozen <- 0
        TISSUE_Snap_frozen <- 0


        df_pData <- read.csv("data/merged_data/pData/imputed/test_pData_imputed.csv")
        df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
        df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
        df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
        #cohort <- sub("\\..*", "", df_pData$X)
        cohort <- sub("^([^0-9]*\\d).*", "\\1", df_pData$X)
        df_pData$cohort <- cohort
        cohorts = unique(cohort)
        print(cohorts)
        df_blockwise_data = read_csv('models/prio_lasso/df_block_data_100_300.csv', lazy = TRUE)
        df_blockwise_data[, 'index'] <- NULL
        df_blockwise_data[, 'cohort'] <- NULL

        columns_to_select <- colnames(df_blockwise_data)
        columns_to_select <- c('MONTH_TO_BCR', 'BCR_STATUS', columns_to_select)

        # TODO: load the correct path
        # data <- as.data.frame(read_csv('data/merged_data/exprs/all_genes/all_genes_test.csv', lazy = TRUE))

        # -------------------- load coh 1
        exprs1 <- read_csv('data\\cohort_data\\exprs\\test_cohort_1.csv', lazy = TRUE)
        exprs1[, 1] <- NULL
        pData1 <- df_pData %>% filter(cohort == cohorts[1])
        pData1 <- pData1[, relevant_vars]

        pData1 <- cbind(pData1, TISSUE_FFPE, TISSUE_Snap_frozen, TISSUE_Fresh_frozen)
        data_co1 <- cbind(pData1, exprs1)
        # Identify missing columns
        missing_cols <- setdiff(columns_to_select, names(data_co1))
        # Add missing columns to df2, filling them with NA
        data_co1[missing_cols] <- 0
        data_co1 <- data_co1 %>% mutate(across(all_of(missing_cols), as.numeric))
        # Reorder df2 to match df1's column order
        # data_co1 <- data_co1[, columns_to_select]
        # print(data_co1 %>% select(where(~ !is.numeric(.))) %>% ncol())
        data_co1 <- data_co1 %>% select(columns_to_select)

        y1 <- Surv(data_co1$MONTH_TO_BCR, data_co1$BCR_STATUS)
        data_co1 = as.matrix(data_co1 %>% select(-c(MONTH_TO_BCR, BCR_STATUS)))
        print(sum(rowSums(is.na(data_co1)) == ncol(data_co1)))
        # -------------------- load coh 2
        exprs2 <- read_csv('data\\cohort_data\\exprs\\test_cohort_2.csv', lazy = TRUE)
        exprs2[, 1] <- NULL
        pData2 <- df_pData %>% filter(cohort == cohorts[2])
        pData2 <- pData2[, relevant_vars]

        pData2 <- cbind(pData2, TISSUE_FFPE, TISSUE_Snap_frozen, TISSUE_Fresh_frozen)
        data_co2 <- cbind(pData2, exprs2)
        # Identify missing columns
        missing_cols2 <- setdiff(columns_to_select, names(data_co2))
        # Add missing columns to df2, filling them with NA
        data_co2[missing_cols2] <- 0
        data_co2 <- data_co2 %>% mutate(across(all_of(missing_cols2), as.numeric))
        # Reorder df2 to match df1's column order
        # data_co1 <- data_co1[, columns_to_select]
        # print(data_co1 %>% select(where(~ !is.numeric(.))) %>% ncol())
        data_co2 <- data_co2 %>% select(columns_to_select)

        y2 <- Surv(data_co2$MONTH_TO_BCR, data_co2$BCR_STATUS)
        data_co2 = as.matrix(data_co2 %>% select(-c(MONTH_TO_BCR, BCR_STATUS)))
        # --------------------------

        test_preds1 <- predict(final_model, data_co1,  type = "response", handle.missingtestdata = c("none"))
        test_preds2 <- predict(final_model, data_co2,  type = "response", handle.missingtestdata = c("none"))
        cindex1 <- apply(test_preds1, 2, glmnet::Cindex, y=y1)
        cindex2 <- apply(test_preds2, 2, glmnet::Cindex, y=y2)

        print(cindex1)
        print(cindex2)

        perf[i, ] <- c(gsub(".Rdata", "", file), cindex1, cindex2)
    }
    return(perf)
}



# ------------------------------------------------------------------------------------------------------------------
# --------------------- load and inspect performance
results_path_nstd <- "models\\prio_lasso\\results\\results"
combined_results_nstd <- load_all_results(results_path = results_path_nstd)
split_results_path <- 'results_modelling_splits\\ov_prio.csv'
write.csv(combined_results_nstd, split_results_path)


combined_results_aggr <- aggregate_results(combined_results_nstd)
print(combined_results_aggr)

test_perf <- test_perf_all_models("models\\prio_lasso\\results\\model")
final_results <- combine_results(combined_results_aggr, test_perf)

final_results_path <- 'results_modelling_ovs\\splits_prio.csv'
write.csv(final_results, final_results_path)

# ACTUALLY: SAVE COMBINED RESULTS TO CSV
#final_results_path <- 'results_modelling\\pen_cox.csv'
#write.csv(combined_results_aggr, final_results_path)
#results_testing_1 <- read.csv("models/pen_cox/results_testing/co1.csv")
#results_testing_2 <- read.csv("models/pen_cox/results_testing/co1.csv")
#combine_results <- combine_results(combined_results_aggr, results_testing_1, results_testing_2)


#---------------------- get feature imp 
#best_model_csv <- combined_results_aggr[combined_results_aggr$mean == max(combined_results_aggr$mean), 'model']
#best_model_rdata <- gsub("\\.csv$", ".Rdata", best_model_csv)
model_path <- paste0("models\\prio_lasso\\results\\model\\", 'prioLasso_100_300.Rdata')
coefs <- load_feat_imp(model_path)
print(coefs)
