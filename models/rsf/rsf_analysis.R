library(randomForestSRC)
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
library(pec)

#' @description Loads and combines performance results from CSV files
#' @param results_path (str): Path to the directory containing CSV result files.
#' @return A combined data frame containing model performance results
load_all_results <- function(results_path) {
    csv_files <- list.files(results_path, pattern = "\\.csv$", full.names = TRUE)
    combined_data <- lapply(csv_files, function(file) {
        df <- read.csv(file)
        df$model <- gsub(".csv", "", basename(file))
        # Perform regex searches
        contains_pData <- grepl("pData", file, ignore.case = TRUE)
        contains_intersection <- grepl("inter|intersection", file, ignore.case = TRUE)
        contains_imputed <- grepl("imp|imputed|common", file, ignore.case = TRUE)
        contains_aenc <- grepl("aenc|auto|autoenc", file, ignore.case = TRUE)
        contains_scores <- grepl("score|scores", file, ignore.case = TRUE)

        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder",
        if (contains_scores) "Scores"
        )

        dataset <- paste(components, collapse = "_")
        df$dataset <- dataset
        df$model_class <- 'RSF'
        return(df)

  }) %>% bind_rows()
  combined_data[, 1] <- NULL

  return(combined_data)
}

#' @description Aggregates model performance results
#' @param results A data frame containing performance results of the different splits across models
#' @return A data frame with mean and standard deviation of the C-Index across splits
aggregate_results <- function(results) {
    print(results)
    results_aggr <- results %>% group_by(model) %>% summarise(mean = mean(ci), sd = sd(ci))
    return(results_aggr)
}


#' @description Merge nested cross-validation results with test performance results
#' @param results_nstd Data frame containing nested resampling results.
#' @param results_test Data frame containing test performance results.
#' @return A merged data frame.
combine_results <- function(results_nstd, results_test){
    df <- merge(results_nstd, results_test)
    # df[, 1] <- NULL
    return(df)
}

#' @description Extract feature importance from a trained model
#' @param final_model A trained rsf model.
#' @return A data frame containing non-zero feature vimps
load_feat_imp <- function(final_model){
    vimp_rsf <- vimp(final_model, block.size="ntree", importance="permute")
    feature_imp = vimp_rsf$importance

    importance_df <- data.frame(
    feature = names(feature_imp),
    value = as.numeric(feature_imp)
    )
    importance_df <- importance_df %>% filter(value > 0)
    return(importance_df)
}

#' @description Extracts feature importance for all models in a given directory
#' @param model_path Path to the directory containing trained models.
#' @return A data frame combining feature importance values for all models.
feat_imp_all_models <- function(model_path){
    files <- list.files(model_path)
    combined_data <- lapply(files, function(file) {
        print(file)
        contains_pData <- grepl("pData", file, ignore.case = TRUE)
        contains_intersection <- grepl("inter|intersection", file, ignore.case = TRUE)
        contains_imputed <- grepl("imp|imputed|common", file, ignore.case = TRUE)
        contains_aenc <- grepl("aenc|auto|autoenc", file, ignore.case = TRUE)
        contains_scores <- grepl("score|scores", file, ignore.case = TRUE)

        if (contains_aenc){
        }
        else{

        load(paste0(model_path, "\\", file))

        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder",
        if (contains_scores) "Scores"
        )

        dataset <- paste(components, collapse = "_")

        imps <- load_feat_imp(final_model)
        imps$dataset <- dataset
        imps$model_class <- 'RSF'
        return(imps)
        }

  }) %>% bind_rows()
  return(combined_data)
}

#' @description Computes C-index for all models in a given directory
#' @param model_path Path to the directory containing trained models.
#' @return A data frame containing test performance results for all models.
test_perf_all_models <- function(model_path){
    files <- list.files(model_path)
    perf = setNames(data.frame(matrix(ncol = 5, nrow = length(files))), c("model", "model_class", "dataset", "ci_coh1", "ci_coh2"))
    for (i  in seq_along(files)) {
        #file = "rsf_pData.Rdata"
        file = files[[i]]
        print(file)

        contains_pData <- grepl("pData", file, ignore.case = TRUE)
        contains_intersection <- grepl("inter|intersection", file, ignore.case = TRUE)
        contains_imputed <- grepl("imp|imputed|common", file, ignore.case = TRUE)
        contains_aenc <- grepl("aenc|auto|autoenc", file, ignore.case = TRUE)
        contains_score <- grepl("score|scores", file, ignore.case = TRUE)

        df_pData <- read.csv("data/merged_data/pData/imputed/test_pData_imputed.csv")
        df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
        df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
        df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
        # cohort <- sub("\\..*", "", df_pData$X)
        cohort <- sub("^([^0-9]*\\d).*", "\\1", df_pData$X)

        data <- data.frame()
        if (contains_intersection){
            data <- as.data.frame(read_csv('data/merged_data/exprs/intersection/intersect_test_genes_imputed.csv', lazy = TRUE))
            data[, 1] <- NULL
        } else if (contains_imputed){
            data <- as.data.frame(read_csv('data/merged_data/exprs/common_genes/common_genes_test_imputed.csv', lazy = TRUE))
            data[, 1] <- NULL
        }
        else if (contains_aenc){
            data <- as.data.frame(read_csv('pretrnd_models_ae/csv_eval/pretrnd_cmplt.csv', lazy = TRUE),)
            data[, 1] <- NULL
            colnames(data) <- sapply(colnames(data), function(col) {
                if (grepl("^\\d+$", col)) { # Check if column name represents only digits
                    paste0("X", col) # Prepend 'X' to the number
                } else {
                    col # Keep other names unchanged
                }
                })

        }
        else if (contains_score){
            data <- as.data.frame(read_csv('data/scores/test_scores.csv', lazy = TRUE))
            data[, 1] <- NULL
        }

        if (contains_pData) {
            relevant_vars <- c('MONTH_TO_BCR', 'BCR_STATUS', "AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
            df_pData <- df_pData[, relevant_vars]
            df_pData <- df_pData %>%
                as_data_frame() %>%
                mutate_if(is.character, factor)

            df_pData$cohort <- cohort
            if(nrow(data) == 0){
                data <- df_pData
            } else {
                data <- cbind(df_pData, data)
            }
            
        }
        else {
            MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
            BCR_STATUS <- df_pData$BCR_STATUS
            #MONTH_TO_DOD <- df_pData$MONTH_TO_DOD
            #DOD_STATUS <- df_pData$DOD_STATUS
            data <- cbind(MONTH_TO_BCR, BCR_STATUS, cohort, data)
        }
        
        load(paste0(model_path, "\\", file))
        cohorts = unique(data$cohort)
        
        data_co1 = data %>% filter(cohort == cohorts[1])  %>% select(-c(cohort))
        data_co2 = data %>% filter(cohort == cohorts[2])  %>% select(-c(cohort))

        test_preds1 <- predict(final_model, newdata = data_co1)
        cindex1 <- get.cindex(data_co1$MONTH_TO_BCR, data_co1$BCR_STATUS, -test_preds1$predicted)

        test_preds2 <- predict(final_model, newdata = data_co2)
        cindex2 <- get.cindex(data_co2$MONTH_TO_BCR, data_co2$BCR_STATUS, -test_preds2$predicted)
        
        # Create a vector of components based on conditions
        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder"
        )

        # Join non-empty components with "_" as a separator
        dataset <- paste(components, collapse = "_")
        models <- list("RF" = final_model)

        # Evaluate prediction error curves over a time range
        pec_results1 <- pec(object = models,
                        formula = Surv(MONTH_TO_BCR, BCR_STATUS) ~ 1,
                        data = data_co1,
                        times = seq(0, max(data_co2$MONTH_TO_BCR), by = 10))
        print(pec_results1)
        pec_results2 <- pec(object = models,
                        formula = Surv(MONTH_TO_BCR, BCR_STATUS) ~ 1,
                        data = data_co2,
                        times = seq(0, max(data_co2$MONTH_TO_BCR), by = 10))
        print(pec_results2)

        perf[i, ] <- c(gsub(".Rdata", "", file), 'RSF', dataset, cindex1, cindex2)
    }
    return(perf)
}


# ------------------------------------------------------------------------------------------------------------------
# --------------------- load and inspect performance
# results_path_nstd <- "models\\rsf\\results\\results"
# combined_results_nstd <- load_all_results(results_path = results_path_nstd)
# split_results_path <- 'results_modelling_splits\\splits_rsf.csv'
# write.csv(combined_results_nstd, split_results_path)


# combined_results_aggr <- aggregate_results(combined_results_nstd)
# print(combined_results_aggr)

# # --------------------- Get test performances
# test_perf <- test_perf_all_models("models\\rsf\\results\\model")
# final_results <- combine_results(combined_results_aggr, test_perf)

# final_results_path <- 'results_modelling_ovs\\ov_rsf.csv'
# write.csv(final_results, final_results_path)

feat_imps <- feat_imp_all_models("models\\rsf\\results\\model")
print(feat_imps)
feat_imp_path <- 'results_modelling_feat_imp\\feat_imp_rsf.csv'
write.csv(feat_imps, feat_imp_path)
