library(randomForestSRC)
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



# ------------------ functions to load perf. results
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

        # Create a vector of components based on conditions
        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder",
        if (contains_scores) "Scores"
        )

        # Join non-empty components with "_" as a separator
        dataset <- paste(components, collapse = "_")

        # Assign dataset to a dataframe column
        df$dataset <- dataset
        return(df)

  }) %>% bind_rows()
  combined_data[, 1] <- NULL

  return(combined_data)
}

aggregate_results <- function(results) {
    print(results)
    results_aggr <- results %>% group_by(model) %>% summarise(mean = mean(ci), sd = sd(ci))
    return(results_aggr)
}

combine_results <- function(results_nstd, results_test){
    df <- merge(results_nstd, results_test)
    df[, 1] <- NULL
    return(df)
}

# -------------------- functions to load feat. imp from model
load_feat_imp <- function(final_model){
    # model = load("pen_lasso_exprs_imputed_pData.Rdata")
    # load(model_path)
    vimp_rsf <- vimp(final_model)
    feature_imp = vimp_rsf$importance

    # Convert to a data frame
    importance_df <- data.frame(
    feature = names(feature_imp),
    value = as.numeric(feature_imp)
    )
    importance_df <- importance_df[order(-importance_df$value), ]
    importance_df <- importance_df %>% filter(value > 0)
    return(importance_df)
}

feat_imp_all_models <- function(model_path){
    files <- list.files(model_path)
    combined_data <- lapply(files, function(file) {
        print(file)
        load(paste0(model_path, "\\", file))
        # Perform regex searches
        contains_pData <- grepl("pData", file, ignore.case = TRUE)
        contains_intersection <- grepl("inter|intersection", file, ignore.case = TRUE)
        contains_imputed <- grepl("imp|imputed|common", file, ignore.case = TRUE)
        contains_aenc <- grepl("aenc|auto|autoenc", file, ignore.case = TRUE)
        contains_scores <- grepl("score|scores", file, ignore.case = TRUE)

        # Create a vector of components based on conditions
        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder",
        if (contains_scores) "Scores"
        )

        # Join non-empty components with "_" as a separator
        dataset <- paste(components, collapse = "_")

        imps <- load_feat_imp(final_model)
        imps$dataset <- dataset
        imps$model <- 'rsf'#gsub(".Rdata", "", basename(file))
        return(imps)

  }) %>% bind_rows()
  combined_data[, 1] <- NULL

  return(combined_data)
}

# ------------------- Get perfroamnce across all models for that model class
test_perf_all_models <- function(model_path){
    files <- list.files(model_path)
    perf = setNames(data.frame(matrix(ncol = 3, nrow = length(files))), c("model", "ci_cohort1", "ci_cohort2"))
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
        
        print(cindex1)
        print(cindex2)
        perf[i, ] <- c(gsub(".Rdata", "", file), cindex1, cindex2)
    }
    return(perf)
}


# ------------------------------------------------------------------------------------------------------------------
# --------------------- load and inspect performance
results_path_nstd <- "models\\rsf\\results\\results"
combined_results_nstd <- load_all_results(results_path = results_path_nstd)
split_results_path <- 'results_modelling_splits\\splits_rsf.csv'
write.csv(combined_results_nstd, split_results_path)


combined_results_aggr <- aggregate_results(combined_results_nstd)
print(combined_results_aggr)

# --------------------- Get test performances
test_perf <- test_perf_all_models("models\\rsf\\results\\model")
final_results <- combine_results(combined_results_aggr, test_perf)

final_results_path <- 'results_modelling_ovs\\ov_rsf.csv'
write.csv(final_results, final_results_path)

# feat_imps <- feat_imp_all_models("models\\rsf\\results\\model")
# print(feat_imps)
# feat_imp_path <- 'results_modelling_feat_imp\\feat_imp_rsf.csv'
# write.csv(combined_results_aggr, feat_imp_path)
