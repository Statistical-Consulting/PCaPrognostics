library(glmnet)
library(caret)
library(survival)
library(readr)
library(rsample)
library(purrr)
library(SurvMetrics)
library(dplyr)

#' @description Prepares data for modelling
#' @param use_exprs (bool) Wether gene data is used as covariates in general
#' @param use_inter (bool) TRUE: Uses intersection data, FALSE: Uses common genes data
#' @param use_pData (bool) Wether clinical data is used as covariates
#' @param vars_pData (string vector) Column names of clinical variables to be used
#' @param use_aenc (bool) Wether to use the latent representation obtained from the autoencoder
#' @return dataframe of loaded data
prepare_data <- function(use_exprs, use_inter, use_pData, use_aenc = FALSE, vars_pData = NA){
    if(use_exprs){
      if(use_inter)
        exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/intersection/exprs_intersect.csv', lazy = TRUE))
      else 
        exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/common_genes/common_genes_knn_imputed.csv', lazy = TRUE))
      exprs_data[, 1] <- NULL
    }
    df_pData = read.csv2('data/merged_data/pData/imputed/merged_imputed_pData.csv', sep = ',')
    df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
    df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
    df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
    cohort <- sub("\\..*", "", df_pData$X)
    X = df_pData$X
    if(length(vars_pData) != 0){
        relevant_vars <- c('MONTH_TO_BCR', 'BCR_STATUS', vars_pData)
        df_pData <- df_pData[, relevant_vars]

        cat_pData <- df_pData %>%
            as_data_frame() %>%
            mutate_if(is.character, factor) %>%
            select_if(~ is.factor(.) == TRUE)

        num_pData <- df_pData %>%
            as_data_frame() %>%
            mutate_if(is.character, factor) %>%
            select_if(~ is.numeric(.) == TRUE)

        dmy <- dummyVars(" ~ .", data = cat_pData)
        ohenc_pData <- data.frame(predict(dmy, newdata = cat_pData))

        df_pData <- cbind(num_pData, ohenc_pData, cohort, X)
    }
    if(use_pData && use_exprs && !use_aenc){
        return(cbind(df_pData, exprs_data))
    } else if (use_pData && !use_exprs && !use_aenc) {
       return(df_pData)
    } else if (use_pData && use_aenc){
        return(df_pData)
    } else if ((!use_pData && use_aenc)){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort)
        print(colnames(df))
        return(df)
    } else if((!use_pData && use_exprs)){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort, exprs_data)
        return(df)
    }
}

#' @description Loads and combines performance results from CSV files
#' @param results_path (str): Path to the directory containing CSV result files.
#' @return A combined data frame containing model performance results
load_all_results <- function(results_path) {
    csv_files <- list.files(results_path, pattern = "\\.csv$", full.names = TRUE)
    combined_data <- lapply(csv_files, function(file) {
        df <- read.csv(file)
        df$model <- gsub(".csv", "", basename(file))
        df$ci <- df$ci_min
        df$ci_se <- NULL
        df$ci_min <- NULL
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
        df$model_class <- "CoxPH"
        return(df)

  }) %>% bind_rows()
  combined_data[, 1] <- NULL

  return(combined_data)
}

#' @description Aggregates model performance results
#' @param results A data frame containing performance results of the different splits across models
#' @return A data frame with mean and standard deviation of the C-Index across splits
aggregate_results <- function(results) {
    results_aggr <- results %>% group_by(model) %>% summarise(mean = mean(ci_min), sd = sd(ci_min))
    return(results_aggr)
}

#' @description Merge nested cross-validation results with test performance results
#' @param results_nstd Data frame containing nested resampling results.
#' @param results_test Data frame containing test performance results.
#' @return A merged data frame.
combine_results <- function(results_nstd, results_test){
    df <- merge(results_nstd, results_test)
    return(df)
}


#' @description Extract feature importance from a trained model
#' @param final_model A trained glmnet Cox model.
#' @param flag_lmbd Specifies which lambda value to use. Default is 'lambda.min'.
#' @return A data frame containing non-zero feature coeffs
load_feat_imp <- function(final_model, flag_lmbd = 'lambda.min'){
    coefs <- as.matrix(coef(final_model, s = flag_lmbd))
    non_zero_coefs <- coefs[coefs > 0, , drop = FALSE] 
    coef_df <- data.frame(feature = rownames(coefs),
    value = as.numeric(coefs))
    coef_df <- coef_df %>% filter(value != 0)
    return(coef_df)
}

#' @description Extracts feature importance for all models in a given directory
#' @param model_path Path to the directory containing trained models.
#' @return A data frame combining feature importance values for all models.
feat_imp_all_models <- function(model_path){
    files <- list.files(model_path)
    combined_data <- lapply(files, function(file) {
        contains_pData <- grepl("pData", file, ignore.case = TRUE)
        contains_intersection <- grepl("inter|int|intersection", file, ignore.case = TRUE)
        contains_imputed <- grepl("imp|imputed|common", file, ignore.case = TRUE)
        contains_aenc <- grepl("aenc|auto|autoenc", file, ignore.case = TRUE)
        contains_scores <- grepl("score|scores", file, ignore.case = TRUE)

        if(contains_aenc){

        } else {
        load(paste0(model_path, "\\", file))
            
        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder",
        if (contains_scores) "Scores"
        )

        dataset <- paste(components, collapse = "_")

        imps <- load_feat_imp(final_model, 'lambda.min')
        imps$dataset <- dataset
        imps$model_class <- 'CoxPH'#gsub(".Rdata", "", basename(file))
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

        df_pData <- read.csv("data/merged_data/pData/imputed/test_pData_imputed.csv")
        df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
        df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
        df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
        #cohort <- sub("\\..*", "", df_pData$X)
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
                if (grepl("^\\d+$", col)) { 
                    paste0("X", col) 
                } else {
                    col 
                }
                })

        }

        if (contains_pData) {
            relevant_vars <- c('MONTH_TO_BCR', 'BCR_STATUS', "AGE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
            df_pData <- df_pData[, relevant_vars]
            TISSUE.FFPE <- 1
            TISSUE.Fresh_frozen <- 0
            TISSUE.Snap_frozen <- 0
            df_pData <- cbind(df_pData, cohort, TISSUE.FFPE, TISSUE.Snap_frozen, TISSUE.Fresh_frozen)
            if(nrow(data) == 0){
                data <- df_pData
            } else {
                data <- cbind(df_pData, data)
                }
        }
        else {
            MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
            BCR_STATUS <- df_pData$BCR_STATUS
            data <- cbind(MONTH_TO_BCR, BCR_STATUS, cohort, data)
        }
        
        load(paste0(model_path, "\\", file))
        cohorts = unique(data$cohort)
        data_co1 = data %>% filter(cohort == cohorts[1])
        data_co2 = data %>% filter(cohort == cohorts[2])

        y1 <- Surv(data_co1$MONTH_TO_BCR, data_co1$BCR_STATUS)
        y2 <- Surv(data_co2$MONTH_TO_BCR, data_co2$BCR_STATUS)

        data_co1 = as.matrix(data_co1 %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))
        data_co2 = as.matrix(data_co2 %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))

        test_preds1 <- predict(final_model, data_co1,  s = 'lambda.min')
        test_preds2 <- predict(final_model, data_co2,  s = 'lambda.min')
        cindex1 <- apply(test_preds1, 2, glmnet::Cindex, y=y1)
        cindex2 <- apply(test_preds2, 2, glmnet::Cindex, y=y2)

        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder"
        )

        dataset <- paste(components, collapse = "_")

        perf[i, ] <- c(gsub(".Rdata", "", file), 'CoxPH', dataset, cindex1, cindex2)
    }
    return(perf)
}

test_prop_hazards <- function(model_path){
    files <- list.files(model_path)
    perf = setNames(data.frame(matrix(ncol = 2, nrow = length(files))), c("model", "global_p_refit"))
    for (i  in seq_along(files)) {
        file = files[[i]]
        load(paste0(model_path, '\\', file))
        coefs <- as.matrix(coef(final_model, s = 'lambda.min'))
        non_zero_coefs <- coefs[coefs > 0, , drop = FALSE] 

        contains_pData <- grepl("pData", file, ignore.case = TRUE)
        contains_intersection <- grepl("inter|intersection", file, ignore.case = TRUE)
        contains_imputed <- grepl("imp|imputed|common", file, ignore.case = TRUE)
        contains_aenc <- grepl("aenc|auto|autoenc", file, ignore.case = TRUE)

        vars_pData = c("AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
        data <- prepare_data((contains_intersection | contains_imputed), (contains_intersection && !contains_imputed), contains_pData, contains_aenc, vars_pData)

        if (contains_aenc){
            data_path <- paste0('pretrnd_models_ae\\csv\\pretrnd_cmplt.csv') 
            aenc_data <- read.csv(data_path) %>% mutate_if(is.character, factor)
            data <- left_join(data, aenc_data, by = "X")
        }

        data <- as.data.frame(data %>% select(-c(X, cohort)))
        predictors <- rownames(non_zero_coefs)
        
        formula_str <- paste("Surv(MONTH_TO_BCR, BCR_STATUS) ~", paste(predictors, collapse = " + "))
        
        print(formula_str)

        cox_mod_refit <- coxph(as.formula(formula_str), data = data)
        ph_test_refit <- cox.zph(cox_mod_refit)
        refit_p <- ph_test_refit$table["GLOBAL", "p"] 

        components <- c(
        if (contains_pData) "pData",
        if (contains_intersection) "Intersection",
        if (contains_imputed) "Imputed",
        if (contains_aenc) "AutoEncoder"
        )

        dataset <- paste(components, collapse = "_")

        perf[i, ] <- c(dataset, refit_p)
    }
    return(perf)

}


# ------------------------------------------------------------------------------------------------------------------
# --------------------- load and inspect performance
# results_path_nstd <- "models\\pen_cox\\results\\results"
# combined_results_nstd <- load_all_results(results_path = results_path_nstd)
# split_results_path <- 'results_modelling_splits\\splits_coxph.csv'
# write.csv(combined_results_nstd, split_results_path)
# combined_results_aggr <- aggregate_results(combined_results_nstd)
# print(combined_results_aggr)

# test_perf <- test_perf_all_models("models\\pen_cox\\results\\model")
# print(test_perf)

# final_results <- combine_results(combined_results_aggr, test_perf)
# print(final_results)
# final_results_path <- 'results_modelling_ovs\\ov_coxph.csv'
# write.csv(final_results, final_results_path)

# feat_imps <- feat_imp_all_models("models\\pen_cox\\results\\model")
# print(feat_imps)
# feat_imp_path <- 'results_modelling_feat_imp\\feat_imp_pencox.csv'
# write.csv(feat_imps, feat_imp_path)


ps <- test_prop_hazards("models\\pen_cox\\results\\model")
print(ps)
