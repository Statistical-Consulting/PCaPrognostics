
library(glmnet)
library(dplyr)
library(readr)
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
    results_aggr <- results %>% group_by(model) %>% summarise(mean = mean(ci_min), sd = sd(ci_min))
    return(results_aggr)
}

combine_results <- function(results_nstd, results_test){
    df <- merge(results_nstd, results_test)
    df[, 1] <- NULL

    return(df)
}
# -------------------- functions to load feat. imp from model
load_feat_imp <- function(final_model, flag_lmbd = 'lambda.min'){
    #model = load("pen_lasso_exprs_imputed_pData.Rdata")
    #load(model_path)
    coefs <- as.matrix(coef(final_model, s = flag_lmbd))
    non_zero_coefs <- coefs[coefs > 0, , drop = FALSE]  # Extract non-zero coefficients
    #print(non_zero_coefs)
    #return(non_zero_coefs)
    coef_df <- data.frame(feature = rownames(coefs),
    values = as.numeric(coefs))
    coef_df <- coef_df %>% filter(values > 0)
    return(coef_df)
}

feat_imp_all_models <- function(model_path){
    files <- list.files(model_path)
    combined_data <- lapply(files, function(file) {
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

        imps <- load_feat_imp(final_model, 'lambda.min')
        imps$dataset <- dataset
        imps$model <- 'pen_cox'#gsub(".Rdata", "", basename(file))
        return(imps)

  }) %>% bind_rows()
  # combined_data[, 1] <- NULL

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
            relevant_vars <- c('MONTH_TO_BCR', 'BCR_STATUS', "AGE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
            df_pData <- df_pData[, relevant_vars]
            TISSUE.FFPE <- 1
            TISSUE.Fresh_frozen <- 0
            TISSUE.Snap_frozen <- 0
            #levels(df_pData$TISSUE) <- c("FFPE", "Fresh_frozen","Snap_frozen")

            # cat_pData <- df_pData %>%
            # as_data_frame() %>%
            # mutate_if(is.character, factor) %>%
            # select_if(~ is.factor(.) == TRUE)

            # print("hi")
            # num_pData <- df_pData %>%
            #     as_data_frame() %>%
            #     mutate_if(is.character, factor) %>%
            #     select_if(~ is.numeric(.) == TRUE)

            # print(str(cat_pData))
            # dmy <- dummyVars(" ~ .", data = cat_pData)
            # ohenc_pData <- data.frame(predict(dmy, newdata = cat_pData))
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
        #print(str(data))
        # split data into cohorts
        cohorts = unique(data$cohort)
        print(cohorts)
        # data_co1 = data %>% filter(cohort == cohorts[1])
        # data_co2 = data %>% filter(cohort == cohorts[2])

        # month_bcr1 = data_co1$MONTH_TO_BCR
        # month_bcr2 = data_co2$MONTH_TO_BCR
        # status_bcr1 = data_co1$BCR_STATUS
        # status_bcr2 = data_co2$BCR_STATUS

        data_co1 = data %>% filter(cohort == cohorts[1])
        data_co2 = data %>% filter(cohort == cohorts[2])

        #print(str(data_co1))
        #print(str(data_co2))

        y1 <- Surv(data_co1$MONTH_TO_BCR, data_co1$BCR_STATUS)
        y2 <- Surv(data_co2$MONTH_TO_BCR, data_co2$BCR_STATUS)

        data_co1 = as.matrix(data_co1 %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))
        data_co2 = as.matrix(data_co2 %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))


        test_preds1 <- predict(final_model, data_co1,  s = 'lambda.min')
        test_preds2 <- predict(final_model, data_co2,  s = 'lambda.min')
        cindex1 <- apply(test_preds1, 2, glmnet::Cindex, y=y1)
        cindex2 <- apply(test_preds2, 2, glmnet::Cindex, y=y2)
        # cindex2 <- NA

        print(cindex1)
        print(cindex2)

        perf[i, ] <- c(gsub(".Rdata", "", file), cindex1, cindex2)
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
# write.csv(combined_results_aggr, final_results_path)

feat_imps <- feat_imp_all_models("models\\pen_cox\\results\\model")
print(feat_imps)
feat_imp_path <- 'results_modelling_feat_imp\\feat_imp_pencox.csv'
write.csv(feat_imps, feat_imp_path)


# final_results_path <- 'results_modelling\\pen_cox.csv'
# write.csv(final_results, final_results_path)


# ACTUALLY: SAVE COMBINED RESULTS TO CSV
#final_results_path <- 'results_modelling\\pen_cox.csv'
#write.csv(combined_results_aggr, final_results_path)
#results_testing_1 <- read.csv("models/pen_cox/results_testing/co1.csv")
#results_testing_2 <- read.csv("models/pen_cox/results_testing/co1.csv")
#combine_results <- combine_results(combined_results_aggr, results_testing_1, results_testing_2)


#---------------------- get feature imp 
# best_model_csv <- combined_results_aggr[combined_results_aggr$mean == max(combined_results_aggr$mean), 'model']
# best_model_rdata <- gsub("\\.csv$", ".Rdata", best_model_csv)
# model_path <- paste0("models\\pen_cox\\results\\model\\", best_model_rdata)
# coefs <- load_feat_imp(model_path, 'lambda.min')
# print(coefs)