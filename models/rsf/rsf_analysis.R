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
        data <- read.csv(file)
        data$model <- basename(file)
        print(data)
        return(data)
  }) %>% bind_rows()

  return(combined_data)
}

aggregate_results <- function(results) {
    results_aggr <- results %>% group_by(model) %>% summarise(mean = mean(ci), sd = sd(ci))
    return(results_aggr)
}

combine_results <- function(results_nstd, results_test_1, results_test_2){
    # TODO: Join on file name, mean perf -test als neue column
}

# -------------------- functions to load feat. imp from model
load_feat_imp <- function(model_path){
    #model = load("pen_lasso_exprs_imputed_pData.Rdata")
    load(model_path)
    vimp_rsf <- vimp(final_model)
    #feature_imp = vimp_rsf$importance
    return(vimp_rsf)
}

# ------------------- Get perfroamnce across all models for that model class
test_perf_all_models <- function(model_path){
    files <- list.files(model_path)

    for (file in files) {
        file = "rsf_pData.Rdata"
        print(file)

        contains_pData <- grepl("pData", file, ignore.case = TRUE)
        contains_intersection <- grepl("inter|intersection", file, ignore.case = TRUE)
        contains_imputed <- grepl("imp|imputed|common", file, ignore.case = TRUE)
        contains_aenc <- grepl("aenc|auto|autoenc", file, ignore.case = TRUE)

        df_pData <- read.csv("data/merged_data/pData/original/test_pData.csv")
        df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
        df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
        df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
        cohort <- sub("\\..*", "", df_pData$X)

        data <- data.frame()
        if (contains_intersection){
            data <- as.data.frame(read_csv('data/merged_data/exprs/intersection/intersect_test_genes_imputed.csv', lazy = TRUE))
            data[, 1] <- NULL
        } else if (contains_imputed){
            data <- as.data.frame(read_csv('data/merged_data/exprs/common_genes/common_genes_test_imputed.csv', lazy = TRUE))
            data[, 1] <- NULL
        }
        else if (contains_aenc){
            # HIER NOCH RICHTIGEN PFAD EINFÜGEN
            data <- as.data.frame(read_csv('data/merged_data/exprs/common_genes/common_genes_knn_imputed.csv', lazy = TRUE))
            data[, 1] <- NULL
        }
        if (contains_pData) {
            relevant_vars <- c('X', 'MONTH_TO_BCR', 'BCR_STATUS', "MONTH_TO_DOD", "DOD_STATUS", "AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
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
            MONTH_TO_DOD <- df_pData$MONTH_TO_DOD
            DOD_STATUS <- df_pData$DOD_STATUS
            data <- cbind(MONTH_TO_BCR, BCR_STATUS, MONTH_TO_DOD, DOD_STATUS, cohort, data)
        }
        
        load(paste0(model_path, "\\", file))

        # split data into cohorts
        cohorts = unique(data$cohort)
        data_co1 = data %>% filter(cohort == cohorts[1]) %>% select(-c(cohort, DOD_STATUS, MONTH_TO_DOD))
        data_co2 = data %>% filter(cohort == cohorts[2]) %>% select(-c(cohort, DOD_STATUS, MONTH_TO_DOD))

        test_preds1 <- predict(final_model, newdata = data_co1)

        print(length(data_co1$MONTH_TO_BCR))
        print(length(data_co1$BCR_STATUS))
        print(length(test_preds1$predicted))

        print(sum(is.na(data_co1$MONTH_TO_BCR)))
        print(sum(is.na(data_co1$BCR_STATUS)))
        print(sum(is.na(test_preds1$predicted)))
        print(colSums(is.na(data_co1)))

        cindex1 <- get.cindex(data_co1$MONTH_TO_BCR, data_co1$BCR_STATUS, -test_preds1$predicted)
        print(cindex1)

        #test_preds2 <- predict(final_model, newdata = data_co2)
        #cindex2 <- get.cindex(data_co2$MONTH_TO_DOD, data_co2$DOD_STATUS, -test_preds2$predicted)
        #print(cindex2)
    }
    # save data frame
}


# ------------------------------------------------------------------------------------------------------------------
# --------------------- load and inspect performance
results_path_nstd <- "models\\rsf\\results\\results"
combined_results_nstd <- load_all_results(results_path = results_path_nstd)
combined_results_aggr <- aggregate_results(combined_results_nstd)
print(combined_results_aggr)

# --------------------- Get test performances
test_perf_all_models("models\\rsf\\results\\model")

# ACTUALLY: SAVE COMBINED RESULTS TO CSV
final_results_path <- 'results_modelling\\rsf.csv'
write.csv(combined_results_aggr, final_results_path)

#results_testing_1 <- read.csv("models/rsf/results_testing/co1.csv")
#results_testing_2 <- read.csv("models/rsf/results_testing/co1.csv")
#combine_results <- combine_results(combined_results_aggr, results_testing_1, results_testing_2)

#---------------------- get feature imp 
#best_model_csv <- combined_results_aggr[combined_results_aggr$mean == max(combined_results_aggr$mean), 'model']
best_model_csv <- 'rsf_pData.csv'
best_model_rdata <- gsub("\\.csv$", ".Rdata", best_model_csv)
model_path <- paste0("models\\rsf\\results\\model\\", best_model_rdata)
print(model_path)
vimps_rsf <- load_feat_imp(model_path)
print(vimps_rsf$importance)
#plot(vimps_rsf)
