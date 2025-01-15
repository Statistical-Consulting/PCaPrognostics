
library(glmnet)
library(dplyr)

load_all_results <- function(results_path) {
    csv_files <- list.files(results_path, pattern = "\\.csv$", full.names = TRUE)
    combined_data <- lapply(csv_files, function(file) {
        data <- read.csv(file)
        data$model <- basename(file)
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
    coefs <- coef(final_model)
    coefs_vals <- coefs[[1]]
    coefs_df <- data.frame(feature = names(coefs_vals), value = as.numeric(coefs_vals))
    non_zero_coefs <- coefs_df %>% filter(value > 0)  # Extract non-zero coefficients
    print(final_model.nzero)
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

        df_pData <- read.csv("data/merged_data/pData/imputed/test_pData.csv")
        df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
        df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
        df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
        cohort <- sub("\\..*", "", df_pData$X)

        df_blockwise_data = read_csv('models/prio_lasso/df_block_data.csv', lazy = TRUE)
        columns_to_select <- colnames(df_blockwise_data)
        # TODO: load the correct path
        data <- as.data.frame(read_csv('data/merged_data/exprs/all_genes/all_genes.csv', lazy = TRUE))

        relevant_vars <- c('MONTH_TO_BCR', 'BCR_STATUS', "AGE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
        df_pData <- df_pData[, relevant_vars]
        TISSUE.FFPE <- 1
        TISSUE.Fresh_frozen <- 0
        TISSUE.Snap_frozen <- 0
        df_pData <- cbind(df_pData, cohort, TISSUE.FFPE, TISSUE.Snap_frozen, TISSUE.Fresh_frozen)

        data <- cbind(df_pData, data, cohort)
        data <- data %>% select(all_of(columns_to_select))

        
        load(paste0(model_path, "\\", file))
        print(str(data))
        # split data into cohorts
        cohorts = unique(data$cohort)

        # data_co1 = data %>% filter(cohort == cohorts[1])
        # data_co2 = data %>% filter(cohort == cohorts[2])

        # month_bcr1 = data_co1$MONTH_TO_BCR
        # month_bcr2 = data_co2$MONTH_TO_BCR
        # status_bcr1 = data_co1$BCR_STATUS
        # status_bcr2 = data_co2$BCR_STATUS

        data_co1 = data %>% filter(cohort == cohorts[1])
        data_co2 = data %>% filter(cohort == cohorts[2])

        print(str(data_co1))
        print(sum(is.na(data_co1)))
        print(sum(is.na(data_co2)))
        #print(str(data_co2))

        y1 <- Surv(data_co1$MONTH_TO_BCR, data_co1$BCR_STATUS)
        y2 <- Surv(data_co2$MONTH_TO_BCR, data_co2$BCR_STATUS)

        data_co1 = as.matrix(data_co1 %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))
        data_co2 = as.matrix(data_co2 %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))


        test_preds1 <- predict(final_model, data_co1,  s = 'lambda.min')
        test_preds2 <- predict(final_model, data_co2,  s = 'lambda.min')
        cindex1 <- apply(test_preds1, 2, glmnet::Cindex, y=y1)
        #outer_cindex2 <- apply(test_preds2, 2, glmnet::Cindex, y=y2)
        cindex2 <- NA

        perf[i, ] <- c(file, cindex1, cindex2)
    }
    return(perf)
}



# ------------------------------------------------------------------------------------------------------------------
# --------------------- load and inspect performance
results_path_nstd <- "models\\prio_lasso\\results\\"
combined_results_nstd <- load_all_results(results_path = results_path_nstd)

combined_results_aggr <- aggregate_results(combined_results_nstd)
print(combined_results_aggr)

#test_perf <- test_perf_all_models("models\\prio_lasso\\model")

# ACTUALLY: SAVE COMBINED RESULTS TO CSV
#final_results_path <- 'results_modelling\\pen_cox.csv'
#write.csv(combined_results_aggr, final_results_path)
#results_testing_1 <- read.csv("models/pen_cox/results_testing/co1.csv")
#results_testing_2 <- read.csv("models/pen_cox/results_testing/co1.csv")
#combine_results <- combine_results(combined_results_aggr, results_testing_1, results_testing_2)


#---------------------- get feature imp 
#best_model_csv <- combined_results_aggr[combined_results_aggr$mean == max(combined_results_aggr$mean), 'model']
#best_model_rdata <- gsub("\\.csv$", ".Rdata", best_model_csv)
#model_path <- paste0("models\\prio_lasso\\model\\", 'prioLasso_intersection_pData.Rdata')
coefs <- load_feat_imp(model_path)
print(coefs)
