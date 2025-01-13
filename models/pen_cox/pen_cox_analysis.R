
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
    results_aggr <- results %>% group_by(model) %>% summarise(mean = mean(ci_min), sd = sd(ci_min))
    return(results_aggr)
}

combine_results <- function(results_nstd, results_test_1, results_test_2){
    # TODO: Join on file name, mean perf -test als neue column
}

# -------------------- functions to load feat. imp from model
load_feat_imp <- function(model_path, flag_lmbd){
    #model = load("pen_lasso_exprs_imputed_pData.Rdata")
    load(model_path)
    coefs <- as.matrix(coef(final_model, s = flag_lmbd))
    non_zero_coefs <- coefs[coefs != 0, , drop = FALSE]  # Extract non-zero coefficients
    #print(non_zero_coefs)
    return(non_zero_coefs)
}


# ------------------------------------------------------------------------------------------------------------------
# --------------------- load and inspect performance
results_path_nstd <- "models\\pen_cox\\results\\results"
combined_results_nstd <- load_all_results(results_path = results_path_nstd)

combined_results_aggr <- aggregate_results(combined_results_nstd)
print(combined_results_aggr)

# ACTUALLY: SAVE COMBINED RESULTS TO CSV
final_results_path <- 'results_modelling\\pen_cox.csv'
write.csv(combined_results_aggr, final_results_path)
#results_testing_1 <- read.csv("models/pen_cox/results_testing/co1.csv")
#results_testing_2 <- read.csv("models/pen_cox/results_testing/co1.csv")
#combine_results <- combine_results(combined_results_aggr, results_testing_1, results_testing_2)


#---------------------- get feature imp 
best_model_csv <- combined_results_aggr[combined_results_aggr$mean == max(combined_results_aggr$mean), 'model']
best_model_rdata <- gsub("\\.csv$", ".Rdata", best_model_csv)
model_path <- paste0("models\\pen_cox\\results\\model\\", best_model_rdata)
coefs <- load_feat_imp(model_path, 'lambda.min')

print(coefs)
