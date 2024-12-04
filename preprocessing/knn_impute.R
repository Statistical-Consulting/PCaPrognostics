library(Biobase)
library(impute)
library(caret)

# Load cohort data from RDS file
load_cohorts <- function(rds_file_path) {
  cohorts <- readRDS(file.path("..", "data", rds_file_path))
  return(cohorts)
}

# Prepare cohort data: standardize and find common genes across all cohorts
prepare_cohort_data <- function(cohorts) {
  # Create lists for dataframes and genes
  cohort_dfs <- list()
  gene_lists <- list()
  
  for(cohort_name in names(cohorts)) {
    # Extract and standardize expression data
    exprs_data <- t(exprs(cohorts[[cohort_name]]))
    gene_lists[[cohort_name]] <- colnames(exprs_data)
    
    # Standardize data
    exprs_data <- scale(exprs_data)
    
    cohort_dfs[[cohort_name]] <- as.data.frame(exprs_data)
  }
  
  # Find genes common to all cohorts
  common_genes <- Reduce(intersect, gene_lists)
  
  # Subset dataframes to only include common genes
  cohort_dfs <- lapply(cohort_dfs, function(df) {
    df[, common_genes, drop=FALSE]
  })
  
  return(list(
    dfs = cohort_dfs,
    common_genes = common_genes
  ))
}

# Create masked data for imputation evaluation
create_hidden_data <- function(cohort_dfs, common_genes, prop=0.1) {
  set.seed(42)
  
  # Create storage for hidden data
  hidden_dfs <- list()
  genes_to_hide <- list()
  
  for(cohort_name in names(cohort_dfs)) {
    # Randomly select genes to hide
    n_genes_to_hide <- floor(length(common_genes) * prop)
    genes_to_hide[[cohort_name]] <- sample(common_genes, n_genes_to_hide)
    
    # Store original values and mask them in the main dataset
    hidden_df <- cohort_dfs[[cohort_name]][, genes_to_hide[[cohort_name]], drop=FALSE]
    hidden_dfs[[cohort_name]] <- hidden_df
    
    # Set selected genes to NA in original dataframe
    cohort_dfs[[cohort_name]][, genes_to_hide[[cohort_name]]] <- NA
  }
  
  return(list(
    masked_dfs = cohort_dfs,
    hidden_dfs = hidden_dfs,
    hidden_genes = genes_to_hide
  ))
}

# Merge cohort dataframes into single dataframe
merge_cohort_data <- function(cohort_dfs) {
  merged_df <- do.call(rbind, cohort_dfs)
  return(merged_df)
}

# Evaluate KNN imputation performance for different k values
evaluate_imputation <- function(merged_df, hidden_dfs, k_values) {
  results <- list()
  
  for(k in k_values) {
    # Perform imputation
    imputed_data <- impute.knn(as.matrix(merged_df), k=k)$data
    
    # Collect true and predicted values
    true_values <- c()
    predicted_values <- c()
    
    # Compare imputed values with original values for each cohort
    for(cohort_name in names(hidden_dfs)) {
      hidden_df <- hidden_dfs[[cohort_name]]
      cohort_rows <- grep(paste0("^", cohort_name, "\\."), rownames(merged_df))
      hidden_cols <- colnames(hidden_df)
      
      true_values <- c(true_values, as.vector(unlist(hidden_df)))
      predicted_values <- c(predicted_values, as.vector(imputed_data[cohort_rows, hidden_cols]))
    }
    
    # Calculate RMSE
    rmse <- sqrt(mean((true_values - predicted_values)^2))
    results[[as.character(k)]] <- rmse
  }
  
  return(results)
}

# Main analysis function
main <- function() {
  # Load cohort data
  cohorts <- load_cohorts("PCa_cohorts.Rds")
  
  # Prepare and standardize data
  prepared_data <- prepare_cohort_data(cohorts)
  
  # Create validation dataset by masking values
  hidden_data <- create_hidden_data(prepared_data$dfs, prepared_data$common_genes)
  
  # Merge datasets
  merged_df <- merge_cohort_data(hidden_data$masked_dfs)
  
  # Evaluate imputation for different k values
  k_values <- c(31, 32, 33, 34, 35, 36, 37, 38, 39)
  results <- evaluate_imputation(merged_df, hidden_data$hidden_dfs, k_values)
  
  # Print RMSE results
  cat("\nRMSE for different k values:\n")
  for(k in names(results)) {
    cat(sprintf("k = %s: RMSE = %.4f\n", k, results[[k]]))
  }
  
  # Create plot data
  k_values <- c(31, 32, 33, 34, 35, 36, 37, 38, 39)
  rmse_values <- unlist(results)
  
  # Create barplot
  rmse_plot <- barplot(rmse_values,
                       names.arg = k_values,
                       main = "RMSE by k-value",
                       xlab = "k",
                       ylab = "RMSE",
                       col = "lightblue",
                       border = "black",
                       ylim = c(min(rmse_values)*0.998, max(rmse_values)*1.001))
  
  # Alternatively, create line plot
  rmse_plot <- plot(k_values, rmse_values,
                    type = "b",  # both line and points
                    main = "RMSE by k-value",
                    xlab = "k",
                    ylab = "RMSE",
                    ylim = c(min(rmse_values)*0.998, max(rmse_values)*1.001))
  
  return(list(
    results = results,
    plot = rmse_plot
  ))
}

# Run analysis
analysis_results <- main()


