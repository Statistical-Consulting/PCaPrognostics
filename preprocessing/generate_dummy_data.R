# Load the required package
library(Biobase)

# Load the original list of ExpressionSets
cohorts <- readRDS(file.path(".", "data", "PCa_cohorts.Rds"))
dummy_cohorts <- cohorts

# Columns to keep in pData
desired_columns <- c('AGE', 'TISSUE', 'PATH_T_STAGE', 'GLEASON_SCORE', 
                     'PRE_OPERATIVE_PSA', 'MONTH_TO_BCR', 'CLIN_T_STAGE', 'BCR_STATUS')

# Lists to store modified components
modified_pData <- list()
modified_fData <- list()
modified_exprs <- list()
cohort_names <- list()

# Loop through each cohort
for (i in seq_along(cohorts)) {
  cohort <- dummy_cohorts[[i]]
  
  # Rename the cohort
  cohort_name <- paste0("Cohort_", i)
  cohort_names[[i]] <- cohort_name
  
  # 1. Modify pData: Keep only selected columns
  pheno_data <- pData(cohort)
  common_columns <- intersect(colnames(pheno_data), desired_columns)
  modified_pData[[cohort_name]] <- pheno_data[, common_columns, drop = FALSE]
  
  # 2. Modify exprs: Retain 50 random rows (features)
  exprs_data <- exprs(cohort)
  random_rows <- sample(nrow(exprs_data), 50)
  modified_exprs[[cohort_name]] <- exprs_data[random_rows, , drop = FALSE]
  
  # 3. Modify fData: Keep only the selected features
  feature_data <- fData(cohort)
  modified_fData[[cohort_name]] <- feature_data[random_rows, , drop = FALSE]
}

# Assign modified cohort names to the lists
names(modified_pData) <- cohort_names
names(modified_fData) <- cohort_names
names(modified_exprs) <- cohort_names

# Lists for renamed data
renamed_pData <- list()
renamed_fData <- list()
renamed_exprs <- list()

for (name in names(modified_exprs)) {
  cohort_index <- gsub("Cohort_", "", name)  # Extract the cohort number
  cohort_index <- as.integer(cohort_index)  # Ensure it's numeric
  
  # Rename rows and columns in exprs
  exprs_data <- modified_exprs[[name]]
  rownames(exprs_data) <- paste0("Gene_", seq_len(nrow(exprs_data)))
  colnames(exprs_data) <- paste0("Patient_", cohort_index, "_", seq_len(ncol(exprs_data)))
  renamed_exprs[[name]] <- exprs_data
  
  # Rename rows in fData
  fdata <- modified_fData[[name]]
  rownames(fdata) <- paste0("Gene_", seq_len(nrow(fdata)))
  renamed_fData[[name]] <- fdata
  
  # Rename rows in pData
  pdata <- modified_pData[[name]]
  rownames(pdata) <- paste0("Patient_", cohort_index, "_", seq_len(nrow(pdata)))
  renamed_pData[[name]] <- pdata
}


# Replace all entries in fData with "dummy info"
dummy_fData <- list()
for (name in names(renamed_fData)) {
  fdata <- renamed_fData[[name]]
  fdata[,] <- "dummy info"  # Replace all values with "dummy info"
  dummy_fData[[name]] <- fdata
}

# List to store anonymized pData
anonymized_pData <- list()

# Anonymize pData by sampling unique values for each column
for (name in names(renamed_pData)) {
  pdata <- renamed_pData[[name]]
  anonymized_pdata <- pdata
  
  for (col in colnames(pdata)) {
    unique_values <- unique(pdata[[col]])  # Get unique values from the column
    anonymized_pdata[[col]] <- sample(unique_values, nrow(pdata), replace = TRUE)  # Sample new values
  }
  
  anonymized_pData[[name]] <- anonymized_pdata
}

# List to store transformed exprs
transformed_exprs <- list()

set.seed(42)  # Ensure reproducible random numbers

# Apply transformations to exprs: multiply and divide each value by random numbers
for (name in names(renamed_exprs)) {
  exprs_data <- renamed_exprs[[name]]
  
  transformed_data <- apply(exprs_data, c(1, 2), function(x) {
    num1 <- sample(10000:20000, 1)  # First random number
    num2 <- sample(10000:20000, 1)  # Second random number
    (x * num1) / num2  # Apply transformation
  })
  
  transformed_exprs[[name]] <- transformed_data
}

# List to store the final ExpressionSet objects
final_expression_sets <- list()

# Create ExpressionSets for each cohort
for (name in names(transformed_exprs)) {
  eset <- ExpressionSet(
    assayData = transformed_exprs[[name]],  # Transformed exprs data
    phenoData = AnnotatedDataFrame(anonymized_pData[[name]]),  # Anonymized pData
    featureData = AnnotatedDataFrame(dummy_fData[[name]])  # Dummy fData
  )
  
  final_expression_sets[[name]] <- eset
}

# Save the final ExpressionSets to an .Rds file
saveRDS(final_expression_sets, file = "data/dummy_expression_sets.Rds")



