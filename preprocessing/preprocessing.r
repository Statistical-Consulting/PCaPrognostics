library(Biobase)
library(dplyr)
library(impute)

# Function to load and convert ExpressionSet objects
load_cohorts <- function(rds_file_paths) {
  all_cohorts <- list()
  for (rds_file_path in rds_file_paths) {
    cohorts <- readRDS(file.path(".", "data", rds_file_path))
    cohorts_list <- lapply(cohorts, function(eset) {
      list(
        pData = as.data.frame(pData(eset)),
        fData = as.data.frame(fData(eset)),
        exprs = as.data.frame(exprs(eset))
      )
    })
    all_cohorts <- c(all_cohorts, cohorts_list)
  }
  return(all_cohorts)
}

# Function to standardize data (for both expression and numeric pData)
standardize <- function(z) {
  rowmean <- apply(z, 1, mean, na.rm = TRUE)
  rowsd <- apply(z, 1, sd, na.rm = TRUE)
  rv <- sweep(z, 1, rowmean, "-")
  rv <- sweep(rv, 1, rowsd, "/")
  return(rv)
}

standardize_pdata_numeric <- function(pdata, numeric_cols) {
  # Überprüfen, ob die Spalten existieren
  existing_cols <- intersect(numeric_cols, names(pdata))
  if(length(existing_cols) == 0) return(pdata)
  
  # Konvertiere zu numerisch und prüfe auf NA
  numeric_data <- as.matrix(sapply(pdata[, existing_cols, drop = FALSE], as.numeric))

  
  # Nur standardisieren, wenn es mindestens zwei Werte gibt
  for(i in 1:ncol(numeric_data)) {
    if(sum(!is.na(numeric_data[,i])) >= 2) {
      col_mean <- mean(numeric_data[,i], na.rm = TRUE)
      col_sd <- sd(numeric_data[,i], na.rm = TRUE)
      if(col_sd > 0) {
        numeric_data[,i] <- (numeric_data[,i] - col_mean) / col_sd
      }
    }
  }
  
  # Zurück in das original dataframe
  pdata[, existing_cols] <- numeric_data
  return(pdata)
}

# Function to create CSV files for individual cohorts
create_cohort_csvs <- function(processed_cohorts) {
  # pData
  pdata_dir <- file.path(".", "data", "cohort_data", "pData", "original")
  dir.create(pdata_dir, recursive = TRUE, showWarnings = FALSE)
  for (i in seq_along(processed_cohorts)) {
    name <- names(processed_cohorts)[i]
    write.csv(processed_cohorts[[i]]$pData, file.path(pdata_dir, paste0(name, ".csv")), row.names = TRUE)
  }
  
  # exprs
  exprs_dir <- file.path(".", "data", "cohort_data", "exprs")
  dir.create(exprs_dir, recursive = TRUE, showWarnings = FALSE)
  for (i in seq_along(processed_cohorts)) {
    name <- names(processed_cohorts)[i]
    write.csv(t(processed_cohorts[[i]]$exprs), file.path(exprs_dir, paste0(name, ".csv")), row.names = TRUE)
  }
}

# Function for merging and creating csvs
create_merged_csvs <- function(processed_cohorts) {
  # Define relevant columns
  rel_cols <- c('AGE', 'TISSUE', 'PATH_T_STAGE', 'GLEASON_SCORE', 
                'PRE_OPERATIVE_PSA', 'MONTH_TO_BCR', 'CLIN_T_STAGE', 'BCR_STATUS')
  
  # Merged pData
  pdata_list <- lapply(processed_cohorts, function(x) x$pData)
  
  # Find common columns across all pData dataframes and intersect with rel_cols
  common_columns <- Reduce(intersect, c(list(rel_cols), lapply(pdata_list, colnames)))
  
  # Merge pData, keeping only common relevant columns
  merged_pdata <- do.call(rbind, lapply(pdata_list, function(x) x[, common_columns, drop = FALSE]))
  
  merged_pdata_dir <- file.path(".", "data", "merged_data", "pData", "original")
  dir.create(merged_pdata_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(merged_pdata, file.path(merged_pdata_dir, "merged_original_pData.csv"), row.names = TRUE)
  
  # Merged exprs (intersection)
  create_merged_exprs <- function(cohorts_dict) {
    exprs_dfs <- lapply(names(cohorts_dict), function(cohort_name) {
      cohort_data <- cohorts_dict[[cohort_name]]
      if ('exprs' %in% names(cohort_data)) {
        exprs_df <- cohort_data$exprs
        colnames(exprs_df) <- paste(cohort_name, colnames(exprs_df), sep=".")
        return(exprs_df)
      }
      return(NULL)
    })
    exprs_dfs <- exprs_dfs[!sapply(exprs_dfs, is.null)]
    
    # Find common genes (row names)
    common_genes <- Reduce(intersect, lapply(exprs_dfs, rownames))
    
    # Filter dataframes for common genes
    filtered_exprs_dfs <- lapply(exprs_dfs, function(df) {
      df[common_genes, , drop = FALSE]
    })
    
    # Merge dataframes
    merged_exprs <- do.call(cbind, filtered_exprs_dfs)
    merged_exprs <- t(merged_exprs)
    return(merged_exprs)
  }
  
  merged_exprs <- create_merged_exprs(processed_cohorts)
  
  merged_exprs_dir <- file.path(".", "data", "merged_data", "exprs", "intersection")
  dir.create(merged_exprs_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(merged_exprs, file.path(merged_exprs_dir, "exprs_intersect.csv"), row.names = TRUE)
  
  return(list(merged_pdata = merged_pdata, merged_exprs = merged_exprs))
}
# Function for imputation within cohorts
impute_cohort_data <- function(cohort_data, rel_cols) {
  # Subset für relevante Spalten
  subset_data <- cohort_data[, intersect(names(cohort_data), rel_cols), drop=FALSE]
  
  # Convert strings to categorical variables
  subset_data[] <- lapply(subset_data, function(x) {
    if(is.character(x)) as.factor(x) 
    else if(is.numeric(x)) x
    else x
  })
  
  # 
  for(col in names(subset_data)) {
    # Skip columns that are fully NA in the given DF
    if(all(is.na(subset_data[[col]]))) next
    
    if(is.numeric(subset_data[[col]])) {
      # Imputed with median in numerical columns
      col_median <- median(subset_data[[col]], na.rm=TRUE)
      subset_data[[col]][is.na(subset_data[[col]])] <- col_median
    } else if(is.factor(subset_data[[col]])) {
      # Impute with mod in categorical columns
      mode_value <- names(sort(table(subset_data[[col]]), decreasing = TRUE))[1]
      subset_data[[col]][is.na(subset_data[[col]])] <- mode_value
    }
  }
  
  return(subset_data)
}
create_merged_exprs_for_imputation <- function(cohorts_dict) {
  exprs_dfs <- lapply(names(cohorts_dict), function(cohort_name) {
    cohort_data <- cohorts_dict[[cohort_name]]
    if ('exprs' %in% names(cohort_data)) {
      df <- t(cohort_data$exprs)
      rownames(df) <- paste(cohort_name, rownames(df), sep=".")
      return(df)
    }
    return(NULL)
  })
  exprs_dfs <- exprs_dfs[!sapply(exprs_dfs, is.null)]
  
  # Find all unique genes
  all_genes <- Reduce(union, lapply(exprs_dfs, colnames))
  
  # Create an empty dataframe with all genes as columns
  all_exprs_merged <- data.frame(matrix(ncol = length(all_genes), nrow = 0))
  colnames(all_exprs_merged) <- all_genes
  
  # Add data from each cohort
  for (df in exprs_dfs) {
    temp_df <- data.frame(matrix(NA, nrow = nrow(df), ncol = length(all_genes)))
    colnames(temp_df) <- all_genes
    common_genes <- intersect(colnames(df), all_genes)
    temp_df[, common_genes] <- df[, common_genes]
    rownames(temp_df) <- rownames(df)
    all_exprs_merged <- rbind(all_exprs_merged, temp_df)
  }
  
  return(all_exprs_merged)
}
# Function to merge the pData cohorts data frames
merge_pdata <- function(processed_cohorts, rel_cols) {
  # Get pData from all cohorts and replace PATH_T_STAGE for Belast cohort by
  # CLIN_T_STAGE (as discussed with Markus)
  pdata_list <- lapply(names(processed_cohorts), function(cohort_name) {
    df <- processed_cohorts[[cohort_name]]$pData
    
    if(cohort_name == "Belfast_2018_Jain" && "CLIN_T_STAGE" %in% names(df)) {
      df$PATH_T_STAGE <- df$CLIN_T_STAGE
    }
    
    df <- impute_cohort_data(df, rel_cols)
    return(df)
  })
  names(pdata_list) <- names(processed_cohorts)
  
  # Find intersection
  common_columns <- Reduce(intersect, lapply(pdata_list, colnames))
  common_rel_cols <- intersect(common_columns, rel_cols)
  
  # Merge data frames
  merged_df <- do.call(rbind, lapply(pdata_list, function(x) {
    x[, common_rel_cols, drop=FALSE]
  }))
  
  # Convert all columns to the correct data type
  merged_df <- merged_df %>%
    mutate(across(everything(), as.character)) %>%
    mutate(
      AGE = as.numeric(AGE),
      GLEASON_SCORE = as.numeric(GLEASON_SCORE),
      PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA),
      MONTH_TO_BCR = as.numeric(MONTH_TO_BCR),
      BCR_STATUS = as.factor(BCR_STATUS),
      across(where(is.character), as.factor)
    )
  
  # Imputing the missing values in the merged DF that were caused by fully missing columns
  for(col in names(merged_df)) {
    if(any(is.na(merged_df[[col]]))) {
      if(is.numeric(merged_df[[col]])) {
        col_median <- median(merged_df[[col]], na.rm=TRUE)
        merged_df[[col]][is.na(merged_df[[col]])] <- col_median
      } else if(is.factor(merged_df[[col]])) {
        mode_value <- names(sort(table(merged_df[[col]]), decreasing = TRUE))[1]
        merged_df[[col]][is.na(merged_df[[col]])] <- mode_value
      }
    }
  }
  
  # replace BCR time of 0 by 0.0001 to avoid bugs in the modelling process
  if('MONTH_TO_BCR' %in% names(merged_df)) {
    merged_df$MONTH_TO_BCR[merged_df$MONTH_TO_BCR == 0] <- 0.0001
  }
  
  return(merged_df)
}

# Main function
main_processing <- function(rds_file_names) {
  # Define relevant columns
  rel_cols <- c('AGE', 'TISSUE', 'PATH_T_STAGE', 'GLEASON_SCORE', 
                'PRE_OPERATIVE_PSA', 'MONTH_TO_BCR', 'CLIN_T_STAGE', 'BCR_STATUS')
  numeric_cols <- c('AGE', 'PRE_OPERATIVE_PSA', 'GLEASON_SCORE')
  
  # Load cohorts
  cohorts <- load_cohorts(rds_file_names)
  
  # Standardize expression data and numeric pData
  processed_cohorts <- lapply(cohorts, function(cohort) {
    cohort$exprs <- standardize(as.matrix(cohort$exprs))
    cohort$pData <- standardize_pdata_numeric(cohort$pData, numeric_cols)
    return(cohort)
  })
  
  # Create CSV files for individual cohorts
  create_cohort_csvs(processed_cohorts)
  
  # Create CSV files for merged data
  merged_data <- create_merged_csvs(processed_cohorts)
  
  # Create merged expression data for imputation
  all_exprs_merged <- create_merged_exprs_for_imputation(processed_cohorts)
  
  # Filter expression data based on missing values
  missing_values <- colSums(is.na(all_exprs_merged))
  total_rows <- nrow(all_exprs_merged)
  valid_columns <- missing_values / total_rows < 0.2
  filtered_exprs <- all_exprs_merged[, valid_columns]
  
  # Save common genes expression data
  common_genes_dir <- file.path(".", "data", "merged_data", "exprs", "common_genes")
  dir.create(common_genes_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(filtered_exprs, file.path(common_genes_dir, "common_genes.csv"), row.names = TRUE)
  
  # KNN imputation for expression data
  imputed_expr_data <- impute.knn(as.matrix(filtered_exprs[, sapply(filtered_exprs, is.numeric)]), k=35)
  imputed_expr_df <- as.data.frame(imputed_expr_data$data)
  write.csv(imputed_expr_df, file.path(common_genes_dir, "common_genes_knn_imputed.csv"), row.names = TRUE)
  
  # Save all genes expression data
  all_genes_dir <- file.path(".", "data", "merged_data", "exprs", "all_genes")
  dir.create(all_genes_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(all_exprs_merged, file.path(all_genes_dir, "all_genes.csv"), row.names = TRUE)
  
  # Impute die einzelnen Kohorten
  imputed_cohorts <- lapply(processed_cohorts, function(cohort) {
    impute_cohort_data(cohort$pData, rel_cols)
  })
  
  # Save imputed individual cohorts
  imputed_dir <- file.path(".", "data", "cohort_data", "pData", "imputed")
  dir.create(imputed_dir, recursive = TRUE, showWarnings = FALSE)
  for (i in seq_along(imputed_cohorts)) {
    name <- names(imputed_cohorts)[i]
    write.csv(imputed_cohorts[[i]], file.path(imputed_dir, paste0(name, ".csv")), row.names = TRUE)
  }
  
  # Merge und impute pData
  merged_pdata <- merge_pdata(processed_cohorts, rel_cols)
  
  # Save imputed merged pData
  merged_imputed_dir <- file.path(".", "data", "merged_data", "pData", "imputed")
  dir.create(merged_imputed_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(merged_pdata, file.path(merged_imputed_dir, "merged_imputed_pData.csv"), row.names = TRUE)
  
  return(list(
    processed_cohorts = processed_cohorts,
    imputed_cohorts = imputed_cohorts,
    merged_pdata = merged_pdata,
    filtered_exprs = filtered_exprs,
    imputed_expr_df = imputed_expr_df
  ))
}
  
process_test_cohorts <- function(test_rds_file) {
  # Define numeric columns for standardization
  numeric_cols <- c('AGE', 'PRE_OPERATIVE_PSA', 'GLEASON_SCORE')
  
  # Load test cohorts
  test_cohorts <- readRDS(file.path(".", "data", test_rds_file))
  
  # Process each cohort
  processed_test_cohorts <- lapply(test_cohorts, function(eset) {
    # Extract and process pData
    pdata <- as.data.frame(pData(eset))
    # Standardize numeric pData
    pdata <- standardize_pdata_numeric(pdata, numeric_cols)
    
    # Standardize expression data
    exprs_standardized <- standardize(as.matrix(exprs(eset)))
    
    list(
      pData = pdata,
      exprs = as.data.frame(exprs_standardized),
      cohort = eset$cohort[1]
    )
  })
  
  # Save individual test cohorts expression data
  exprs_dir <- file.path(".", "data", "cohort_data", "exprs")
  dir.create(exprs_dir, recursive = TRUE, showWarnings = FALSE)
  
  for(i in seq_along(processed_test_cohorts)) {
    cohort_name <- paste0("test_cohort_", i)
    cohort_exprs <- processed_test_cohorts[[i]]$exprs
    
    # Filter for training genes
    training_intersect <- read.csv(file.path(".", "data", "merged_data", "exprs", "intersection", "exprs_intersect.csv"), row.names=1)
    training_genes <- colnames(training_intersect)
    cohort_final_genes <- intersect(rownames(cohort_exprs), training_genes)
    cohort_exprs_final <- cohort_exprs[cohort_final_genes,]
    
    # Save individual test cohort expression data
    write.csv(t(cohort_exprs_final), 
              file.path(exprs_dir, paste0(cohort_name, ".csv")),
              row.names = TRUE)
  }
  
  # Save individual test cohorts pData
  pdata_dir <- file.path(".", "data", "cohort_data", "pData", "original")
  dir.create(pdata_dir, recursive = TRUE, showWarnings = FALSE)
  for(i in seq_along(processed_test_cohorts)) {
    cohort_name <- paste0("test_cohort_", i)
    write.csv(processed_test_cohorts[[i]]$pData,
              file.path(pdata_dir, paste0(cohort_name, ".csv")),
              row.names = TRUE)
  }
  
  # Merge test pData
  test_pdata <- do.call(rbind, lapply(processed_test_cohorts, function(x) x$pData))
  
  # Impute NA values in PRE_OPERATIVE_PSA with median
  psa_median <- median(test_pdata$PRE_OPERATIVE_PSA, na.rm=TRUE)
  test_pdata$PRE_OPERATIVE_PSA[is.na(test_pdata$PRE_OPERATIVE_PSA)] <- psa_median
  
  test_pdata[is.na(test_pdata$MONTH_TO_BCR),"MONTH_TO_BCR"] <- test_pdata[is.na(test_pdata$MONTH_TO_BCR),]$MONTH_TO_DOD
  test_pdata[is.na(test_pdata$BCR_STATUS), "BCR_STATUS"] <- test_pdata[is.na(test_pdata$BCR_STATUS), ]$DOD_STATUS
  
  # Create merged expression matrix for test cohorts
  test_exprs_list <- lapply(processed_test_cohorts, function(x) x$exprs)
  test_common_genes <- Reduce(intersect, lapply(test_exprs_list, rownames))
  test_exprs_filtered <- lapply(test_exprs_list, function(df) df[test_common_genes,])
  test_exprs <- do.call(cbind, test_exprs_filtered)
  
  # Filter test expression data for genes present in training data
  training_intersect <- read.csv(file.path(".", "data", "merged_data", "exprs", "intersection", "exprs_intersect.csv"), row.names=1)
  training_common_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "common_genes", "common_genes.csv"), row.names=1)
  training_all_genes <- read.csv(file.path(".", "data", "merged_data", "exprs", "all_genes", "all_genes.csv"), row.names=1)
  
  training_genes <- colnames(training_intersect)
  final_genes <- intersect(rownames(test_exprs), training_genes)
  test_exprs_final <- test_exprs[final_genes,]
  
  # Create copies for test data
  test_intersect_imputed <- training_intersect
  test_common_genes_imputed <- training_common_genes
  test_all_genes <- training_all_genes
  
  # For each test cohort
  for(i in seq_along(processed_test_cohorts)) {
    cohort_exprs <- processed_test_cohorts[[i]]$exprs
    cohort_name <- paste0("test_cohort_", i)
    
    # For each patient in the cohort
    for(j in 1:ncol(cohort_exprs)) {
      # For intersection genes
      common_genes_intersect <- intersect(rownames(cohort_exprs), colnames(test_intersect_imputed))
      new_row <- data.frame(matrix(NA, nrow=1, ncol=ncol(test_intersect_imputed)))
      colnames(new_row) <- colnames(test_intersect_imputed)
      new_row[1, common_genes_intersect] <- as.numeric(cohort_exprs[common_genes_intersect, j])
      rownames(new_row) <- paste0(cohort_name, "_patient_", j)
      test_intersect_imputed <- rbind(test_intersect_imputed, new_row)
      
      # For common genes
      common_genes_common <- intersect(rownames(cohort_exprs), colnames(test_common_genes_imputed))
      new_row <- data.frame(matrix(NA, nrow=1, ncol=ncol(test_common_genes_imputed)))
      colnames(new_row) <- colnames(test_common_genes_imputed)
      new_row[1, common_genes_common] <- as.numeric(cohort_exprs[common_genes_common, j])
      rownames(new_row) <- paste0(cohort_name, "_patient_", j)
      test_common_genes_imputed <- rbind(test_common_genes_imputed, new_row)
      
      # For all genes - without imputation
      common_genes_all <- intersect(rownames(cohort_exprs), colnames(test_all_genes))
      new_row <- data.frame(matrix(NA, nrow=1, ncol=ncol(test_all_genes)))
      colnames(new_row) <- colnames(test_all_genes)
      new_row[1, common_genes_all] <- as.numeric(cohort_exprs[common_genes_all, j])
      rownames(new_row) <- paste0(cohort_name, "_patient_", j)
      test_all_genes <- rbind(test_all_genes, new_row)
    }
  }
  
  # Perform KNN imputation on intersection and common genes datasets
  # For intersection genes
  test_intersect_matrix <- as.matrix(test_intersect_imputed)
  imputed_intersect <- impute.knn(test_intersect_matrix, k=35)
  test_intersect_imputed <- as.data.frame(imputed_intersect$data)
  
  # For common genes  
  test_common_matrix <- as.matrix(test_common_genes_imputed)
  imputed_common <- impute.knn(test_common_matrix, k=35)
  test_common_genes_imputed <- as.data.frame(imputed_common$data)
  
  # Remove training observations (keep only test cohort rows)
  test_cohort_rows <- grep("^test_cohort_", rownames(test_intersect_imputed))
  test_intersect_imputed <- test_intersect_imputed[test_cohort_rows,]
  test_common_genes_imputed <- test_common_genes_imputed[test_cohort_rows,]
  test_all_genes <- test_all_genes[test_cohort_rows,]
  
  # Remove training observations (keep only test cohort rows)
  test_cohort_rows <- grep("^test_cohort_", rownames(test_intersect_imputed))
  test_intersect_imputed <- test_intersect_imputed[test_cohort_rows,]
  test_common_genes_imputed <- test_common_genes_imputed[test_cohort_rows,]
  test_all_genes <- test_all_genes[test_cohort_rows,]
  
  # Ensure rownames of pData match exprs data
  original_rownames <- rownames(test_pdata)
  new_rownames <- rownames(test_intersect_imputed)
  rownames(test_pdata) <- new_rownames[1:nrow(test_pdata)]
  
  #test_pdata$TISSUE <-  "Fresh_frozen"
  # Save merged test data
  write.csv(test_pdata, 
            file.path(".", "data", "merged_data", "pData", "imputed", "test_pData_imputed.csv"))
  
  # Save intersection test genes
  write.csv(t(test_exprs_final), 
            file.path(".", "data", "merged_data", "exprs", "intersection", "intersect_test_genes.csv"),
            row.names = TRUE)
  
  # Save the imputed test data
  write.csv(test_intersect_imputed, 
            file.path(".", "data", "merged_data", "exprs", "intersection", "intersect_test_genes_imputed.csv"),
            row.names = TRUE)
  
  write.csv(test_common_genes_imputed,
            file.path(".", "data", "merged_data", "exprs", "common_genes", "common_genes_test_imputed.csv"), 
            row.names = TRUE)
  
  # Save all genes test data (without imputation)
  write.csv(test_all_genes,
            file.path(".", "data", "merged_data", "exprs", "all_genes", "all_genes_test.csv"), 
            row.names = TRUE)
  
  return(list(
    test_pdata = test_pdata,
    test_exprs = test_exprs_final,
    test_exprs_common_imputed = test_common_genes_imputed,
    test_exprs_intersect_imputed = test_intersect_imputed,
    test_exprs_all = test_all_genes
  ))
}

process_risk_scores <- function(rds_file_path) {
  # Read the RDS file
  risk_scores <- readRDS(file.path(".", "data", rds_file_path))
  
  # Create directory for scores
  scores_dir <- file.path(".", "data", "scores")
  cohort_scores_dir <- file.path(scores_dir, "cohort_specific")
  dir.create(cohort_scores_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Save individual cohort scores
  for(cohort_name in names(risk_scores)) {
    cohort_scores <- risk_scores[[cohort_name]]
    
    # Create data frame for this cohort
    cohort_df <- data.frame(
      sample_id = names(cohort_scores),
      risk_score = unname(cohort_scores),
      stringsAsFactors = FALSE
    )
    
    # Add cohort prefix to sample IDs if not already present
    cohort_df$sample_id <- sapply(cohort_df$sample_id, function(name) {
      if (!grepl(paste0("^", cohort_name), name)) {
        return(paste(cohort_name, name, sep="_"))
      }
      return(name)
    })
    
    # Save individual cohort scores
    write.csv(cohort_df,
              file.path(cohort_scores_dir, paste0(cohort_name, "_scores.csv")),
              row.names = FALSE)
  }
  
  # Create all scores vector (merge all cohorts)
  all_scores <- unlist(risk_scores)
  
  # Create train data scores (all except TCGA_PRAD and UKD2)
  train_cohorts <- setdiff(names(risk_scores), c("TCGA_PRAD", "UKD2"))
  train_data_scores <- unlist(risk_scores[train_cohorts])
  
  # Create test data scores (only TCGA_PRAD and UKD2)
  test_cohorts <- c("TCGA_PRAD", "UKD2")
  test_data_scores <- unlist(risk_scores[test_cohorts])
  
  # Create data frames with proper sample IDs including cohort names
  all_scores_df <- data.frame(
    sample_id = names(all_scores),
    risk_score = all_scores,
    stringsAsFactors = FALSE
  )
  
  # Add cohort prefix to sample IDs where missing
  add_cohort_prefix <- function(scores_vector) {
    names_with_prefix <- sapply(names(scores_vector), function(name) {
      # Get cohort name from the list structure
      cohort <- names(risk_scores)[sapply(risk_scores, function(x) name %in% names(x))]
      if (!grepl(paste0("^", cohort), name)) {
        return(paste(cohort, name, sep="_"))
      }
      return(name)
    })
    return(names_with_prefix)
  }
  
  all_scores_df$sample_id <- add_cohort_prefix(all_scores)
  
  train_scores_df <- data.frame(
    sample_id = add_cohort_prefix(train_data_scores),
    risk_score = train_data_scores,
    stringsAsFactors = FALSE
  )
  
  test_scores_df <- data.frame(
    sample_id = add_cohort_prefix(test_data_scores),
    risk_score = test_data_scores,
    stringsAsFactors = FALSE
  )
  
  # Save merged CSV files
  write.csv(all_scores_df, 
            file.path(scores_dir, "all_scores.csv"), 
            row.names = FALSE)
  write.csv(train_scores_df, 
            file.path(scores_dir, "train_scores.csv"), 
            row.names = FALSE)
  write.csv(test_scores_df, 
            file.path(scores_dir, "test_scores.csv"), 
            row.names = FALSE)
  
  return(list(
    all_scores = all_scores_df,
    train_scores = train_scores_df, 
    test_scores = test_scores_df,
    cohort_scores = risk_scores  # Return individual cohort scores as well
  ))
}

# Main function that runs all processes
main <- function(rds_file_names, test_rds_file) {
  # Run original preprocessing
  result <- main_processing(rds_file_names)
  
  # Process test cohorts
  test_result <- process_test_cohorts(test_rds_file)
  
  # Process risk scores
  risk_scores_result <- process_risk_scores("Revised_ProstaTrend.Rds")
  
  # Return combined results
  return(list(
    training = result,
    test = test_result,
    risk_scores = risk_scores_result
  ))
}

# Call the main function
rds_file_names <- c("PCa_cohorts.RDS")
test_rds_file <- "PCa_cohorts_2.RDS"
result <- main(rds_file_names, test_rds_file)






