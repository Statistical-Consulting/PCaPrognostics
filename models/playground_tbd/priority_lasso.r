library(dplyr)
library(prioritylasso)
library(survival)
library(readr)
library(rsample)
library(purrr)
library(SurvMetrics)

construct_indcs_bp <- function(df){
    range_list <- lapply(1:nrow(df), function(i) {
        (df$i_start[i]+1):(df$i_end[i]+1)
        })
    range_list
}

remove_small_blocks <- function(indcs, df_data){
    indcs = indcs[indcs$nmb_genes <= 300,]
    for(i in 1:nrow(indcs)){
        i_start = indcs[i, 'i_start']+1
        i_end = indcs[i, 'i_end']+1
        df_data = df_data[, -c(i_start:i_end)]
        print(ncol(df_data))
    }       
    return(df_data) 
}


# Nested resampling function
nested_resampling <- function(outer_split, blocks) {
  # Extract training and testing data for the outer split
  outer_train <- analysis(outer_split)
  outer_test <- assessment(outer_split)

  print("Test cohort: ")
  print(outer_test['cohort'])
  
  # Convert outer training data to matrix format
  y_train_outer <- Surv(outer_train$MONTH_TO_BCR, outer_train$BCR_STATUS)
  inner_indcs <- as.numeric(as.factor(outer_train$cohort))
  x_train_outer <- as.matrix(outer_train %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))

  print(max(inner_indcs))
  
  # Convert outer testing data to matrix format
  y_test_outer <- Surv(outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
  x_test_outer <- as.matrix(outer_test %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))

  # inherently does CV on provided splits    
  prio_lasso <- prioritylasso(
    x_train_outer,
    y_train_outer,
    family = "cox",
    blocks = blocks, 
    block1.penalization = TRUE,
    nfolds = 8, 
    foldid = inner_indcs,
    lambda.type = "lambda.1se",
    type.measure = "deviance",
    mcontrol = missing.control(handle.missingdata = "ignore")
  )
  
  # Predict on the outer test set
  if (any(is.na(x_test_outer))) {
    outer_predictions <- predict(prio_lasso, newdata = x_test_outer, type = "response", handle.missingtestdata = c("set.zero"))
  } 
  else {
    outer_predictions <- predict(prio_lasso, newdata = x_test_outer, type = "response", handle.missingtestdata = c("none"))

  }
  # Compute metrics for the outer test set
  outer_cindex <- Cindex(y_test_outer, outer_predictions)
  print(outer_cindex)
  # Return the results for this outer fold
  outer_cindex
}

df_blockwise_data = read_csv('models/playground_tbd/df_block_data.csv', lazy = TRUE)
df_blockwise_indcs = read_csv('models/playground_tbd/df_block_indices.csv')
df_pData = read_csv('data/merged_data/pData/original/merged_original_pData.csv')
df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001

data <- df_blockwise_data %>% select(-c(index))
data['MONTH_TO_BCR'] <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
data['BCR_STATUS'] <- df_pData$BCR_STATUS
blocks <- construct_indcs_bp(df_blockwise_indcs)

# Outer leave one out resampling split
outer_splits <- group_vfold_cv(data, group = cohort)

# Perform nested resampling for all outer folds
nested_results <- outer_splits %>% mutate(metrics = map(splits, ~ nested_resampling(.x, blocks)))

nested_results
