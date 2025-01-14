library(dplyr)
library(prioritylasso)
library(survival)
library(readr)
library(rsample)
library(purrr)
library(SurvMetrics)
library(glmnet)


construct_indcs_bp <- function(df){
    range_list <- lapply(1:nrow(df), function(i) {
        (df$i_start[i]+1):(df$i_end[i]+1)
        })
    range_list
}

remove_small_blocks <- function(indcs, df_data){
    indcs = indcs[indcs$nmb_genes <= 2,]
    for(i in 1:nrow(indcs)){
        i_start = indcs[i, 'i_start']+1
        i_end = indcs[i, 'i_end']+1
        df_data = df_data[, -c(i_start:i_end)]
        print(ncol(df_data))
    }       
    return(df_data) 
}

check_overlap <- function(blocks){
  blocks_ul <- unlist(blocks)
  duplicate_elements <- duplicated(blocks_ul)  # Find duplicates
  sum(duplicate_elements)
} 


# Nested resampling function
do_resampling <- function(outer_split, blocks) {
  # Extract training and testing data for the outer split
  outer_train <- analysis(outer_split)
  outer_test <- assessment(outer_split)

  #print("Test cohort: ")
  #print(outer_test[0, 'cohort'])
  
  # Convert outer training data to matrix format
  y_train_outer <- Surv(outer_train$MONTH_TO_BCR, outer_train$BCR_STATUS)
  inner_indcs <- as.numeric(as.factor(outer_train$cohort))
  x_train_outer <- as.matrix(outer_train %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))

  print(max(inner_indcs))

  print(check_overlap(blocks))
  
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
    #foldid = inner_indcs,
    lambda.type = "lambda.1se",
    type.measure = "deviance",
    mcontrol = missing.control(handle.missingdata = "ignore")
    #mcontrol = missing.control(handle.missingdata = "impute.offset", 
    #nfolds.imputation = 3,
    #lambda.imputation = "lambda.min",
    #impute.offset.cases = "available.cases", 
    #select.available.cases = "max")
  )
  
  # Predict on the outer test set
  if (any(is.na(x_test_outer))) {
    outer_predictions <- predict(prio_lasso, newdata = x_test_outer, type = "response", handle.missingtestdata = c("set.zero"))
  } 
  else {
    outer_predictions <- predict(prio_lasso, newdata = x_test_outer, type = "response", handle.missingtestdata = c("none"))

  }
  # Compute metrics for the outer test set
  #outer_cindex <- SurvMetrics::Cindex(y_test_outer, outer_predictions)
  #print(outer_cindex)
  outer_cindex_se <- apply(outer_predictions, 2, glmnet::Cindex, y=y_test_outer)
  print(outer_cindex_se)
  # Return the results for this outer fold
  outer_cindex_se
}

df_blockwise_data = read_csv('models/prio_lasso/df_block_data.csv', lazy = TRUE)
df_blockwise_indcs = read_csv('models/prio_lasso/df_block_indices.csv')
df_pData = read_csv('data/merged_data/pData/imputed/merged_imputed_pData.csv')
df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001

data <- df_blockwise_data %>% select(-c(index))
data['MONTH_TO_BCR'] <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
data['BCR_STATUS'] <- df_pData$BCR_STATUS
blocks <- construct_indcs_bp(df_blockwise_indcs)

# Outer leave one out resampling split
outer_splits <- group_vfold_cv(data, group = cohort)



# Perform nested resampling for all outer folds
# nested_results <- outer_splits %>% mutate(metrics = map(splits, ~ nested_resampling(.x, blocks)))

outer_perf = setNames(data.frame(matrix(ncol = 2, nrow = 9)), c("testing_cohort", "ci"))
for (i in seq_along(outer_splits$splits)) {
  # Get the split object
  outer_split <- outer_splits$splits[[i]]
  outer_train <- analysis(outer_split)
  outer_test <- assessment(outer_split)
  test_cohort <- as.character(outer_test$cohort[1])
  print(test_cohort)

  ci <- do_resampling(outer_split, blocks)

  # modify this one 
  # y_test_outer <- Surv(outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
  # X_test_outer <- as.matrix(outer_test %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))

  # test_preds_se <- predict(best_mod, X_test_outer,  s = 'lambda.1se')
  # test_preds_min <- predict(best_mod, X_test_outer,  s = 'lambda.min')
  # outer_cindex_se <- apply(test_preds_se, 2, glmnet::Cindex, y=y_test_outer)
  # outer_cindex_min <- apply(test_preds_min, 2, glmnet::Cindex, y=y_test_outer)
  # print(outer_cindex_se)
  # print(outer_cindex_min)
  outer_perf[i, ] <- c(test_cohort, ci)
}

print(outer_perf)

write.csv(outer_perf, "prioLasso.csv")
