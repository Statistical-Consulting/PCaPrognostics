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
do_resampling <- function(outer_train, blocks) {
  #print("Test cohort: ")
  #print(outer_test[0, 'cohort'])
  
  # Convert outer training data to matrix format
  y_train_outer <- Surv(outer_train$MONTH_TO_BCR, outer_train$BCR_STATUS)
  inner_indcs <- as.numeric(as.factor(outer_train$cohort))
  x_train_outer <- as.matrix(outer_train %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))

  print(max(inner_indcs))

  print(check_overlap(blocks))
  
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
    mcontrol = missing.control(handle.missingdata = "ignore"), 
    #cvoffset = TRUE, 
    #cvoffsetnfolds = 4,
    #mcontrol = missing.control(handle.missingdata = "impute.offset", 
    #nfolds.imputation = 3,
    #lambda.imputation = "lambda.min",
    #impute.offset.cases = "available.cases", 
    #select.available.cases = "max")
  ) 
  return(prio_lasso)
}

df_blockwise_data = read_csv('models/prio_lasso/df_block_data_100_300.csv', lazy = TRUE)
df_blockwise_indcs = read_csv('models/prio_lasso/df_block_indices_100_300.csv')
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

  best_mod <- do_resampling(outer_train = outer_train, blocks)

    # Convert outer testing data to matrix format
    y_test_outer <- Surv(outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
    x_test_outer <- as.matrix(outer_test %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))

    # Predict on the outer test set
    if (any(is.na(x_test_outer))) {
      outer_predictions <- predict(best_mod, newdata = x_test_outer, type = "response", handle.missingtestdata = c("set.zero"), include.allintercepts = TRUE)
    } 
    else {
      outer_predictions <- predict(best_mod, newdata = x_test_outer, type = "response", handle.missingtestdata = c("none"), include.allintercepts = TRUE)
    }

    # Compute metrics for the outer test set
    ci <- apply(outer_predictions, 2, glmnet::Cindex, y=y_test_outer)

    outer_perf[i, ] <- c(test_cohort, ci)
}

print(outer_perf)
write.csv(outer_perf, "prioLasso_100_300_intercepts.csv")


final_model <- do_resampling(data, blocks = blocks)
save(final_model,file="prioLasso_100_300_intercepts.Rdata")
