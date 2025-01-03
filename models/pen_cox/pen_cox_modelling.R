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
library(glmnet)

# ---------------------------------------------------- resampling functions
do_resampling <- function(data) {
  # Convert outer training data to matrix format
  y_train <- Surv(data$MONTH_TO_BCR, data$BCR_STATUS)
  inner_indcs <- as.numeric(as.factor(data$cohort))
  x_train <- as.matrix(data %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))

  print(max(inner_indcs))

  # inherently does CV on provided splits; provide folds to ensure grouped CV    
    cvfit <-  cv.glmnet(x_train, y_train, 
        family = "cox", 
        nfolds = 8, 
        foldid = inner_indcs, 
        alpha = 1
        )

    # coxph <- survfit(
    #     cvfit,
    #     s = "lambda.1se", 
    #     x = x_train, 
    #     y = y_train
    # )
    return(cvfit)
}

# ------------------------------------------------------------- Load data
#df_exprs = read_csv('models/playground_tbd/df_block_data.csv', lazy = TRUE)
#df_pData = read_csv('data/merged_data/pData/original/merged_original_pData.csv')
# df_pData = read.csv2('data/merged_data/pData/imputed/merged_imputed_pData.csv', sep = ',')
# df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA)) 
# df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
# df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
# df_pData$cohort <- sub("\\..*", "", df_pData$X)

# df_pData <- as.data.frame(unclass(df_pData),stringsAsFactors=TRUE)
# exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/intersection/exprs_intersect.csv', lazy = TRUE))

# data <- cbind(exprs_data, df_pData)
# data[, 1] <- NULL



prepare_data <- function(use_exprs, use_pData, vars_pData = NA){
    if(use_exprs){
        exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/intersection/exprs_intersect.csv', lazy = TRUE))
        exprs_data[, 1] <- NULL
    }
        df_pData = read.csv2('data/merged_data/pData/imputed/merged_imputed_pData.csv', sep = ',')
        df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA)) 
        df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
        df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
        cohort <- sub("\\..*", "", df_pData$X)
    if(length(vars_pData) != 0){
        relevant_vars <- c('MONTH_TO_BCR', 'BCR_STATUS', vars_pData)
        df_pData <- df_pData[, relevant_vars]

        cat_pData <- df_pData %>%
            as_data_frame() %>%
            mutate_if(is.character, factor) %>%
            select_if(~ is.factor(.) == TRUE)

        num_pData <- df_pData %>%
            as_data_frame() %>%
            mutate_if(is.character, factor) %>%
            select_if(~ is.numeric(.) == TRUE)

        dmy <- dummyVars(" ~ .", data = cat_pData)
        ohenc_pData <- data.frame(predict(dmy, newdata = cat_pData))

        df_pData <- cbind(num_pData, ohenc_pData, cohort)
    }
    if(use_pData && use_exprs){
        return(cbind(df_pData, exprs_data))

    } else if (use_pData && !use_exprs) {
       return(df_pData)
    } else {
        return(cbind(df_pData$MONTH_TO_BCR, df_pData$BCR_STATUS, cohort, exprs_data))
    }
}

data = prepare_data(TRUE, TRUE, c("AGE", "TISSUE", "GLEASON_SCORE"))  
#print(str(data))

# ------------------------------------------------------------- Create Splits and grids for tuning
outer_splits <- group_vfold_cv(data, group = cohort)

# ------------------------------------------------------------- Do nested resampling
outer_perf = setNames(data.frame(matrix(ncol = 2, nrow = 9)), c("testing_cohort", "ci"))
for (i in seq_along(outer_splits$splits)) {
  # Get the split object
  outer_split <- outer_splits$splits[[i]]
  outer_train <- analysis(outer_split)
  outer_test <- assessment(outer_split)
  test_cohort <- as.character(outer_test$cohort[1])
  print(test_cohort)

  best_mod <- do_resampling(outer_train)
  print(best_mod)

# modify this one 
y_test_outer <- Surv(outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
  X_test_outer <- as.matrix(outer_test %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))

  # TODO: insert hps from above

  test_preds <- predict(best_mod, X_test_outer)
  outer_cindex <- SurvMetrics::Cindex(y_test_outer, test_preds)
  print(outer_cindex)
  outer_perf[i, ] <- c(test_cohort, outer_cindex)
}

print(outer_perf)



# ------------------------------------------------------------- Tuning + fitting of final model
final_model <- do_resampling(data)