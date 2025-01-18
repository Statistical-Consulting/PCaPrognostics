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
        alpha = 1, 
        type.measure = "C"
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



prepare_data <- function(use_exprs, use_pData, vars_pData = NA, use_aenc = FALSE){
    if(use_exprs){
        #exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/intersection/exprs_intersect.csv', lazy = TRUE))
        #exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/common_genes/common_genes_knn_imputed', lazy = TRUE))
        exprs_data <- as.data.frame(read_csv('data/scores/train_scores.csv', lazy = TRUE))
        exprs_data[, 1] <- NULL
    }
    df_pData = read.csv2('data/merged_data/pData/imputed/merged_imputed_pData.csv', sep = ',')
    df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
    df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
    df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
    cohort <- sub("\\..*", "", df_pData$X)
    X = df_pData$X
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

        df_pData <- cbind(num_pData, ohenc_pData, cohort, X)
    }
    if(use_pData && use_exprs && !use_aenc){
        return(cbind(df_pData, exprs_data))

    } else if (use_pData && !use_exprs && !use_aenc) {
       return(df_pData)
    } else if (use_pData && use_aenc){
        return(df_pData)
    } else if (!use_pData && use_aenc){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort)
        return(df)
    }
}

use_aenc = TRUE
data_cmplt = prepare_data(FALSE, FALSE, c("AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA'), use_aenc = use_aenc)  
print(str(data_cmplt))
# ------------------------------------------------------------- Create Splits and grids for tuning
outer_splits <- group_vfold_cv(data_cmplt, group = cohort)

# ------------------------------------------------------------- Do nested resampling
outer_perf = setNames(data.frame(matrix(ncol = 3, nrow = 9)), c("testing_cohort", "ci_se", "ci_min"))
for (i in seq_along(outer_splits$splits)) {
  # Get the split object
  outer_split <- outer_splits$splits[[i]]
  outer_train <- analysis(outer_split)
  outer_test <- assessment(outer_split)
  test_cohort <- as.character(outer_test$cohort[1])
  print(test_cohort)

if (use_aenc){
    data_path <- paste0('pretrnd_models_ae\\csv\\' , test_cohort, '.csv') 
    anec_data = read.csv(data_path) %>% mutate_if(is.character, factor)
    #print(str(anec_data))

    outer_train <- as.data.frame(outer_train)
    outer_test <- as.data.frame(outer_test)
    #print(str(outer_train))

    outer_train = left_join(outer_train, anec_data, by = "X")
    outer_test = left_join(outer_test, anec_data, by = "X")

    #print(nrow(outer_train))
    #print(ncol(outer_train))

    #print(nrow(outer_test))
    #print(ncol(outer_test))

    #outer_train <- as.data.frame(outer_train %>% select(-c(X)))
    #outer_test <- as.data.frame(outer_test %>% select(-c(X)))

} 
    outer_train <- as.data.frame(outer_train %>% select(-c(X)))
    outer_test <- as.data.frame(outer_test %>% select(-c(X)))

    best_mod <- do_resampling(outer_train)


# modify this one 
y_test_outer <- Surv(outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
X_test_outer <- as.matrix(outer_test %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))

test_preds_se <- predict(best_mod, X_test_outer,  s = 'lambda.1se')
test_preds_min <- predict(best_mod, X_test_outer,  s = 'lambda.min')
outer_cindex_se <- apply(test_preds_se, 2, glmnet::Cindex, y=y_test_outer)
outer_cindex_min <- apply(test_preds_min, 2, glmnet::Cindex, y=y_test_outer)
print(outer_cindex_se)
print(outer_cindex_min)
outer_perf[i, ] <- c(test_cohort, outer_cindex_se, outer_cindex_min)
}

print(outer_perf)

write.csv(outer_perf, "pen_autoenc_paper.csv")

# --------------------------------------------------------------- Tuning + fitting of final model with Aenc
data_path <- paste0('pretrnd_models_ae\\csv\\pretrnd_cmplt.csv') 
aenc_data <- read.csv(data_path) %>% mutate_if(is.character, factor)
aenc_data <- left_join(data_cmplt, aenc_data, by = "X")
str(aenc_data)
anec_data <- as.data.frame(aenc_data %>% select(-c(X)))
final_model <- do_resampling(anec_data)

# # ------------------------------------------------------------- Tuning + fitting of final model
# data_cmplt <- as.data.frame(data_cmplt %>% select(-c(X)))
# final_model <- do_resampling(data_cmplt)
save(final_model,file="pen_autoenc_paper.Rdata")