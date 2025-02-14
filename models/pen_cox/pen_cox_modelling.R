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

#' @description Does a complete resampling procedure on the provided data.
#'              If more than one alpha is provided, it loops over them and returns
#'              the cv.glmnet object that had the best performance.
#' @param data (dataframe) data set (both train and test data)
#' @param alphas (numeric vector) One or more alpha values to try in cv.glmnet.
#' @return Fitted cv.glment object
do_resampling <- function(data, alphas, use_pData) {
  # Prepare survival outcome and predictors
  inner_indcs <- as.numeric(as.factor(data$cohort))
  if(use_pData){
  y_train <- Surv(data$tstart, data$MONTH_TO_BCR, data$BCR_STATUS)
  x_train <- as.matrix(data %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS, tstart)))
  }
  else {
    y_train <- Surv(data$MONTH_TO_BCR, data$BCR_STATUS)
    x_train <- as.matrix(data %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))
}
  
  if (length(alphas) == 1) {
    cvfit <- cv.glmnet(x_train, y_train, 
                       family = "cox", 
                       nfolds = 8, 
                       foldid = inner_indcs, 
                       alpha = alphas,
                       type.measure = "C")
    return(list(alpha = alphas, cvfit = cvfit))
  } else {
    results <- list()
    perf <- numeric(length(alphas))
    for (i in seq_along(alphas)) {
      alpha_val <- alphas[i]
      cvfit <- cv.glmnet(x_train, y_train, 
                         family = "cox", 
                         nfolds = 8, 
                         foldid = inner_indcs, 
                         alpha = alpha_val, 
                         type.measure = "C")

      best_index <- which.max(cvfit$cvm)
      perf[i] <- cvfit$cvm[best_index]
      results[[i]] <- list(alpha = alpha_val, cvfit = cvfit, cvm_min = cvfit$cvm[best_index])
    }
    best_idx <- which.max(perf)
    best_result <- results[[best_idx]]
    return(best_result$cvfit)
  }
}


#' @description Prepares data for modelling
#' @param use_exprs (bool) Wether gene data is used as covariates in general
#' @param use_inter (bool) TRUE: Uses intersection data, FALSE: Uses common genes data
#' @param use_pData (bool) Wether clinical data is used as covariates
#' @param vars_pData (string vector) Column names of clinical variables to be used
#' @param use_aenc (bool) Wether to use the latent representation obtained from the autoencoder
#' @return dataframe of loaded data
prepare_data <- function(use_exprs, use_inter, use_pData, use_aenc = FALSE, vars_pData = NA){
    if(use_exprs){
      if(use_inter)
        exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/intersection/exprs_intersect.csv', lazy = TRUE))
      else 
        exprs_data <- as.data.frame(read_csv('data/merged_data/exprs/common_genes/common_genes_knn_imputed.csv', lazy = TRUE))
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
        # df_pData$gleason_time <- df_pData$GLEASON_SCORE * log(df_pData$MONTH_TO_BCR)
    }
    if(use_pData && use_exprs && !use_aenc){
        df <- cbind(df_pData, exprs_data)
        df <- survSplit(Surv(MONTH_TO_BCR, BCR_STATUS)~., data = df, id = "ID",
        cut = c(6, 64)) 
        df <- df%>% select(-c(ID))
        df$gleason_tstart <- scale(df$GLEASON_SCORE * df$tstart)
    } else if ((use_pData && !use_exprs && !use_aenc) | use_pData && use_aenc) {
        df <- df_pData
        df <- survSplit(Surv(MONTH_TO_BCR, BCR_STATUS)~., data = df, id = "ID",
        cut = c(6, 64))  
        df <- df%>% select(-c(ID))
        df$gleason_tstart <- scale(df$GLEASON_SCORE * df$tstart)
    } else if ((!use_pData && use_aenc)){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort)
    } else if((!use_pData && use_exprs)){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort, exprs_data)
    }
    return(df)
}

# ------------------------------------------------------------- Modelling
# set bools for preparing the data
use_aenc = TRUE # if latent space from AE is to be used
use_inter = FALSE # if gene data in general is to be used
use_exprs = FALSE # if intersection data is to be used --> if FALSE & use_inter then imputed/common genes are used
use_pData = FALSE # if clinical data is used
vars_pData = c("AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')

data_cmplt = prepare_data(use_exprs, use_inter, use_pData, use_aenc, vars_pData)
print(nrow(data_cmplt))
# Create Splits and grids for tuning
outer_splits <- group_vfold_cv(data_cmplt, group = cohort)
candidate_alphas <- seq(0.6, 1, by = 0.2)

# Do nested resampling
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

    outer_train <- as.data.frame(outer_train)
    outer_test <- as.data.frame(outer_test)

    outer_train = left_join(outer_train, anec_data, by = "X")
    outer_test = left_join(outer_test, anec_data, by = "X")
    print(nrow(outer_test))
    print(nrow(outer_train))
    } 
    outer_train <- as.data.frame(outer_train %>% select(-c(X)))
    outer_test <- as.data.frame(outer_test %>% select(-c(X)))

    best_mod <- do_resampling(outer_train, candidate_alphas, use_pData)

if(use_pData){
  y_test_outer <- Surv(outer_test$tstart, outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
  X_test_outer <- as.matrix(outer_test %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS, tstart)))
}
else {
  y_test_outer <- Surv(outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
  X_test_outer <- as.matrix(outer_test %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))
}

test_preds_se <- predict(best_mod, X_test_outer,  s = 'lambda.1se')
test_preds_min <- predict(best_mod, X_test_outer,  s = 'lambda.min')
outer_cindex_se <- apply(test_preds_se, 2, glmnet::Cindex, y=y_test_outer)
outer_cindex_min <- apply(test_preds_min, 2, glmnet::Cindex, y=y_test_outer)
outer_perf[i, ] <- c(test_cohort, outer_cindex_se, outer_cindex_min)
}

write.csv(outer_perf, "final_aenc.csv")

# # --------------------------------------------------------------- Tuning + fitting of final model
if (use_aenc){
  data_path <- paste0('pretrnd_models_ae\\csv\\pretrnd_cmplt.csv') 
  aenc_data <- read.csv(data_path) %>% mutate_if(is.character, factor)
  aenc_data <- left_join(data_cmplt, aenc_data, by = "X")
  anec_data <- as.data.frame(aenc_data %>% select(-c(X)))
  final_model <- do_resampling(anec_data, alphas = candidate_alphas, use_pData)
} else {
  data_cmplt <- as.data.frame(data_cmplt %>% select(-c(X)))
  final_model <- do_resampling(data_cmplt, candidate_alphas, use_pData)
}
save(final_model,file="final_aenc.Rdata")