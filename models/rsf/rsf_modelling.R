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


#' @description Performs Nested Resampling with Autoencoder Data
#' @param data Data frame containing the training data for tuning.
#' @param hps Data frame of hyperparameter combinations, where each row is one set.
#' @param curr_coh String representing the current cohort
#' @return A single row (set of hyperparameters) from hps that achieved the highest
#'   average concordance index (c-index) across the inner folds.
do_resampling_autoenc <- function(data, hps, curr_coh){
  inner_splits <- group_vfold_cv(data, group = cohort)
  inner_perf = setNames(data.frame(matrix(ncol = 2, nrow = nrow(hps))), c("hp_indx", "ci"))
  for (i in 1:nrow(hps)){
      mean_ci <- numeric(8)
      for (j in seq_along(inner_splits$splits)) {
        inner_split <- inner_splits$splits[[j]]
        inner_train <- analysis(inner_split)
        inner_test <- assessment(inner_split)
        test_cohort <- as.character(inner_test$cohort[1])
        if(curr_coh == ""){
          data_path <- paste0('pretrnd_models_ae\\csv\\' , test_cohort, '.csv') 

        }else {
          data_path <- paste0('pretrnd_models_ae\\csv\\' , curr_coh, '_', test_cohort, '.csv') 

        }
        data = read.csv(data_path) %>% mutate_if(is.character, factor)

        X_train_inner <- as.data.frame(inner_train)
        X_test_inner <- as.data.frame(inner_test)

        X_train_inner = left_join(X_train_inner, data, by = "X")
        X_test_inner = left_join(X_test_inner, data, by = "X")

        X_train_inner <- as.data.frame(X_train_inner %>% select(-c(cohort, X)))
        X_test_inner <- as.data.frame(X_test_inner %>% select(-c(cohort, X)))
        rfsrc_tmp <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_inner, 
          perf.type = 'none', 
          mtry = hps[i, 'mtry'], ntree = hps[i, 'ntree'], nodesize = hps[i, 'nodesize'], 
          nsplit = hps[i, 'nsplit'], save.memory = TRUE,
          forest = TRUE)

        preds <- predict(rfsrc_tmp, newdata = X_test_inner)
        ci <- get.cindex(X_test_inner$MONTH_TO_BCR, X_test_inner$BCR_STATUS, -preds$predicted)
        mean_ci[j] = ci
        }
      inner_perf[i, ] <- c(i, mean(mean_ci))
    }
    best_hp <- inner_perf[which.max(inner_perf$ci), 'hp_indx']
    return(hps[best_hp,])
  }


#' @description Performs Nested Resampling without Autoencoder Data
#' @param data Data frame containing the training data for tuning.
#' @param hps Data frame of hyperparameter combinations, where each row is one set.
#' @param curr_coh String representing the current cohort
#' @return A single row (set of hyperparameters) from hps that achieved the highest
#'   average concordance index (c-index) across the inner folds.
do_resampling <- function(data, hps, curr_coh) {
  inner_splits <- group_vfold_cv(data, group = cohort)
  inner_perf = setNames(data.frame(matrix(ncol = 2, nrow = nrow(hps))), c("hp_indx", "ci"))
  for (i in 1:nrow(hps)){
    mean_ci <- numeric(8)
    for (j in seq_along(inner_splits$splits)) {
      inner_split <- inner_splits$splits[[j]]
      inner_train <- analysis(inner_split)
      inner_test <- assessment(inner_split)
      test_cohort <- as.character(inner_test$cohort[1])
      print(test_cohort)

      X_train_inner <- as.data.frame(inner_train %>% select(-c(cohort)))
      X_test_inner <- as.data.frame(inner_test %>% select(-c(cohort)))

      rfsrc_tmp <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_inner, 
        perf.type = 'none', 
        mtry = hps[i, 'mtry'], ntree = hps[i, 'ntree'], nodesize = hps[i, 'nodesize'], 
        nsplit = hps[i, 'nsplit'], save.memory = TRUE,
        forest = TRUE)

      preds <- predict(rfsrc_tmp, newdata = X_test_inner)
      ci <- get.cindex(X_test_inner$MONTH_TO_BCR, X_test_inner$BCR_STATUS, -preds$predicted)
      mean_ci[j] = ci
      }
    inner_perf[i, ] <- c(i, mean(mean_ci))
  }
  best_hp <- inner_perf[which.max(inner_perf$ci), 'hp_indx']
  return(hps[best_hp,])
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
        print(str(df_pData))
        df_pData = df_pData %>% mutate(AGE = as.numeric(AGE), PRE_OPERATIVE_PSA = as.numeric(PRE_OPERATIVE_PSA), GLEASON_SCORE = as.numeric(GLEASON_SCORE)) 
        df_pData$MONTH_TO_BCR <- as.numeric(as.character(df_pData$MONTH_TO_BCR))
        df_pData$MONTH_TO_BCR[df_pData$MONTH_TO_BCR == 0] <- 0.0001
        cohort <- sub("\\..*", "", df_pData$X)

        # doesn't need one hot encoding
    if(length(vars_pData) != 0){
        relevant_vars <- c('X', 'MONTH_TO_BCR', 'BCR_STATUS', vars_pData)
        df_pData <- df_pData[, relevant_vars]
        df_pData <- df_pData %>%
            as_data_frame() %>%
            mutate_if(is.character, factor)

        df_pData$cohort <- cohort
    }
    if(use_pData && use_exprs && !use_aenc){
        return(cbind(df_pData, exprs_data))

    } else if (use_pData && !use_exprs && !use_aenc) {
       return(df_pData)
    } else if (!use_pData && use_exprs && !use_aenc){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        return(cbind(MONTH_TO_BCR, BCR_STATUS, X, cohort, exprs_data))
    } else if (use_pData && use_aenc){
        return(df_pData)
    } else if ((!use_pData && use_aenc)){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort)
        return(df)
    } else if (!use_pData && use_aenc){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort)
        return(df)
    }
}

# ------------------------------------------------------------- Modelling
# set bools for preparing the data
use_aenc = FALSE # if latent space from AE is to be used
use_inter = TRUE # if gene data in general is to be used
use_exprs = TRUE # if intersection data is to be used --> if FALSE & use_inter then imputed/common genes are used
use_pData = TRUE # if clinical data is used
vars_pData = c("AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')


data_cmplt = prepare_data(use_exprs, use_inter, use_pData, use_aenc, vars_pData)
# ------------------------------------------------------------- Create Splits and grids for tuning
outer_splits <- group_vfold_cv(data_cmplt, group = cohort)

# hyper_grid <- expand.grid(
#   ntree = c(100), 
#   nsplit = 10, 
#   mtry = c(ceiling(sqrt(ncol(data_cmplt))), ceiling(log2(ncol(data_cmplt))), NULL), 
#   nodesize = c(10, 15, 20), 
#   perf.type = "none", 
#   save.memory = TRUE
# )


hyper_grid <- expand.grid(
  ntree = c(100), 
  nsplit = 10, 
  mtry = c(ceiling(sqrt(ncol(data_cmplt)))), 
  nodesize = c(20), 
  perf.type = "none", 
  save.memory = TRUE
)

# ------------------------------------------------------------ Do nested resampling
outer_perf = setNames(data.frame(matrix(ncol = 2, nrow = 9)), c("testing_cohort", "ci"))
for (i in seq_along(outer_splits$splits)) {
  # Get the split object
  outer_split <- outer_splits$splits[[i]]
  outer_train <- analysis(outer_split)
  outer_test <- assessment(outer_split)
  test_cohort <- as.character(outer_test$cohort[1])

  if(use_aenc){
      best_hps <- do_resampling_autoenc(outer_train, hyper_grid, test_cohort)
      data_path <- paste0('pretrnd_models_ae\\csv\\' , test_cohort, '.csv') 
      anec_data = read.csv(data_path) %>% mutate_if(is.character, factor)
      X_train_outer <- as.data.frame(outer_train)
      X_test_outer <- as.data.frame(outer_test)

      X_train_outer = left_join(X_train_outer, anec_data, by = "X")
      X_test_outer = left_join(X_test_outer, anec_data, by = "X")

      X_train_outer <- as.data.frame(X_train_outer %>% select(-c(cohort, X)))
      X_test_outer <- as.data.frame(X_test_outer %>% select(-c(cohort, X)))
  }
  else {
     best_hps <- do_resampling(outer_train, hyper_grid, test_cohort)
     X_train_outer <- as.data.frame(outer_train %>% select(-c(cohort, X)))
     X_test_outer <- as.data.frame(outer_test %>% select(-c(cohort, X)))
  }

  outer_mod <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_outer, 
        perf.type = 'none', 
        mtry = best_hps[1, 'mtry'], nodesize = best_hps[1, 'nodesize'], 
        nsplit = best_hps[1, 'nsplit'], save.memory = TRUE, ntree = best_hps[1, 'ntree'],
        forest = TRUE)

  test_preds <- predict(outer_mod, newdata = X_test_outer)
  outer_cindex <- get.cindex(X_test_outer$MONTH_TO_BCR, X_test_outer$BCR_STATUS, -test_preds$predicted)
  
  outer_perf[i, ] <- c(test_cohort, outer_cindex)
}
print(outer_perf)
write.csv(outer_perf, "test.csv")

if(use_aenc){
  final_best_hps <- do_resampling_autoenc(data_cmplt, hyper_grid, '')
  data_path <- paste0('pretrnd_models_ae\\csv\\pretrnd_cmplt.csv') 
  aenc_data <- read.csv(data_path) %>% mutate_if(is.character, factor)
  aenc_data <- left_join(data_cmplt, aenc_data, by = "X")
  str(aenc_data)
  anec_data <- as.data.frame(aenc_data %>% select(-c(cohort, X)))
  final_model <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = anec_data, 
          perf.type = 'none', 
          mtry = final_best_hps[1, 'mtry'], ntree = final_best_hps[1, 'ntree'], nodesize = final_best_hps[1, 'nodesize'], 
          nsplit = final_best_hps[1, 'nsplit'], save.memory = FALSE,
          forest = TRUE)
} else {
  final_best_hps <- do_resampling(data_cmplt, hyper_grid)
  data_cmplt <- as.data.frame(data_cmplt %>% select(-c(cohort)))
  final_model <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = data_cmplt, 
          perf.type = 'none', 
          mtry = final_best_hps[1, 'mtry'], ntree = final_best_hps[1, 'ntree'], nodesize = final_best_hps[1, 'nodesize'], 
          nsplit = final_best_hps[1, 'nsplit'], save.memory = FALSE,
          forest = TRUE)
}
save(final_model,file="test.Rdata")


# # ------------------------------------------------------------- Do nested resampling wo autoenc
# outer_perf = setNames(data.frame(matrix(ncol = 2, nrow = 9)), c("testing_cohort", "ci"))
# for (i in seq_along(outer_splits$splits)) {
#   # Get the split object
#   outer_split <- outer_splits$splits[[i]]
#   outer_train <- analysis(outer_split)
#   outer_test <- assessment(outer_split)
#   test_cohort <- as.character(outer_test$cohort[1])
#   # TODO: Generate path for test cohort
#   print(test_cohort)

#   best_hps <- do_resampling(outer_train, hyper_grid, test_cohort)

#   # reload outer train, outer test based on 
#   X_train_outer <- as.data.frame(outer_train %>% select(-c(cohort, X)))
#   X_test_outer <- as.data.frame(outer_test %>% select(-c(cohort, X)))

#   outer_mod <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_outer, 
#         perf.type = 'none', 
#         mtry = best_hps[1, 'mtry'], nodesize = best_hps[1, 'nodesize'], 
#         nsplit = best_hps[1, 'nsplit'], save.memory = TRUE, ntree = best_hps[1, 'ntree'],
#         forest = TRUE)

#   test_preds <- predict(outer_mod, newdata = X_test_outer)
#   outer_cindex <- get.cindex(X_test_outer$MONTH_TO_BCR, X_test_outer$BCR_STATUS, -test_preds$predicted)
  
#   outer_perf[i, ] <- c(test_cohort, outer_cindex)
# }

# print(outer_perf)
# write.csv(outer_perf, "rsf_scores.csv")

# # ------------------------------------------------------------- Tuning + fitting of final model
# final_best_hps <- do_resampling(data_cmplt, hyper_grid)
# data_cmplt <- as.data.frame(data_cmplt %>% select(-c(cohort)))
# # todo: insert HPs from above + Remove irrelevant cols
# final_model <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = data_cmplt, 
#         perf.type = 'none', 
#         mtry = final_best_hps[1, 'mtry'], ntree = final_best_hps[1, 'ntree'], nodesize = final_best_hps[1, 'nodesize'], 
#         nsplit = final_best_hps[1, 'nsplit'], save.memory = FALSE,
#         forest = TRUE)


# save(final_model,file="rsf_scores.Rdata")
