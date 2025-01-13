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

# ---------------------------------------------------- resampling functions
do_resampling_autoenc <- function(data, hps, curr_coh){
  inner_splits <- group_vfold_cv(data, group = cohort)
  inner_perf = setNames(data.frame(matrix(ncol = 2, nrow = nrow(hps))), c("hp_indx", "ci"))
  for (i in 1:nrow(hps)){
      print(i)
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
        print(ncol(X_train_inner))
        print(ncol(X_test_inner))
        # rfsrc_tmp <- rfsrc(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_inner, ntree = hps[i,'ntree'])
        #print(hps[i,])
        rfsrc_tmp <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_inner, 
          perf.type = 'none', 
          mtry = hps[i, 'mtry'], ntree = hps[i, 'ntree'], nodesize = hps[i, 'nodesize'], 
          nsplit = hps[i, 'nsplit'], save.memory = TRUE,
          forest = TRUE)

        preds <- predict(rfsrc_tmp, newdata = X_test_inner)
        ci <- get.cindex(X_test_inner$MONTH_TO_BCR, X_test_inner$BCR_STATUS, -preds$predicted)
        #print(ci)
        mean_ci[j] = ci
        }
      #print(mean_ci)
      inner_perf[i, ] <- c(i, mean(mean_ci))
    }
    #print(inner_perf)
    best_hp <- inner_perf[which.max(inner_perf$ci), 'hp_indx']
    return(hps[best_hp,])
  }


do_resampling <- function(data, hps, curr_coh) {
  inner_splits <- group_vfold_cv(data, group = cohort)
  inner_perf = setNames(data.frame(matrix(ncol = 2, nrow = nrow(hps))), c("hp_indx", "ci"))
  for (i in 1:nrow(hps)){
    print(i)
    mean_ci <- numeric(8)
    for (j in seq_along(inner_splits$splits)) {
      inner_split <- inner_splits$splits[[j]]
      inner_train <- analysis(inner_split)
      inner_test <- assessment(inner_split)
      test_cohort <- as.character(inner_test$cohort[1])
      print(test_cohort)
      # TODO: load csv file using cur cohort and test cohort
      # relaod inner train, inner split

      X_train_inner <- as.data.frame(inner_train %>% select(-c(cohort)))
      X_test_inner <- as.data.frame(inner_test %>% select(-c(cohort)))

      # rfsrc_tmp <- rfsrc(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_inner, ntree = hps[i,'ntree'])
      #print(hps[i,])
      rfsrc_tmp <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = X_train_inner, 
        perf.type = 'none', 
        mtry = hps[i, 'mtry'], ntree = hps[i, 'ntree'], nodesize = hps[i, 'nodesize'], 
        nsplit = hps[i, 'nsplit'], save.memory = TRUE,
        forest = TRUE)

      preds <- predict(rfsrc_tmp, newdata = X_test_inner)
      ci <- get.cindex(X_test_inner$MONTH_TO_BCR, X_test_inner$BCR_STATUS, -preds$predicted)
      #print(ci)
      mean_ci[j] = ci
      }
    #print(mean_ci)
    inner_perf[i, ] <- c(i, mean(mean_ci))
  }
  #print(inner_perf)
  best_hp <- inner_perf[which.max(inner_perf$ci), 'hp_indx']
  return(hps[best_hp,])
}

# ------------------------------------------------------------- Load data
prepare_data <- function(use_exprs, use_pData, vars_pData = NA, use_aenc = FALSE){
    if(use_exprs){
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
        exprs_data[, 1] <- NULL
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
    } else if (!use_pData && use_aenc){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort)
        return(df)
    }
}

data_cmplt = prepare_data(FALSE, FALSE, c("AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA'), use_aenc = TRUE)  
print(str(data_cmplt))
# ------------------------------------------------------------- Create Splits and grids for tuning
outer_splits <- group_vfold_cv(data_cmplt, group = cohort)

hyper_grid <- expand.grid(
  ntree = c(100), 
  nsplit = 10, 
  mtry = c(ceiling(sqrt(ncol(data_cmplt))), ceiling(log2(ncol(data_cmplt))), NULL), 
  nodesize = c(10, 15, 20), 
  perf.type = "none", 
  save.memory = TRUE
)

print(hyper_grid)

# ------------------------------------------------------------ Do nested resampling with AUTOENC
outer_perf = setNames(data.frame(matrix(ncol = 2, nrow = 9)), c("testing_cohort", "ci"))
for (i in seq_along(outer_splits$splits)) {
  # Get the split object
  outer_split <- outer_splits$splits[[i]]
  outer_train <- analysis(outer_split)
  outer_test <- assessment(outer_split)
  test_cohort <- as.character(outer_test$cohort[1])
  # TODO: Generate path for test cohort
  print(test_cohort)

  best_hps <- do_resampling_autoenc(outer_train, hyper_grid, test_cohort)
  print(best_hps)
  print(best_hps[1, 'ntree'])
  print(best_hps[1, 'mtry'])

  data_path <- paste0('pretrnd_models_ae\\csv\\' , test_cohort, '.csv') 
  anec_data = read.csv(data_path) %>% mutate_if(is.character, factor)
  print(str(anec_data))

  X_train_outer <- as.data.frame(outer_train)
  X_test_outer <- as.data.frame(outer_test)

  X_train_outer = left_join(X_train_outer, data, by = "X")
  X_test_outer = left_join(X_test_outer, data, by = "X")

  X_train_outer <- as.data.frame(X_train_outer %>% select(-c(cohort, X)))
  X_test_outer <- as.data.frame(X_test_outer %>% select(-c(cohort, X)))
  print(ncol(X_train_outer))
  print(ncol(X_test_outer))

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
write.csv(outer_perf, "rsf_aenc_pData.csv")


final_best_hps <- do_resampling_autoenc(data_cmplt, hyper_grid, '')
data_path <- paste0('pretrnd_models_ae\\csv\\pretrnd_cmplt.csv') 
aenc_data <- read.csv(data_path) %>% mutate_if(is.character, factor)
aenc_data <- left_join(data_cmplt, aenc_data, by = "X")
str(aenc_data)
anec_data <- as.data.frame(aenc_data %>% select(-c(cohort, X)))
# todo: insert HPs from above + Remove irrelevant cols
final_model <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = anec_data, 
        perf.type = 'none', 
        mtry = final_best_hps[1, 'mtry'], ntree = final_best_hps[1, 'ntree'], nodesize = final_best_hps[1, 'nodesize'], 
        nsplit = final_best_hps[1, 'nsplit'], save.memory = FALSE,
        forest = TRUE)


save(final_model,file="rsf_aenc_pData.Rdata")


# ------------------------------------------------------------- Do nested resampling wo autoenc
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
#   print(best_hps)
#   print(best_hps[1, 'ntree'])
#   print(best_hps[1, 'mtry'])
#   print(i)

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
# write.csv(outer_perf, "rsf_exprs_imp_pData.csv")

# ------------------------------------------------------------- Tuning + fitting of final model
# final_best_hps <- do_resampling(data, hyper_grid)
# data <- as.data.frame(data %>% select(-c(cohort)))
# # todo: insert HPs from above + Remove irrelevant cols
# final_model <- rfsrc.fast(Surv(MONTH_TO_BCR, BCR_STATUS) ~ . , data = data, 
#         perf.type = 'none', 
#         mtry = final_best_hps[1, 'mtry'], ntree = final_best_hps[1, 'ntree'], nodesize = final_best_hps[1, 'nodesize'], 
#         nsplit = final_best_hps[1, 'nsplit'], save.memory = FALSE,
#         forest = TRUE)


# save(final_model,file="rsf_pData.Rdata")