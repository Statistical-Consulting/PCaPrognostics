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
do_resampling <- function(data, alphas) {
  # Prepare survival outcome and predictors
  y_train <- Surv(data$MONTH_TO_BCR, data$BCR_STATUS)
  inner_indcs <- as.numeric(as.factor(data$cohort))
  x_train <- as.matrix(data %>% select(-c(MONTH_TO_BCR, BCR_STATUS, cohort)))
  
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
    }
    if(use_pData && use_exprs && !use_aenc){
        return(cbind(df_pData, exprs_data))

    } else if (use_pData && !use_exprs && !use_aenc) {
       return(df_pData)
    } else if (use_pData && use_aenc){
        return(df_pData)
    } else if ((!use_pData && use_aenc)){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort)
        return(df)
    } else if((!use_pData && use_exprs)){
        MONTH_TO_BCR <- df_pData$MONTH_TO_BCR
        BCR_STATUS <- df_pData$BCR_STATUS
        X <- df_pData$X
        df <- data.frame(MONTH_TO_BCR = MONTH_TO_BCR, BCR_STATUS = BCR_STATUS, X = X, cohort = cohort, exprs_data)
        return(df)
    }
}

# ------------------------------------------------------------- Modelling
# set bools for preparing the data
use_aenc = FALSE
use_inter = FALSE
use_exprs = FALSE
use_pData = TRUE
vars_pData = c("AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')

data_cmplt = prepare_data(use_exprs, use_inter, use_pData, use_aenc, vars_pData)

# Create Splits and grids for tuning
outer_splits <- group_vfold_cv(data_cmplt, group = cohort)
candidate_alphas <- seq(0.2, 1, by = 0.2)

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
    } 
    outer_train <- as.data.frame(outer_train %>% select(-c(X)))
    outer_test <- as.data.frame(outer_test %>% select(-c(X)))

    best_mod <- do_resampling(outer_train, candidate_alphas)


y_test_outer <- Surv(outer_test$MONTH_TO_BCR, outer_test$BCR_STATUS)
X_test_outer <- as.matrix(outer_test %>% select(-c(cohort, MONTH_TO_BCR, BCR_STATUS)))

test_preds_se <- predict(best_mod, X_test_outer,  s = 'lambda.1se')
test_preds_min <- predict(best_mod, X_test_outer,  s = 'lambda.min')
outer_cindex_se <- apply(test_preds_se, 2, glmnet::Cindex, y=y_test_outer)
outer_cindex_min <- apply(test_preds_min, 2, glmnet::Cindex, y=y_test_outer)
outer_perf[i, ] <- c(test_cohort, outer_cindex_se, outer_cindex_min)
}

write.csv(outer_perf, "test.csv")

# # --------------------------------------------------------------- Tuning + fitting of final model
if (use_aenc){
  data_path <- paste0('pretrnd_models_ae\\csv\\pretrnd_cmplt.csv') 
  aenc_data <- read.csv(data_path) %>% mutate_if(is.character, factor)
  aenc_data <- left_join(data_cmplt, aenc_data, by = "X")
  anec_data <- as.data.frame(aenc_data %>% select(-c(X)))
  final_model <- do_resampling(anec_data, alphas = candidate_alphas)
} else {
  data_cmplt <- as.data.frame(data_cmplt %>% select(-c(X)))
  final_model <- do_resampling(data_cmplt, candidate_alphas)
}
save(final_model,file="test.Rdata")

# ------------------------------------------------------------- 
# library(survival)



# # Trainingsdaten laden für baseline hazard
# train_data <- read.csv('data/merged_data/pData/imputed/merged_imputed_pData.csv')
# train_data$MONTH_TO_BCR <- as.numeric(as.character(train_data$MONTH_TO_BCR))
# train_data$MONTH_TO_BCR[train_data$MONTH_TO_BCR == 0] <- 0.0001
# train_data$BCR_STATUS <- as.numeric(as.logical(train_data$BCR_STATUS))

# # Testdaten vorbereiten
# pen_cox_test_pData_cohort1 <- as.data.frame(read_csv('data/cohort_data/pData/imputed/low_risk_pData_test_cohort2.csv', lazy = TRUE))[,c("AGE", "TISSUE", "GLEASON_SCORE", "PRE_OPERATIVE_PSA")]
# pen_cox_test_pData_cohort1$TISSUE <- "Fresh_frozen"
# pen_cox_test_common_genes_cohort1 <- as.data.frame(read_csv('data/cohort_data/exprs/common_genes_test_imputed_cohort2_low_risk.csv', lazy = TRUE))

# # Manuelles One-hot encoding für TISSUE
# tissue_encoded <- data.frame(
#   TISSUE_FFPE = 0,
#   TISSUE_Fresh_frozen = 1,
#   TISSUE_Snap_frozen = 0
# )

# # Numerische klinische Daten
# clinical_num <- pen_cox_test_pData_cohort1 %>% 
#   select(AGE, GLEASON_SCORE, PRE_OPERATIVE_PSA)

# # Alles zusammenführen
# X_test <- cbind(clinical_num, tissue_encoded, pen_cox_test_common_genes_cohort1)

# # In Matrix umwandeln
# X_test_matrix <- as.matrix(X_test[,-7])

# # Load the saved model
# loaded_objects <- load("models/pen_cox/results/model/pen_lasso_exprs_imputed_pData.Rdata")



# # Angepasste Survival-Kurven Vorhersage-Funktion für cv.glmnet
# predict_survival_curve <- function(cv_fit, X_test, train_time, train_status, max_time = 120) {
#   # Get risk scores using lambda.min
#   risk_scores <- predict(cv_fit, X_test, s = "lambda.min", type = "link")
 
#   # Create baseline hazard using training data
#   base_surv <- survfit(Surv(train_time, train_status) ~ 1)

#   # Get time points and cumulative hazard
#   times <- base_surv$time
#   cumhaz <- base_surv$cumhaz
  
#   # Calculate survival probabilities for each patient
#   surv_matrix <- exp(-outer(exp(risk_scores), cumhaz))
  
#   # Calculate mean survival
#   mean_surv <- colMeans(surv_matrix)
  
#   # Create dataframe
#   surv_df <- data.frame(
#     time = times,
#     survival = mean_surv
#   )
  
#   return(surv_df)
# }

# test <- function(cv_fit, X_test, train_time, train_status, max_time = 120) {
#   # Predict risk scores (log hazard)
# risk_scores <- predict(cv_fit, X_test, s = "lambda.min", type = "link")

# # Extract baseline hazard from the Cox model
# base_haz <- basehaz(cv_fit)

# # Get time points and cumulative hazard
# times <- base_haz$time
# cumhaz <- base_haz$hazard  # Baseline cumulative hazard

# # Calculate survival probabilities using Cox model formula
# surv_matrix <- outer(exp(-risk_scores), cumhaz, `^`)

# # Calculate mean survival across patients
# mean_surv <- colMeans(surv_matrix)

# # Create dataframe with time and survival probabilities
# surv_df <- data.frame(
#   time = times,
#   survival = mean_surv
# )
# return(surv_df)
# }

# # Survival curves berechnen und speichern
# surv_data <- predict_survival_curve(final_model, X_test_matrix, 
#                                     train_time = train_data$MONTH_TO_BCR,
#                                     train_status = train_data$BCR_STATUS)
# avrg_survival_data <- data.frame(
#   time = surv_data$time,
#   survival = colMeans(surv_data[, 2:649], na.rm = TRUE)
# )

# avrg_survival_data_cohort1_low_risk <- avrg_survival_data
# avrg_survival_data_cohort1_high_risk <- avrg_survival_data
# avrg_survival_data_cohort2_low_risk <- avrg_survival_data
# avrg_survival_data_cohort2_high_risk <- avrg_survival_data


# all_avrg_survival_data <- cbind(avrg_survival_data_cohort1_low_risk, avrg_survival_data_cohort1_high_risk$survival,
#                                 avrg_survival_data_cohort2_low_risk$survival, avrg_survival_data_cohort2_high_risk$survival)


# colnames(all_avrg_survival_data) <- c("time", "cohort1_low_risk", "cohort1_high_risk", "cohort2_low_risk", "cohort2_high_risk")
# write.csv(all_avrg_survival_data, "data/predicted_survival_curves_coxph.csv", row.names = FALSE)

# # Plot erstellen
# ggplot(surv_data, aes(x = time, y = survival)) +
#   geom_step() +
#   theme_minimal() +
#   labs(x = "Time (months)", 
#        y = "Survival Probability",
#        title = "Predicted Survival Curve for Test Cohort") +
#   ylim(0, 1)
# ggsave("predicted_survival_curve.pdf", width = 10, height = 6)
