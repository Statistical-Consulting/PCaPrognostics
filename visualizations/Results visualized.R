library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
#################################################################################
# Test Data Results



#######Keep this part to properly create the csv files
ov_results_cBoost <- as.data.frame(read_csv('results_modelling_ovs/ov_cBoost.csv', lazy = TRUE))
if(ov_results_cBoost$model[1] !="cboost_autoencoder_paper") {
  ov_results_cBoost[c(6, 7), ] <- ov_results_cBoost[c(7, 6), ]
  ov_results_cBoost[c(1, 2), ] <- ov_results_cBoost[c(2, 1), ]
  colnames(ov_results_cBoost)[colnames(ov_results_cBoost)=="ci_coh1"] <- "ci_cohort1"
  colnames(ov_results_cBoost)[colnames(ov_results_cBoost)=="ci_coh2"] <- "ci_cohort2"
  write.csv(ov_results_cBoost, file.path(".", "results_modelling_ovs", "ov_cBoost.csv"))
}

ov_results_coxPas <- as.data.frame(read_csv('results_modelling_ovs/ov_coxPAS.csv', lazy = TRUE))
if(colnames(ov_results_coxPas)[5] =="ci_coh1"){
  colnames(ov_results_coxPas)[colnames(ov_results_coxPas)=="ci_coh1"] <- "ci_cohort1"
  colnames(ov_results_coxPas)[colnames(ov_results_coxPas)=="ci_coh2"] <- "ci_cohort2"
  write.csv(ov_results_coxPas, file.path(".", "results_modelling_ovs", "ov_coxPAS.csv"))
  print("r")
}


ov_results_coxph <- as.data.frame(read_csv('results_modelling_ovs/ov_coxph.csv', lazy = TRUE))
ov_results_DeepSurv <- as.data.frame(read_csv('results_modelling_ovs/ov_DeepSurv.csv', lazy = TRUE))
ov_results_prio <- as.data.frame(read_csv('results_modelling_ovs/ov_prio.csv', lazy = TRUE))
ov_results_rsf <- as.data.frame(read_csv('results_modelling_ovs/ov_rsf.csv', lazy = TRUE))




ov_results_DeepSurv$ci_coh1[3:7] <- c(0.6627, 0.6979, 0.6601, 0.6795, 0.6464)
ov_results_DeepSurv$ci_coh2[3:7] <- c(0.8173, 0.7683, 0.7019, 0.7481, 0.7447)
ov_results_DeepSurv[c(4,6),3] <- c(0.6777, 0.6982)
colnames(ov_results_DeepSurv)[colnames(ov_results_DeepSurv)=="ci_coh1"] <- "ci_cohort1"
colnames(ov_results_DeepSurv)[colnames(ov_results_DeepSurv)=="ci_coh2"] <- "ci_cohort2"


#manually adding results of deep surv on test data

if (any(is.na(ov_results_DeepSurv$ci_coh1))) {
  ov_results_DeepSurv$ci_coh1[3:7] <- c(0.6627, 0.6979, 0.6601, 0.6795, 0.6464)
  ov_results_DeepSurv$ci_coh2[3:7] <- c(0.8173, 0.7683, 0.7019, 0.7481, 0.7447)
  ov_results_DeepSurv[c(4,6),3] <- c(0.6777, 0.6982)
  colnames(ov_results_DeepSurv)[colnames(ov_results_DeepSurv)=="ci_coh1"] <- "ci_cohort1"
  colnames(ov_results_DeepSurv)[colnames(ov_results_DeepSurv)=="ci_coh2"] <- "ci_cohort2"
  write.csv(ov_results_DeepSurv, file.path(".", "results_modelling_ovs", "ov_DeepSurv.csv"))
}



ordered_datasets <- c("autoencoder", "autoencoder+pData", "common", "common+pData", "intersect", "intersect+pData", "pData")
ov_results_cBoost <- cbind(ov_results_cBoost, ordered_datasets)
ov_results_coxph <- cbind(ov_results_coxph, ordered_datasets)
ov_results_DeepSurv <- cbind(ov_results_DeepSurv, ordered_datasets)
ov_results_rsf <- cbind(ov_results_rsf, ordered_datasets)


#################################################################################
# Prepare Train Data Results
##cBoost load and filter and find best dataset
splits_results_cBoost <- as.data.frame(read_csv('results_modelling_splits/splits_cBoost.csv', lazy = TRUE))
subset_splits_results <- splits_results_cBoost[splits_results_cBoost$dataset %in% c("Intersection", "Imputed"), ] %>%filter(dataset %in% c("Intersection", "Imputed"))
splits_best_cBoost <- subset_splits_results %>%
  group_by(test_cohort) %>%
  slice_max(order_by = ci, n = 1) %>%
  ungroup()

# coxPas load and filter and find best dataset
splits_results_coxPas <- as.data.frame(read_csv('results_modelling_splits/splits_coxPAS.csv', lazy = TRUE))
splits_best_coxPas <- splits_results_coxPas

# coxph load and filter and find best dataset
splits_results_coxph <- as.data.frame(read_csv('results_modelling_splits/splits_coxph.csv', lazy = TRUE))
subset_splits_results_coxph <- splits_results_coxph %>%
  filter(dataset %in% c("Intersection", "Imputed"))
splits_best_coxph <- subset_splits_results_coxph %>%
  group_by(testing_cohort) %>%
  slice_max(order_by = ci_min, n = 1) %>%#ci_min genutzt, da mean ci besser als bei ci_se
  ungroup()
colnames(splits_best_coxph) <- c("...1", "test_cohort", "ci_se", "ci", "model", "dataset")


# DeepSurv load and filter and find best dataset
splits_results_DeepSurv <- as.data.frame(read_csv('results_modelling_splits/splits_DeepSurv.csv', lazy = TRUE))
subset_splits_results_DeepSurv <- splits_results_DeepSurv %>%
  filter(dataset %in% c("Intersection", "Imputed"))
splits_best_DeepSurv <- subset_splits_results_DeepSurv %>%
  group_by(test_cohort) %>%
  slice_max(order_by = ci, n = 1) %>%
  ungroup()


# prio load and filter and find best dataset
splits_results_prio <- as.data.frame(read_csv('results_modelling_splits/splits_prio.csv', lazy = TRUE))
colnames(splits_results_prio)[colnames(splits_results_prio)=="testing_cohort"] <- "test_cohort"
splits_best_prio <- splits_results_prio

# rsf load and filter and find best dataset
splits_results_rsf <- as.data.frame(read_csv('results_modelling_splits/splits_rsf.csv', lazy = TRUE))
subset_splits_results_rsf <- splits_results_rsf %>%
  filter(dataset %in% c("Intersection", "Imputed"))
splits_best_rsf <- subset_splits_results_rsf %>%
  group_by(testing_cohort) %>%
  slice_max(order_by = ci, n = 1) %>%
  ungroup()
colnames(splits_best_rsf)[colnames(splits_best_rsf)=="testing_cohort"] <- "test_cohort"


splits_results_score <- as.data.frame(read_csv('results_modelling_splits/splits_score_cindices.csv', lazy = TRUE))
colnames(splits_results_score)[colnames(splits_results_score) == "c_index"] <- "ci"
colnames(splits_results_score)[colnames(splits_results_score) == "cohort"] <- "test_cohort"
splits_best_score <- splits_results_score

all_ov_df <- c("ov_results_cBoost", "ov_results_coxPas", "ov_results_coxph","ov_results_DeepSurv",
               "ov_results_prio", "ov_results_rsf")
ov_results_cBoost <- cbind(ov_results_cBoost, ci_mean = rowMeans(ov_results_cBoost[, c(5, 6)], na.rm = TRUE))
ov_results_coxPas <- cbind(ov_results_coxPas, ci_mean = rowMeans(ov_results_coxPas[, c(5, 6)], na.rm = TRUE))
ov_results_coxph <- cbind(ov_results_coxph, ci_mean = rowMeans(ov_results_coxph[, c(5, 6)], na.rm = TRUE))
ov_results_DeepSurv <- cbind(ov_results_DeepSurv, ci_mean = rowMeans(ov_results_DeepSurv[, c(5, 6)], na.rm = TRUE))
ov_results_prio <- cbind(ov_results_prio, ci_mean = rowMeans(ov_results_prio[, c(4, 5)], na.rm = TRUE))
ov_results_rsf <- cbind(ov_results_rsf, ci_mean = rowMeans(ov_results_rsf[, c(5, 6)], na.rm = TRUE))



#################################################################################
#Boxplot best Model results by cohort
# Beispiel: Sicherstellen, dass die Spalte "model" in allen DataFrames vorhanden ist

splits_best_cBoost$model <- "Gradient Boosting"
splits_best_coxPas$model <- "Cox-PASNet"
splits_best_coxph$model <- "Cox"
splits_best_prio$model <- "Priority Lasso"
splits_best_rsf$model <- "Random Survival Forest"
splits_best_score$model <- "ProstaTrend-ffpe"


all_best_splits_data <- bind_rows(
  splits_best_cBoost,
  splits_best_coxPas,
  splits_best_coxph,
  splits_best_prio,
  splits_best_rsf,
  splits_best_score
)

# Mapping der Kohortennamen
rename_map <- c(
  "Atlanta_2014_Long" = "Cohort 1",
  "Belfast_2018_Jain" = "Cohort 2",
  "CamCap_2016_Ross_Adams" = "Cohort 3",
  "CancerMap_2017_Luca" = "Cohort 4",
  "CPC_GENE_2017_Fraser" = "Cohort 5",
  "CPGEA_2020_Li" = "Cohort 6",
  "DKFZ_2018_Gerhauser" = "Cohort 7",
  "MSKCC_2010_Taylor" = "Cohort 8",
  "Stockholm_2016_Ross_Adams" = "Cohort 9"
)

# Test-Kohorten umbenennen
all_best_splits_data <- all_best_splits_data %>%
  mutate(test_cohort = rename_map[test_cohort])

dummy_for_corr_best_splits <- all_best_splits_data %>%
  group_by(test_cohort) %>%
  slice_max(ci, n = 1) %>%
  ungroup()

ggplot(all_best_splits_data[,1:5], aes(x = test_cohort, y = ci)) +
  geom_boxplot(aes(fill = "Other Models"), outlier.shape = NA) + 
  geom_jitter(
    data = all_best_splits_data %>% filter(model == "ProstaTrend-ffpe"),
    aes(color = model), width = 0.2, alpha = 0.7
  ) +
  scale_color_manual("", values = c(
    "ProstaTrend-ffpe" = "#0c4252"
  )) +
  scale_fill_manual("", values = c(
    "Other Models" = "#ffcd66"
  )) +
  labs(
    title = "Best Model per Model Class vs. ProstaTrend-ffpe - Comparison by Cohort",
    x = "Cohorts in Group A",
    y = "C-Index"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),  # Überschrift fett und Schriftgröße 14
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )



#################################################################################
#Sort models by input data types on test data
# hier CoxPas und prio lasso nicht genutzt da nicht für alle datentypen verfügbar

# Modellnamen hinzufügen und DataFrames kombinieren
ov_results_cBoost$general_model <- "Gradient Boosting"     # Gradient Boosting
ov_results_coxph$general_model <- "Pen. Cox PH"           # Pen. Cox PH
ov_results_DeepSurv$general_model <- "DeepSurv"           # DeepSurv
ov_results_rsf$general_model <- "Random Survival Forest"  # Random Survival Forest
ov_results_coxPas$general_model <- "Cox PASNet"            # Cox PASNet
ov_results_prio$general_model <- "Priority Lasso"         # Priority Lasso

# X Achse nochmal umbennen, also all_data usw
# Black Data
#pData->clinical data
#Common Genes
# Intersection Genes
#Pathways + Intersection + Clinical
ov_results_cBoost_dot_plot <- rbind(
  ov_results_cBoost, 
  c("7", "all_data", NA, NA, NA, NA, "all_data", NA, "Gradient Boosting"), 
  c("8", "pathways", NA, NA, NA, NA, "pathways", NA, "Gradient Boosting")
)

ov_results_coxph_dot_plot <- rbind(
  ov_results_coxph,  
  c("7", "all_data", NA, NA, NA, NA, "all_data", NA, "Pen. Cox PH"), 
  c("8", "pathways", NA, NA, NA, NA, "pathways", NA, "Pen. Cox PH")
)

ov_results_DeepSurv_dot_plot <- rbind(
  ov_results_DeepSurv,  
  c("7", "all_data", NA, NA, NA, NA, "all_data", NA, "DeepSurv"), 
  c("8", "pathways", NA, NA, NA, NA, "pathways", NA, "DeepSurv")
)

ov_results_rsf_dot_plot <- rbind(
  ov_results_rsf,  
  c("7", "all_data", NA, NA, NA, NA, "all_data", NA, "Random Survival Forest"), 
  c("8", "pathways", NA, NA, NA, NA, "pathways", NA, "Random Survival Forest")
)

# Cox PASNet Anpassung
ov_results_cox_pas_dot_plot <- ov_results_rsf_dot_plot
ov_results_cox_pas_dot_plot$general_model <- "Cox PASNet"
ov_results_cox_pas_dot_plot[, c(3, 4, 5, 6, 8)] <- NA
ov_results_cox_pas_dot_plot$ci_cohort1[9] <- ov_results_coxPas$ci_cohort1

# Priority Lasso Anpassung
ov_results_prio_dot_plot <- ov_results_rsf_dot_plot
ov_results_prio_dot_plot$general_model <- "Priority Lasso"
ov_results_prio_dot_plot[, c(3, 4, 5, 6, 8)] <- NA
ov_results_prio_dot_plot$ci_cohort1[8] <- ov_results_prio$ci_cohort1

# Kombiniere die Daten mit den Modellen
combined_results <- rbind(
  transform(ov_results_cBoost_dot_plot, model = "Gradient Boosting")[, c("ordered_datasets", "ci_cohort1", "general_model")],
  transform(ov_results_coxph_dot_plot, model = "Pen. Cox PH")[, c("ordered_datasets", "ci_cohort1", "general_model")],
  transform(ov_results_DeepSurv_dot_plot, model = "DeepSurv")[, c("ordered_datasets", "ci_cohort1", "general_model")],
  transform(ov_results_rsf_dot_plot, model = "Random Survival Forest")[, c("ordered_datasets", "ci_cohort1", "general_model")],
  transform(ov_results_cox_pas_dot_plot, model = "Cox PASNet")[, c("ordered_datasets", "ci_cohort1", "general_model")],
  transform(ov_results_prio_dot_plot, model = "Priority Lasso")[, c("ordered_datasets", "ci_cohort1", "general_model")]
)

# Berechnung der Mittelwerte
mean_values <- aggregate(ci_cohort1 ~ ordered_datasets, data = combined_results, FUN = mean)

ggplot() +
  geom_point(
    data = combined_results %>% 
      filter(!is.na(ci_cohort1)) %>% 
      mutate(ci_cohort1 = as.numeric(ci_cohort1)),  # Konvertiere in numerisch
    aes(x = ordered_datasets, y = ci_cohort1, color = general_model), 
    size = 3
  ) +
  scale_y_continuous(
    limits = c(0.6, 0.8),   # Setze die Grenzen der Y-Achse
    breaks = c(0.6, 0.7, 0.8)  # Zeige nur die gewünschten Werte
  ) +
  scale_x_discrete(
    labels = c(
      "all_data" = "Block Data",
      "autoencoder" = "Autoencoder",
      "autoencoder+pData" = "Autoencoder + Clin. Data",
      "common" = "Common ",
      "common+pData" = "Common + Clin. Data",
      "intersect" = "Intersect. ",
      "intersect+pData" = "Intersect. + Clin. Data",
      "pathways" = "Pathway Data",
      "pData" = "Clin. Data"
    )
  ) +
  labs(
    title = "Performance of Models on Cohort 10 by Data Set",
    x = "Data Sets",
    y = "C-Index",
    color = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),  # Überschrift fett und Schriftgröße 14
    axis.text.x = element_text(angle = 45, hjust = 1),    # Rotiert die X-Achsen-Beschriftung
    legend.position = "bottom",                           # Legende unten platzieren
    legend.direction = "horizontal",                      # Legende horizontal ausrichten
    legend.box = "horizontal",                            # Legendenbox horizontal ausrichten
    legend.justification = "center",                      # Legende zentrieren
    legend.margin = margin(t = 10, unit = "pt"),          # Abstand zur Legende vergrößern
    legend.text = element_text(size = 10),                # Größe des Legendentexts
    legend.title = element_text(size = 12)                # Größe des Legendentitels
  ) +
  guides(
    color = guide_legend(nrow = 1)  # Alle Legendenwerte in eine Zeile
  )





###### Calc avrg performances exprs + pdata vs only one
#only one
comb_res_only_one <- combined_results[ combined_results$ordered_datasets %in% c( "common", "pData", "intersect" ) ,]
#both
comb_res_without_pData <- combined_results[ combined_results$ordered_datasets %in% c( "common+pData", "pathways", "intersect+pData", "autoencoder+pData",  "all_data" ) ,]

mean(as.numeric(comb_res_without_pData$ci_cohort1), na.rm = TRUE)
mean(as.numeric(comb_res_only_one$ci_cohort1), na.rm = TRUE)





















#####################################################################
#surival curve ae deepsurv vs. coxph common+pData
# Erstellen eines Dummy-Datensatzes

time_points <- seq(0, 100, length.out = 50)
survival_deepsurv <- exp(-0.03 * time_points) + rnorm(length(time_points), mean = 0, sd = 0.01)
survival_coxph <- exp(-0.025 * time_points) + rnorm(length(time_points), mean = 0, sd = 0.01)

df <- data.frame(
  TimePoints = time_points,
  Survival_DeepSurv = survival_deepsurv,
  Survival_CoxPH = survival_coxph
)



ggplot(df, aes(x = TimePoints)) +
  geom_line(aes(y = Survival_DeepSurv, color = "DeepSurv"), size = 1) +
  geom_line(aes(y = Survival_CoxPH, color = "Cox PH"), linetype = "dashed", size = 1) +
  labs(
    title = "Survival Curves",
    x = "Time Points",
    y = "Survival Probability",
    color = "Model"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("DeepSurv" = "blue", "Cox PH" = "red")) +
  theme(legend.title = element_text(size = 12), legend.text = element_text(size = 10))



#####################################################################
#surival curve ae deepsurv vs. coxph common+pData für differen risk score groups
# Erstellen eines erweiterten Dummy-Datensatzes




# CoxPH Survival-Daten laden
coxph_survival_data <- as.data.frame(read_csv('data/survival_data/predicted_survival_curves_coxph.csv'))
ae_deep_surv_survival_data_coh1_high_risk <- as.data.frame(read_csv('data/survival_data/predicted_survival_curves_cohort1_high_risk.csv'))
ae_deep_surv_survival_data_coh1_low_risk <- as.data.frame(read_csv('data/survival_data/predicted_survival_curves_cohort1_low_risk.csv'))
ae_deep_surv_survival_data_coh2_high_risk <- as.data.frame(read_csv('data/survival_data/predicted_survival_curves_cohort2_high_risk.csv'))
ae_deep_surv_survival_data_coh2_low_risk <- as.data.frame(read_csv('data/survival_data/predicted_survival_curves_cohort2_low_risk.csv'))

# Den DataFrame einlesen (ersetze `df` durch den Namen deines DataFrames)
df <- ae_deep_surv_survival_data_coh1_high_risk

# Den Mittelwert über die Spalten ab der 2. Spalte für jede Zeile berechnen
ae_deep_surv_survival_coh1_high_risk_vec <- rowMeans(ae_deep_surv_survival_data_coh1_high_risk[, 2:ncol(ae_deep_surv_survival_data_coh1_high_risk)], na.rm = TRUE)
ae_deep_surv_survival_coh1_low_risk_vec <- rowMeans(ae_deep_surv_survival_data_coh1_low_risk[, 2:ncol(ae_deep_surv_survival_data_coh1_low_risk)], na.rm = TRUE)
ae_deep_surv_survival_coh2_high_risk_vec <- rowMeans(ae_deep_surv_survival_data_coh2_high_risk[, 2:ncol(ae_deep_surv_survival_data_coh2_high_risk)], na.rm = TRUE)
ae_deep_surv_survival_coh2_low_risk_vec <- rowMeans(ae_deep_surv_survival_data_coh2_low_risk[, 2:ncol(ae_deep_surv_survival_data_coh2_low_risk)], na.rm = TRUE)
ae_deep_surv_surival_times <- ae_deep_surv_survival_data_coh1_high_risk$time





# Survival-Daten vorbereiten
cox_survival_ch1_df <- coxph_survival_data[, 1:3] %>%
  rename(Survival_CoxPH_LowRisk = cohort1_low_risk, Survival_CoxPH_HighRisk = cohort1_high_risk)

ae_deep_surv_survival_ch1_df <- data.frame(
  time = ae_deep_surv_surival_times,
  Survival_DeepSurv_HighRisk = ae_deep_surv_survival_coh1_high_risk_vec,
  Survival_DeepSurv_LowRisk = ae_deep_surv_survival_coh1_low_risk_vec
)




# Vektoren der Zeitpunkte aus beiden DataFrames kombinieren
all_times <- sort(unique(c(coxph_survival_data$time, ae_deep_surv_survival_ch1_df$time)))

# Leeren DataFrame mit allen einzigartigen Zeitpunkten erstellen
survival_df_cohort1 <- data.frame(TimePoints = all_times)

# Survival-Daten aus coxph_survival_data hinzufügen
survival_df_cohort1 <- survival_df_cohort1 %>%
  left_join(
    coxph_survival_data %>%
      rename(Survival_CoxPH_LowRisk = cohort1_low_risk, Survival_CoxPH_HighRisk = cohort1_high_risk),
    by = c("TimePoints" = "time")
  )

# Survival-Daten aus ae_deep_surv_survival_ch1_df hinzufügen (korrekt umbenannt)
survival_df_cohort1 <- survival_df_cohort1 %>%
  left_join(
    ae_deep_surv_survival_ch1_df %>%
      rename(Survival_DeepSurv_LowRisk = Survival_DeepSurv_LowRisk, Survival_DeepSurv_HighRisk = Survival_DeepSurv_HighRisk),
    by = c("TimePoints" = "time")
  )

# Fehlende Werte mit dem letzten bekannten Wert auffüllen
survival_df_cohort1 <- survival_df_cohort1 %>%
  fill(Survival_CoxPH_LowRisk, Survival_CoxPH_HighRisk, Survival_DeepSurv_LowRisk, Survival_DeepSurv_HighRisk, .direction = "down")





# Erstellen des ggplot
ggplot(survival_df_cohort1, aes(x = TimePoints)) +
  geom_line(aes(y = Survival_DeepSurv_LowRisk, color = "DeepSurv Low Risk"), size = 1) +
  geom_line(aes(y = Survival_DeepSurv_HighRisk, color = "DeepSurv High Risk"), size = 1) +
  geom_line(aes(y = Survival_CoxPH_LowRisk, color = "CoxPH Low Risk"), size = 1) +
  geom_line(aes(y = Survival_CoxPH_HighRisk, color = "CoxPH High Risk"),  size = 1) +
  scale_linetype_manual(values = c(
    "DeepSurv Low Risk" = "solid",
    "DeepSurv High Risk" = "dashed",
    "CoxPH Low Risk" = "dotdash",
    "CoxPH High Risk" = "dotted"
  )) +
  labs(
    title = "Cohort 10 Survival Curves for Risk Groups and Selected Models",
    x = "Month",
    y = "Survival Probability",
    color = "Model & Risk Group"
  ) +
  theme_minimal() +
  scale_color_manual(values = c(
    "DeepSurv Low Risk" = "red", 
    "DeepSurv High Risk" = "#ffcd66", 
    "CoxPH Low Risk" = "darkblue", 
    "CoxPH High Risk" = "blue"
  )) +
  scale_x_continuous(limits = c(0, 160)) +  # Begrenzung der x-Achse auf 0 bis 160
  theme(
    plot.title = element_text(face = "bold", size = 14),  # Überschrift fett und Schriftgröße 14
    legend.title = element_text(size = 12), 
    legend.text = element_text(size = 10)
  )


#####################################################################
# Model Performance vs. Risk scores 
# Referenz-Linie
mean_risk <- 0.698

# Find best performances for both respective models cohorts inkluding pData 
models <- c("CatBoost", "PasNet", "PenalizedCox", "DeepSurv","PriorityLasso", "RandomSurvivalForest")
test_results_cohort1_selected_data <- c(ov_results_cBoost[ov_results_cBoost$mean==max(ov_results_cBoost$mean),][,2], 
                                        ov_results_coxPas[ov_results_coxPas$mean==max(ov_results_coxPas$mean),][,2],
                                        ov_results_coxph[ov_results_coxph$mean==max(ov_results_coxph$mean),][,2],
                                        ov_results_DeepSurv[ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean),][,3],
                                        "prio_lasso_all_data",
                                        ov_results_rsf[ov_results_rsf$mean==max(ov_results_rsf$mean),][,2])

test_results_cohort1 <- c(ov_results_cBoost$ci_cohort1[which(ov_results_cBoost$mean==max(ov_results_cBoost$mean))],ov_results_coxPas$ci_cohort1[which(ov_results_coxPas$mean==max(ov_results_coxPas$mean))],ov_results_coxph$ci_cohort1[which(ov_results_coxph$mean==max(ov_results_coxph$mean))],ov_results_DeepSurv$ci_cohort1[which(ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean))],ov_results_prio$ci_cohort1[which(ov_results_prio$mean==max(ov_results_prio$mean))],ov_results_rsf$ci_cohort1[which(ov_results_rsf$mean==max(ov_results_rsf$mean))])
test_results_cohort2_selected_data <- c(ov_results_cBoost[ov_results_cBoost$mean==max(ov_results_cBoost$mean),][,2], 
                                        ov_results_coxPas[ov_results_coxPas$mean==max(ov_results_coxPas$mean),][,2],
                                        ov_results_coxph[ov_results_coxph$mean==max(ov_results_coxph$mean),][,2],
                                        ov_results_DeepSurv[ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean),][,3],
                                        "prio_lasso_all_data",
                                        ov_results_rsf[ov_results_rsf$mean==max(ov_results_rsf$mean),][,2])
test_results_cohort2 <- c(ov_results_cBoost$ci_cohort2[which(ov_results_cBoost$mean==max(ov_results_cBoost$mean))],ov_results_coxPas$ci_cohort2[which(ov_results_coxPas$mean==max(ov_results_coxPas$mean))],ov_results_coxph$ci_cohort2[which(ov_results_coxph$mean==max(ov_results_coxph$mean))],ov_results_DeepSurv$ci_cohort2[which(ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean))],ov_results_prio$ci_cohort2[which(ov_results_prio$mean==max(ov_results_prio$mean))],ov_results_rsf$ci_cohort2[which(ov_results_rsf$mean==max(ov_results_rsf$mean))])







###The same with the best models based on test data
models <- c("CatBoost", "PasNet", "PenalizedCox", "DeepSurv","PriorityLasso", "RandomSurvivalForest")
test_results_cohort1_selected_data_based_on_test <- c(ov_results_cBoost[ov_results_cBoost$ci_cohort1==max(ov_results_cBoost$ci_cohort1),][,2], 
                                        ov_results_coxPas[ov_results_coxPas$ci_cohort1==max(ov_results_coxPas$ci_cohort1),][,2],
                                        ov_results_coxph[ov_results_coxph$ci_cohort1==max(ov_results_coxph$ci_cohort1),][,2],
                                        ov_results_DeepSurv[ov_results_DeepSurv$ci_cohort1==max(ov_results_DeepSurv$ci_cohort1),][,3],
                                        "prio_lasso_all_data",
                                        ov_results_rsf[ov_results_rsf$ci_cohort1==max(ov_results_rsf$ci_cohort1),][,2])
test_results_cohort1_based_on_test <- c(max(ov_results_cBoost$ci_cohort1), max(ov_results_coxPas$ci_cohort1), max(ov_results_coxph$ci_cohort1), max(ov_results_DeepSurv$ci_cohort1), max(ov_results_prio$ci_cohort1), max(ov_results_rsf$ci_cohort1))
test_results_cohort2_selected_data_based_on_test <- c(ov_results_cBoost[ov_results_cBoost$ci_cohort2==max(ov_results_cBoost$ci_cohort2),][,2], 
                                        ov_results_coxPas[ov_results_coxPas$ci_cohort2==max(ov_results_coxPas$ci_cohort2),][,2],
                                        ov_results_coxph[ov_results_coxph$ci_cohort2==max(ov_results_coxph$ci_cohort2),][,2],
                                        ov_results_DeepSurv[ov_results_DeepSurv$ci_cohort2==max(ov_results_DeepSurv$ci_cohort2),][,3],
                                        "prio_lasso_all_data",
                                        ov_results_rsf[ov_results_rsf$ci_cohort2==max(ov_results_rsf$ci_cohort2),][,2])
test_results_cohort2_based_on_test <- c(max(ov_results_cBoost$ci_cohort2), max(ov_results_coxPas$ci_cohort2), max(ov_results_coxph$ci_cohort2), max(ov_results_DeepSurv$ci_cohort2), max(ov_results_prio$ci_cohort2), max(ov_results_rsf$ci_cohort2))














#subset_exprs_only_models
ov_results_cBoost_exprs_models <- ov_results_cBoost[c(1, 3, 5),]
ov_results_coxph_exprs_models <- ov_results_coxph[c(1, 3, 5),]
ov_results_DeepSurv_exprs_models <- ov_results_DeepSurv[c(1, 3, 5),]
ov_results_rsf_exprs_models <- ov_results_rsf[c(1, 3, 5),]


models_expr_based <- c("CatBoost", "PenalizedCox","DeepSurv",   "PriorityLasso", "RandomSurvivalForest")
test_results_cohort1_selected_data_exprs_data_only <- c(ov_results_cBoost[ov_results_cBoost$mean==max(ov_results_cBoost$mean),][,2], 
                                                        ov_results_coxph[ov_results_coxph$mean==max(ov_results_coxph$mean),][,2],
                                                        ov_results_DeepSurv[ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean),][,3],
                                                        "prio_lasso_all_data",
                                                        ov_results_rsf[ov_results_rsf$mean==max(ov_results_rsf$mean),][,2])
test_results_cohort1_exprs_only <- c(ov_results_cBoost$ci_cohort1[which(ov_results_cBoost$mean==max(ov_results_cBoost$mean))],ov_results_coxph$ci_cohort1[which(ov_results_coxph$mean==max(ov_results_coxph$mean))],ov_results_DeepSurv$ci_cohort1[which(ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean))],ov_results_prio$ci_cohort1[which(ov_results_prio$mean==max(ov_results_prio$mean))],ov_results_rsf$ci_cohort1[which(ov_results_rsf$mean==max(ov_results_rsf$mean))])
test_results_cohort2_selected_data_exprs_data_only <- c(ov_results_cBoost[ov_results_cBoost$mean==max(ov_results_cBoost$mean),][,2], 
                                                        ov_results_coxph[ov_results_coxph$mean==max(ov_results_coxph$mean),][,2],
                                                        ov_results_DeepSurv[ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean),][,3],
                                                        "prio_lasso_all_data",
                                                        ov_results_rsf[ov_results_rsf$mean==max(ov_results_rsf$mean),][,2])
test_results_cohort2_exprs_only <- c(ov_results_cBoost$ci_cohort2[which(ov_results_cBoost$mean==max(ov_results_cBoost$mean))],ov_results_coxph$ci_cohort2[which(ov_results_coxph$mean==max(ov_results_coxph$mean))],ov_results_DeepSurv$ci_cohort2[which(ov_results_DeepSurv$mean==max(ov_results_DeepSurv$mean))],ov_results_prio$ci_cohort2[which(ov_results_prio$mean==max(ov_results_prio$mean))],ov_results_rsf$ci_cohort2[which(ov_results_rsf$mean==max(ov_results_rsf$mean))])

# Create Data Frames for all input data sets
best_test_results_both_cohorts <- data.frame(
  Model = rep(models, each = 2), 
  ci = as.vector(rbind(test_results_cohort1, test_results_cohort2)),  
  Gruppe = rep(c("A", "B"), times = length(models)) 
)
best_test_results_cohort_1 <- data.frame(
  Model = rep(models), 
  ci = c(test_results_cohort1)
)

# Create Data Frames for all input data sets
best_test_results_both_cohorts_test_cohort_based <- data.frame(
  Model = rep(models, each = 2), 
  ci = as.vector(rbind(test_results_cohort1_based_on_test, test_results_cohort2_based_on_test)),  
  Gruppe = rep(c("A", "B"), times = length(models)) 
)
best_test_results_cohort_1_test_cohort_based <- data.frame(
  Model = rep(models), 
  ci = c(test_results_cohort1_based_on_test)
)




# Create Data Frames for exprs based input data sets only
best_test_results_both_cohorts_exprs_based <- data.frame(
  Model = rep(models_expr_based, each = 2), 
  ci = as.vector(rbind(test_results_cohort1_exprs_data_only, test_results_cohort2_exprs_data_only)),  
  Gruppe = rep(c("A", "B"), times = length(models_expr_based)) 
)
best_test_results_cohort_1_exprs_based <- data.frame(
  Model = rep(models_expr_based), 
  ci = c(test_results_cohort1_exprs_data_only)
)





## Plots for both cohorts preprocessing
colnames(best_test_results_both_cohorts) <- c("Model", "ci", "Cohort")
colnames(best_test_results_both_cohorts_exprs_based) <- c("Model", "ci", "Cohort")


best_test_results_both_cohorts$Cohort <- ifelse(best_test_results_both_cohorts$Cohort == "A", "10",
                                                ifelse(best_test_results_both_cohorts$Cohort == "B", "11", best_test_results_both_cohorts$Cohort))

best_test_results_both_cohorts_exprs_based$Cohort <- ifelse(best_test_results_both_cohorts_exprs_based$Cohort == "A", "10",
                                                            ifelse(best_test_results_both_cohorts_exprs_based$Cohort == "B", "11", best_test_results_both_cohorts_exprs_based$Cohort))


best_test_results_both_cohorts$ci <- as.numeric(best_test_results_both_cohorts$ci)
best_test_results_both_cohorts_exprs_based$ci <- as.numeric(best_test_results_both_cohorts_exprs_based$ci)




best_test_results_both_cohorts$Model <- factor(best_test_results_both_cohorts$Model, levels = models)


## Plots for both cohorts preprocessing for test data based plots
colnames(best_test_results_both_cohorts_test_cohort_based) <- c("Model", "ci", "Cohort")
colnames(best_test_results_both_cohorts_exprs_based) <- c("Model", "ci", "Cohort")


best_test_results_both_cohorts_test_cohort_based$Cohort <- ifelse(best_test_results_both_cohorts_test_cohort_based$Cohort == "A", "10",
                                                ifelse(best_test_results_both_cohorts_test_cohort_based$Cohort == "B", "11", best_test_results_both_cohorts_test_cohort_based$Cohort))





best_test_results_both_cohorts_test_cohort_based$ci <- as.numeric(best_test_results_both_cohorts_test_cohort_based$ci)





best_test_results_both_cohorts_test_cohort_based$Model <- factor(best_test_results_both_cohorts_test_cohort_based$Model, levels = models)





# expression and pData inputs

ggplot(best_test_results_both_cohorts, aes(y = Model, x = ci, group = Cohort)) +
  
  # Graue Verbindungslinien zur Referenzlinie
  geom_segment(aes(yend = Model), xend = mean_risk, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = ci, 
    xend = ci,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = mean_risk, color = "blue") +
  
  # Punkte für die zwei Werte je Modell
  geom_point(aes(color = Cohort), size = 2) +
  
  # Manuelle Farbzuweisung für die neuen Gruppen
  scale_color_manual(values = c("10" = "turquoise", "11" = "#0c4252")) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.6, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Model Performances with Two Values vs. Risk Score Reference",
    x = "C-Index",
    y = "Model",
    color = "Cohort"  # Legendentitel geändert
  ) +
  theme_minimal()





# Only expression based inputs

best_test_results_both_cohorts_exprs_based$Model <- factor(best_test_results_both_cohorts_exprs_based$Model, levels = models_expr_based)


ggplot(best_test_results_both_cohorts_exprs_based, aes(y = Model, x = ci, group = Cohort)) +
  
  # Graue Verbindungslinien zur Referenzlinie
  geom_segment(aes(yend = Model), xend = mean_risk, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = ci, 
    xend = ci,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = mean_risk, color = "blue") +
  
  # Punkte für die zwei Werte je Modell
  geom_point(aes(color = Cohort), size = 2) +
  
  # Manuelle Farbzuweisung für die neuen Gruppen
  scale_color_manual(values = c("10" = "turquoise", "11" = "#0c4252")) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.6, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Model Performances for Both Test Cohorts vs. Risk Score Reference (Only Expression Based)",
    x = "C-Index",
    y = "Model",
    color = "Cohort"  # Legendentitel geändert
  ) +
  theme_minimal()


# Graphics for just one cohort
# Sicherstellen, dass 'ci' numerisch ist
best_test_results_cohort_1$ci <- as.numeric(best_test_results_cohort_1$ci)
best_test_results_cohort_1_exprs_based$ci <- as.numeric(best_test_results_cohort_1_exprs_based$ci)

# Definieren der Faktorlevels für 'Model'
best_test_results_cohort_1$Model <- factor(best_test_results_cohort_1$Model, levels = models)
best_test_results_cohort_1_exprs_based$Model <- factor(best_test_results_cohort_1_exprs_based$Model, levels = models_expr_based)


# Exprs and pData
ggplot(best_test_results_cohort_1, aes(y = Model, x = ci)) +
  
  # Graue Verbindungslinien zur Referenzlinie
  geom_segment(aes(yend = Model), xend = mean_risk, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = ci, 
    xend = ci,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = mean_risk, color = "blue") +
  
  # Punkte für die Werte je Modell
  geom_point(color = "turquoise", size = 2) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.6, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Model Performances in Cohort 10 vs. Risk Score Reference",
    x = "C-Index",
    y = "Model"
  ) +
  theme_minimal()






# Only exprs based models
ggplot(best_test_results_cohort_1_exprs_based, aes(y = Model, x = ci)) +
  
  # Graue Verbindungslinien zur Referenzlinie
  geom_segment(aes(yend = Model), xend = mean_risk, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = ci, 
    xend = ci,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = mean_risk, color = "blue") +
  
  # Punkte für die Werte je Modell
  geom_point(color = "turquoise", size = 2) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.6, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Model Performances for Cohort 10 vs. Risk Score Reference (Only Expression Based)",
    x = "C-Index",
    y = "Model"
  ) +
  theme_minimal()


#####################################################################
#Model using average of all of our models as reference
# average ovver all model performance on both test cohorts
all_models_performances <- c(ov_results_cBoost$ci_cohort1, ov_results_cBoost$ci_cohort2, 
                             ov_results_coxph$ci_cohort1, ov_results_coxph$ci_cohort2, 
                             ov_results_DeepSurv$ci_cohort1, ov_results_DeepSurv$ci_cohort2, 
                             ov_results_rsf$ci_cohort1, ov_results_rsf$ci_cohort2, 
                             ov_results_coxPas$ci_cohort1, ov_results_coxPas$ci_cohort2, 
                             ov_results_prio$ci_cohort1, ov_results_prio$ci_cohort2)
# average over the performances on both test cohorts of only the best models in each model class
best_models_performances <- c(best_test_results_both_cohorts$ci)


avrg_all_models_performances <- mean(all_models_performances)
avrg_best_models_performances <- mean(best_models_performances)



# Plot with all models avrg reference
# expression and pData inputs

ggplot(best_test_results_both_cohorts, aes(y = Model, x = ci, group = Cohort)) +
  
  # Graue Verbindungslinien zur Referenzlinie
  geom_segment(aes(yend = Model), xend = avrg_all_models_performances, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = ci, 
    xend = ci,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = avrg_all_models_performances, color = "blue") +
  
  # Punkte für die zwei Werte je Modell
  geom_point(aes(color = Cohort), size = 2) +
  
  # Manuelle Farbzuweisung für die neuen Gruppen
  scale_color_manual(values = c("10" = "turquoise", "11" = "#0c4252")) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.6, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Performances of Best Models vs. All Models Benchmark",
    x = "C-Index",
    y = "Model",
    color = "Cohort"  
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),  # Überschrift fett und Schriftgröße 14
    axis.text.x = element_text(angle = 45, hjust = 1)     # Optional: Anpassung der X-Achsen-Beschriftung
  )


# Plot with best models avrg reference

ggplot(best_test_results_both_cohorts, aes(y = Model, x = ci, group = Cohort)) +
  
  # Graue Verbindungslinien zur Referenzlinie
  geom_segment(aes(yend = Model), xend = avrg_best_models_performances, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = ci, 
    xend = ci,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = avrg_best_models_performances, color = "blue") +
  
  # Punkte für die zwei Werte je Modell
  geom_point(aes(color = Cohort), size = 2) +
  
  # Manuelle Farbzuweisung für die neuen Gruppen
  scale_color_manual(values = c("10" = "turquoise", "11" = "#0c4252")) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.6, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Model Performances with Two Values vs. Risk Score Reference",
    x = "C-Index",
    y = "Model",
    color = "Cohort"  # Legendentitel geändert
  ) +
  theme_minimal()






# Plot with all models avrg reference and average performance on test cohorts


best_test_results_both_cohorts_with_avrg <- data.frame()
for (i in seq(1, nrow(best_test_results_both_cohorts_test_cohort_based), by = 2)) {
  # Originalzeilen hinzufügen
  best_test_results_both_cohorts_with_avrg <- rbind(best_test_results_both_cohorts_with_avrg, best_test_results_both_cohorts_test_cohort_based[i, ], best_test_results_both_cohorts_test_cohort_based[i + 1, ])
  
  # Durchschnittszeile berechnen und hinzufügen
  avg_row <- data.frame(
    Model = best_test_results_both_cohorts_test_cohort_based$Model[i],
    ci = mean(c(best_test_results_both_cohorts_test_cohort_based$ci[i], best_test_results_both_cohorts_test_cohort_based$ci[i + 1])),
    Cohort = "Average"
  )
  best_test_results_both_cohorts_with_avrg <- rbind(best_test_results_both_cohorts_with_avrg, avg_row)
}




ggplot(best_test_results_both_cohorts_with_avrg, aes(y = Model, x = ci, group = Cohort)) +
  
  # Graue Verbindungslinien zur Referenzlinie
  geom_segment(aes(yend = Model), xend = avrg_best_models_performances, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = ci, 
    xend = ci,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = avrg_all_models_performances, color = "blue") +
  
  # Punkte für die Werte (inkl. Average)
  geom_point(aes(color = Cohort), size = 2) +
  
  # Manuelle Farbzuweisung, einschließlich Average
  scale_color_manual(values = c("10" = "turquoise", "11" = "#0c4252", "Average" = "orange")) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.6, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Performances of Best Models vs. All Models Benchmark",
    x = "C-Index",
    y = "Model",
    color = "Cohort"  # Legendentitel geändert
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0, size = 14, face = "bold"),  # Titel linksbündig und fett
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )



