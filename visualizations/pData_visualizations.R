library(ggplot2)
library(dplyr)
library(tidyr)
library(reshape2)
library(readr)
library(Biobase)
library(forcats)
#recodinBiobase#recoding for all  data
#recode(
#  all_pData_MONTH_TO_BCR$study,
#  "Atlanta_2014_Long" = "Cohort 1",
#  "Belfast_2018_Jain" = "Cohort 2",
#  "CamCap_2016_Ross_Adams" = "Cohort 3",
#  "CancerMap_2017_Luca" = "Cohort 4",
#  "CPC_GENE_2017_Fraser" = "Cohort 5",
#  "CPGEA_2020_Li" = "Cohort 6",
#  "DKFZ_2018_Gerhauser" = "Cohort 7",
#  "MSKCC_2010_Taylor" = "Cohort 8",
#  "Stockholm_2016_Ross_Adams" = "Cohort 9",
#  "Ribolution_prad" = "Cohort 10",
#  "Ribolution_ukd2" = "Cohort 11"
#)















# Define Train and Test cohorts
train_studies <- c(
  "Atlanta_2014_Long", "Belfast_2018_Jain", "CamCap_2016_Ross_Adams",
  "CancerMap_2017_Luca", "CPC_GENE_2017_Fraser", "CPGEA_2020_Li",
  "DKFZ_2018_Gerhauser", "MSKCC_2010_Taylor", "Stockholm_2016_Ross_Adams"
)
test_studies <- c("Ribolution_prad", "Ribolution_ukd2")

all_studies <- c(train_studies, test_studies)

load_cohorts <- function(rds_file_paths) {
  all_cohorts <- list()
  for (rds_file_path in rds_file_paths) {
    cohorts <- readRDS(file.path(".", "data", rds_file_path))
    cohorts_list <- lapply(cohorts, function(eset) {
      list(
        pData = as.data.frame(pData(eset)),
        fData = as.data.frame(fData(eset)),
        exprs = as.data.frame(exprs(eset))
      )
    })
    all_cohorts <- c(all_cohorts, cohorts_list)
  }
  return(all_cohorts)
}
train_cohorts <- load_cohorts("PCa_cohorts.Rds")

train_cohorts$Stockholm_2016_Ross_Adams$pData$AGE <- 
  ifelse(train_cohorts$Stockholm_2016_Ross_Adams$pData$AGE == "unknown", 
         NA, 
         train_cohorts$Stockholm_2016_Ross_Adams$pData$AGE)

test_cohorts <- load_cohorts( "PCa_cohorts_2.Rds")
train_pData <- data.frame()
test_pData <- data.frame()
for (i in 1:length(train_cohorts)) { 
  train_pData <- rbind(train_pData, train_cohorts[[i]]$pData[,c("AGE", "STUDY", "TISSUE", "GLEASON_SCORE", "PRE_OPERATIVE_PSA", "BCR_STATUS", "MONTH_TO_BCR", "DOD_STATUS", "MONTH_TO_DOD")])
}
for (i in 1:length(test_cohorts)) { 
  test_pData <- rbind(test_pData, test_cohorts[[i]]$pData[,c("AGE", "STUDY", "TISSUE", "GLEASON_SCORE", "PRE_OPERATIVE_PSA", "BCR_STATUS", "MONTH_TO_BCR", "DOD_STATUS", "MONTH_TO_DOD")])
  
}

all_pData_AGE <- rbind(train_pData[, c("AGE", "STUDY")], (test_pData[, c("AGE", "STUDY")]))
train_exprs_merged_imputed <- exprs <- as.data.frame(read_csv('data/merged_data/exprs/common_genes/common_genes_knn_imputed.csv', lazy = TRUE))
train_exprs_merged_intersect<- exprs <- as.data.frame(read_csv('data/merged_data/exprs/intersection/exprs_intersect.csv', lazy = TRUE))
train_exprs_merged_all_genes<- exprs <- as.data.frame(read_csv('data/merged_data/exprs/all_genes/all_genes.csv', lazy = TRUE))


################################################################################################################
#Tissue Bar Plot for train data

library(ggplot2)
library(tidyr)

# Train- und Test-Tissue-Counts erstellen
train_tissue_counts <- table(train_pData$TISSUE)
test_pData$TISSUE[test_pData$TISSUE == "fresh-frozen"] <- "Fresh_frozen"
train_tissue_counts <- table(train_pData$TISSUE)
test_tissue_counts <- table(test_pData$TISSUE)

all_tissue_types <- union(names(train_tissue_counts), names(test_tissue_counts))
train_tissue_counts <- train_tissue_counts[all_tissue_types]
test_tissue_counts <- test_tissue_counts[all_tissue_types]

# NA-Werte durch 0 ersetzen
train_tissue_counts[is.na(train_tissue_counts)] <- 0
test_tissue_counts[is.na(test_tissue_counts)] <- 0

# Daten in ein DataFrame umwandeln
stacked_counts <- rbind(train_tissue_counts, test_tissue_counts)
rownames(stacked_counts) <- c("Group A", "Group B")

# Konvertierung in ein DataFrame für ggplot2
df <- as.data.frame(t(stacked_counts))
df$Tissue_Type <- rownames(df)
df_long <- pivot_longer(df, -Tissue_Type, names_to = "Group", values_to = "Count")

ggplot(df_long, aes(x = Tissue_Type, y = Count, fill = Group)) +
  geom_bar(stat = "identity", position = position_stack(reverse = TRUE)) +  # Gelben Balken (Group A) nach unten verschieben
  scale_fill_manual(values = c("Group A" = "#ffcd66", "Group B" = "#00706d")) +
  labs(
    title = "Number of patients by tissue type",
    x = "Tissue Type",
    y = "Number of patients",
    fill = "Group"
  ) +
  scale_x_discrete(labels = c("FFPE" = "FFPE", "Fresh_frozen" = "Fresh-Frozen", "Snap_frozen" = "Snap-Frozen")) +  # X-Achsen-Beschriftungen ändern
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),  # Überschrift fett, Größe 14, linksbündig
    axis.text.x = element_text(angle = 45, hjust = 1)               # X-Achse rotieren
  )



################################################################################################################
#Boxplot for age of train and test



# Mapping der STUDY-Namen zu Kohortennamen
rename_map <- c(
  "Atlanta_2014_Long" = "Cohort 1",
  "Belfast_2018_Jain" = "Cohort 2",
  "CamCap_2016_Ross_Adams" = "Cohort 3",
  "CancerMap_2017_Luca" = "Cohort 4",
  "CPC_GENE_2017_Fraser" = "Cohort 5",
  "CPGEA_2020_Li" = "Cohort 6",
  "DKFZ_2018_Gerhauser" = "Cohort 7",
  "MSKCC_2010_Taylor" = "Cohort 8",
  "Stockholm_2016_Ross_Adams" = "Cohort 9",
  "Ribolution_prad" = "Cohort 10",
  "Ribolution_ukd2" = "Cohort 11"
)

# Definieren der gewünschten Reihenfolge der Kohorten
desired_order <- rename_map  # 'Cohort 1', 'Cohort 2', etc.

# Sicherstellen, dass AGE numerisch ist und Hinzufügen der COHORT_TYPE-Spalte
train_pData_AGE <- train_pData %>%
  mutate(
    AGE = as.numeric(AGE),
    COHORT_TYPE = "Group 1"  # Trainingsdaten markieren
  )

test_pData_AGE <- test_pData %>%
  mutate(
    AGE = as.numeric(AGE),
    COHORT_TYPE = "Group 2"  # Testdaten markieren
  )

# Kombinieren von Trainings- und Testdaten mit bind_rows
all_pData_AGE <- bind_rows(train_pData_AGE[, c(1, 2, 10)], test_pData_AGE[, c(1, 2, 10)])

# Verarbeitung der kombinierten Daten
all_pData_AGE <- all_pData_AGE %>%
  mutate(
    STUDY = rename_map[STUDY]  # Umbenennen der STUDY-Namen zu Kohortennamen
  ) %>%
  mutate(
    STUDY = factor(STUDY, levels = desired_order)  # Festlegen der Faktorstufen entsprechend der gewünschten Reihenfolge
  ) %>%
  complete(
    STUDY = factor(desired_order, levels = desired_order),  # Sicherstellen, dass alle Kohortenstufen vorhanden sind
    fill = list(AGE = NA, COHORT_TYPE = "NA")  # Fehlende Werte auffüllen
  ) %>%
  mutate(
    COHORT_TYPE = factor(COHORT_TYPE, levels = c("Group 1", "Group 2", "NA"))  # Sicherstellen, dass 'COHORT_TYPE' als Faktor definiert ist
  )

# Überprüfen der Struktur des Datensatzes (optional, aber empfohlen)
str(all_pData_AGE)
head(all_pData_AGE)

# Erstellung des Boxplots
ggplot(all_pData_AGE, aes(x = STUDY, y = AGE, fill = COHORT_TYPE)) +
  geom_boxplot(outlier.shape = NA, na.rm = TRUE) +  # Boxplots ohne Outlier
  geom_jitter(data = all_pData_AGE %>% filter(!is.na(AGE)),  # Punkte nur für nicht-NA-Werte
              aes(color = COHORT_TYPE), width = 0.2, alpha = 0.6) +
  scale_fill_manual(
    values = c("Group 1" = "#ffcd66", "Group 2" = "#00706d", "NA" = "white"),
    na.value = "white"  # Farbe für NA-Werte
  ) + 
  scale_color_manual(
    values = c("Group 1" = "yellow", "Group 2" = "#0c4252", "NA" = "grey")
  ) + 
  theme_minimal() +
  labs(
    title = "Age Distribution by Cohort",
    x = "Cohort",
    y = "Age",
    fill = "Group",
    color = "Group"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),  # Überschrift fett und Größe 14
    axis.text.x = element_text(angle = 45, hjust = 1)     # X-Achse-Beschriftung rotieren
  )



################################################################################################################
# 1) Rohdaten extrahieren
train_scores <- na.omit(train_pData$GLEASON_SCORE)
test_scores  <- na.omit(test_pData$GLEASON_SCORE)

# 2) Gemeinsames Data-Frame erstellen
gleason_df <- data.frame(
  Score = c(train_scores, test_scores),
  Group = c(rep("Train", length(train_scores)),
            rep("Test",  length(test_scores)))
)

# 3) Box-Plot erstellen
library(ggplot2)

ggplot(gleason_df, aes(x = Group, y = Score, fill = Group)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Train" = "skyblue", "Test" = "orange")) +
  labs(
    title = "Verteilung der Gleason Scores nach Train- und Test-Datensatz",
    x = "Gruppe",
    y = "Gleason Score"
  ) +
  theme_minimal()




###################################################
# Grouped Bar Plot


# 2) Rohdaten extrahieren und Data-Frame erstellen
train_scores <- na.omit(train_pData$GLEASON_SCORE)
test_scores  <- na.omit(test_pData$GLEASON_SCORE)

gleason_df <- data.frame(
  Score = c(train_scores, test_scores),
  Group = c(rep("Group A", length(train_scores)),  # "Train" zu "Group A" geändert
            rep("Group B", length(test_scores)))    # "Test" zu "Group B" geändert
)

# 3) Berechnung der prozentualen Anteile
gleason_percent <- gleason_df %>%
  group_by(Group, Score) %>%
  summarise(Count = n(), .groups = 'drop') %>%
  group_by(Group) %>%
  mutate(Percentage = (Count / sum(Count)) * 100) %>%
  ungroup()

# Optional: Sicherstellen, dass alle Kombinationen vorhanden sind (für vollständige Darstellung)
gleason_percent <- gleason_percent %>%
  complete(Group, Score, fill = list(Count = 0, Percentage = 0))

# 4) Erstellung des gruppierten Balkendiagramms mit der Legende rechts
ggplot(gleason_percent, aes(x = as.factor(Score), y = Percentage, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  scale_fill_manual(
    values = c("Group A" = "#ffcd66", "Group B" = "#00706d")  # Farben angepasst
  ) +
  labs(
    title = "Gleason Score Distribution by Group in %",
    x = "Gleason Score",
    y = "% ",
    fill = "Gruppe"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title = element_text(size = 12),
    plot.title = element_text(hjust = 0, size = 14, face = "bold"),  # Überschrift linksbündig (hjust = 0)
    legend.position = "right",  # Legende nach rechts verschieben
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  ) +
  geom_text(
    aes(label = sprintf("%.1f%%", Percentage)),
    position = position_dodge(width = 0.8),
    vjust = -0.5,
    size = 3,
    color = "black"  # Farbe der Textbeschriftungen für bessere Lesbarkeit
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))  # Raum für Text über den Balken




###################################################
#andere darstellung wegen dem quantilproblem

# 1) Rohdaten extrahieren
train_scores <- na.omit(train_pData$GLEASON_SCORE)
test_scores  <- na.omit(test_pData$GLEASON_SCORE)

# 2) Gemeinsames Data-Frame erstellen
gleason_df <- data.frame(
  Score = c(train_scores, test_scores),
  Group = c(rep("Train", length(train_scores)),
            rep("Test",  length(test_scores)))
)

df_counts <- df_counts %>%
  mutate(Group = if_else(Group == "Train", "Group 1", "Group 2"))

gleason_df <- gleason_df %>%
  mutate(Group = if_else(Group == "Train", "Group 1", "Group 2"))


ggplot(df_counts, aes(x = factor(Score), y = side * prop_perc, fill = Group)) +
  geom_col(position = "identity", width = 0.8) +
  coord_flip() +
  scale_y_continuous(
    limits = c(-75, 75),  # Achsenskala von -50% bis 50%
    breaks = seq(-75, 75, by = 10),  # Intervalle alle 10%
    labels = function(x) paste0(abs(x), "%"),  # Prozentanzeige
    expand = c(0, 0)  # Kein zusätzlicher Rand
  ) +
  scale_fill_manual(
    values = c("Group 1" = "skyblue", "Group 2" = "orange")
  ) +
  labs(
    x = "Gleason Score",
    y = "% of Score Values",
    title = "Gleason Scores Distribution (Group 1 vs. Group 2)"
  ) +
  theme_minimal()


################################################################################################################
#PSA Box Plot


# Sicherstellen, dass PRE_OPERATIVE_PSA numerisch ist
train_pData$PRE_OPERATIVE_PSA <- as.numeric(train_pData$PRE_OPERATIVE_PSA)
test_pData$PRE_OPERATIVE_PSA <- as.numeric(test_pData$PRE_OPERATIVE_PSA)

# Kombinieren der Trainings- und Testdaten mit einer neuen Spalte "Group"
psa_data <- bind_rows(
  train_pData %>%
    select(PRE_OPERATIVE_PSA, STUDY) %>%
    mutate(Group = "Train"),
  test_pData %>%
    select(PRE_OPERATIVE_PSA, STUDY) %>%
    mutate(Group = "Test")
)



# Definieren der Mapping-Tabelle von STUDY zu Cohort
rename_map <- c(
  "Atlanta_2014_Long" = "Cohort 1",
  "Belfast_2018_Jain" = "Cohort 2",
  "CamCap_2016_Ross_Adams" = "Cohort 3",
  "CancerMap_2017_Luca" = "Cohort 4",
  "CPC_GENE_2017_Fraser" = "Cohort 5",
  "CPGEA_2020_Li" = "Cohort 6",
  "DKFZ_2018_Gerhauser" = "Cohort 7",
  "MSKCC_2010_Taylor" = "Cohort 8",
  "Stockholm_2016_Ross_Adams" = "Cohort 9",
  "Ribolution_prad" = "Cohort 10",
  "Ribolution_ukd2" = "Cohort 11"
)

# Anwenden des Mappings und Umwandeln in einen geordneten Faktor
psa_data <- psa_data %>%
  mutate(
    Cohort = recode(STUDY, !!!rename_map),  # Mapping von STUDY zu Cohort
    Cohort = factor(Cohort, levels = paste0("Cohort ", 1:11))  # Reihenfolge festlegen
  ) %>%
  # Reihenfolge umkehren, damit Cohort 1 oben ist
  mutate(
    Cohort = fct_rev(Cohort)
  )



# Definieren der Farbgruppen
psa_data <- psa_data %>%
  mutate(
    ColorGroup = ifelse(Cohort %in% paste0("Cohort ", 10:11), "Group B", "Group A")
  )

# Definieren der Farbzuweisung für die beiden Gruppen
fill_colors <- c("Group A" = "#ffcd66", "Group B" = "#00706d")
color_colors <- c("Group A" = "orange", "Group B" = "#0c4252")

# Erstellen des Boxplots für alle Daten mit angepasster Legende
ggplot(psa_data, aes(y = Cohort, x = PRE_OPERATIVE_PSA, fill = ColorGroup)) +
  geom_boxplot(outlier.shape = NA) + 
  geom_jitter(aes(color = ColorGroup), width = 0.2, alpha = 0.5) +  
  scale_fill_manual(values = fill_colors, name = "Group") + 
  scale_color_manual(values = color_colors, name = "Group") + 
  scale_x_log10() +  
  labs(
    title = "Distribution of PSA Values by Cohort",
    x = "PSA Value (Log Scale)",
    y = "Cohort"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(angle = 0),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )



# Erstellen des Boxplots mit eingeschränktem X-Bereich (0 bis 100) und angepasster Legende
ggplot(psa_data, aes(y = Cohort, x = PRE_OPERATIVE_PSA, fill = ColorGroup)) +
  geom_boxplot(outlier.shape = NA) +  
  geom_jitter(aes(color = ColorGroup), width = 0.2, alpha = 0.5) + 
  scale_fill_manual(values = fill_colors, name = "Group") + 
  scale_color_manual(values = color_colors, name = "Group") + 
  labs(
    title = "Distribution of PSA Values by Cohort",
    x = "PSA Value",
    y = "Cohort"
  ) +
  coord_cartesian(xlim = c(0, 100)) +  
  theme_minimal() +
  theme(
    axis.text.y = element_text(angle = 0),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )




ggplot(psa_data, aes(y = Cohort, x = PRE_OPERATIVE_PSA, fill = ColorGroup)) +
  geom_boxplot(outlier.shape = NA) +  
  geom_jitter(aes(color = ColorGroup), width = 0.2, alpha = 0.5) + 
  scale_fill_manual(values = fill_colors, name = "Group") + 
  scale_color_manual(values = color_colors, name = "Group") + 
  scale_x_log10(
    limits = c(1, 2000),  # Setzen der Grenzen (1 statt 0)
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::label_number()  # Standardzahlenformat ohne Exponenten
  ) +  
  labs(
    title = "Distribution of PSA Values by Cohort",
    x = "PSA Value (Log Scale)",
    y = "Cohort"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),  # Titel fett und Größe 14
    axis.text.y = element_text(angle = 0),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )






################################################################################################################
#Bar Chart Expressions




# -------------------------------------------------
# Beispielliste, wie Du exprs_num_df aus Deinen Daten erstellst
# (An diesem Teil ändert sich nichts Grundlegendes)
# -------------------------------------------------
exprs_num_df <- data.frame(
  `Exprs Source` = character(), 
  `Exprs Count` = numeric()     
)

for (i in 1:length(train_cohorts)) {
  study_name <- train_cohorts[[i]]$pData$STUDY[1]  
  gene_count <- nrow(train_cohorts[[i]]$exprs)     
  
  if (!study_name %in% exprs_num_df$`Exprs Source`) {
    exprs_num_df <- rbind(
      exprs_num_df,
      data.frame(
        `Exprs Source` = study_name,
        `Exprs Count` = gene_count
      )
    )
  }
}

TCGA_PRAD_genes <- rownames(test_cohorts$TCGA_PRAD$exprs)
UKD2_genes      <- rownames(test_cohorts$UKD2$exprs)
all_test_genes  <- unique(c(TCGA_PRAD_genes, UKD2_genes))
all_train_and_test_genes <- unique(c(all_test_genes,
                                     colnames(train_exprs_merged_all_genes)))

exprs_num_df <- rbind(
  exprs_num_df,
  data.frame(`Exprs Source` = "TCGA_PRAD_genes", 
             `Exprs Count` = nrow(test_cohorts$TCGA_PRAD$exprs)),
  data.frame(`Exprs Source` = "UKD2", 
             `Exprs Count` = nrow(test_cohorts$UKD2$exprs))
)

exprs_num_df <- exprs_num_df %>% 
  arrange(desc(`Exprs.Count`))

exprs_num_df <- rbind(
  exprs_num_df,
  data.frame(`Exprs Source` = "Common Genes", 
             `Exprs Count` = ncol(train_exprs_merged_imputed)),
  data.frame(`Exprs Source` = "Intersection", 
             `Exprs Count` = ncol(train_exprs_merged_intersect)),
  data.frame(`Exprs Source` = "All Training Genes", 
             `Exprs Count` = ncol(train_exprs_merged_all_genes)),
  data.frame(`Exprs Source` = "All Test Genes", 
             `Exprs Count` = length(all_test_genes)),
  data.frame(`Exprs Source` = "All Genes", 
             `Exprs Count` = length(all_train_and_test_genes))
)
plot1
# -------------------------------------------------
# Farbenzuordnung / Mappen der Namen
# -------------------------------------------------
exprs_num_df <- exprs_num_df %>%
  mutate(
    Color_Group = case_when(
      row_number() >= 12 ~ "Gray",   # z.B. die Kombinations-Items
      row_number() ==  6 ~ "Orange", # 
      row_number() ==  1 ~ "Orange", #
      TRUE ~ "Blue"
    )
  )

color_map <- c("Blue"   = "skyblue",
               "Orange" = "orange",
               "Gray"   = "gray")

legend_labels <- c("Blue"   = "Group 1",
                   "Orange" = "Group 2 ",
                   "Gray"   = "Combinations")

exprs_num_df <- exprs_num_df %>%
  mutate(
    Exprs.Source = case_when(
      Exprs.Source == "Atlanta_2014_Long"          ~ "Cohort 1",
      Exprs.Source == "Belfast_2018_Jain"          ~ "Cohort 2",
      Exprs.Source == "CamCap_2016_Ross_Adams"     ~ "Cohort 3",
      Exprs.Source == "CancerMap_2017_Luca"        ~ "Cohort 4",
      Exprs.Source == "CPC_GENE_2017_Fraser"       ~ "Cohort 5",
      Exprs.Source == "CPGEA_2020_Li"              ~ "Cohort 6",
      Exprs.Source == "DKFZ_2018_Gerhauser"        ~ "Cohort 7",
      Exprs.Source == "MSKCC_2010_Taylor"          ~ "Cohort 8",
      Exprs.Source == "Stockholm_2016_Ross_Adams"  ~ "Cohort 9",
      Exprs.Source == "TCGA_PRAD_genes"            ~ "Cohort 10",
      Exprs.Source == "UKD2"                       ~ "Cohort 11",
      TRUE ~ as.character(Exprs.Source)
    )
  )

# -------------------------------------------------
# --- Sortierung und Faktorlevels ---
# -------------------------------------------------
# Schritt 1: Eigene Sortierspalte anlegen,
#            Orange = ganz oben (1), Blue = 2, Gray = 3
exprs_num_df <- exprs_num_df %>%
  mutate(
    SortGroup = case_when(
      Color_Group == "Orange" ~ 1,  # soll ganz oben erscheinen
      Color_Group == "Blue"   ~ 2,
      Color_Group == "Gray"   ~ 3,
      TRUE ~ 99
    )
  ) %>%
  # Zuerst Orange, dann Blue, dann Gray – 
  # innerhalb jeder Gruppe nach Count absteigend
  arrange(SortGroup, desc(Exprs.Count))

# Schritt 2: Den Faktor in **umgekehrter** Reihenfolge definieren,
#            sodass die ersten Zeilen im Dataframe (Orange) oben im Plot stehen.
exprs_num_df$Exprs.Source <- factor(
  exprs_num_df$Exprs.Source, 
  levels = rev(exprs_num_df$Exprs.Source)
)

# -------------------------------------------------
# Plot
# -------------------------------------------------
ggplot(exprs_num_df, aes(x = Exprs.Source, y = Exprs.Count, fill = Color_Group)) +
  geom_bar(stat = "identity", width = 0.8) +
  geom_text(aes(label = Exprs.Count), hjust = -0.2, size = 3) +
  scale_fill_manual(values = color_map, labels = legend_labels) +
  scale_y_continuous(limits = c(0, 65000)) +
  coord_flip() +
  labs(
    title = "Number of Genes by Source",
    x = "Source",
    y = "Gene Count",
    fill = "Group"
  ) +
  theme_minimal() +
  theme(
    axis.text.y  = element_text(size = 10),
    axis.text.x  = element_text(size = 10),
    plot.title   = element_text(hjust = 0.5,size = 14, face = "bold")
  )

###########################
#Nur Cohorten Genes


library(dplyr)
library(ggplot2)

# a. Erstellen des Sub-Datenframes für Kohorten 1-11
exprs_num_df_cohorts <- exprs_num_df %>%
  filter(Exprs.Source %in% paste0("Cohort ", 1:11)) %>%
  arrange(SortGroup, desc(`Exprs.Count`))

# b. Verschiebe die ersten zwei Zeilen ans Ende
exprs_num_df_cohorts <- exprs_num_df_cohorts %>%
  slice(-c(1,2)) %>%
  bind_rows(exprs_num_df_cohorts %>% slice(1:2))

# c. Anpassen der Faktorlevels nach der Neuanordnung
exprs_num_df_cohorts$Exprs.Source <- factor(
  exprs_num_df_cohorts$Exprs.Source, 
  levels = rev(exprs_num_df_cohorts$Exprs.Source)  # Umgekehrte Reihenfolge für coord_flip()
)

ggplot(exprs_num_df_cohorts, aes(x = Exprs.Source, y = `Exprs.Count`, fill = Color_Group)) +
  geom_bar(stat = "identity", width = 0.8) +
  geom_text(aes(label = `Exprs.Count`), hjust = -0.2, size = 3) +
  scale_fill_manual(
    values = c("Orange" = "#00706d", "Blue" = "#ffcd66"),
    labels = c("Orange" = "Group B", "Blue" = "Group A")
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  coord_flip() +
  labs(
    title = "Number of Genes by Cohorts",
    x = "Cohort",
    y = "Gene Count",
    fill = "Group"
  ) +
  theme_minimal() +
  theme(
    axis.text.y  = element_text(size = 10),
    axis.text.x  = element_text(size = 10),
    plot.title   = element_text(hjust = 0, size = 14, face = "bold"),  # Titel linksbündig (hjust = 0)
    legend.position = "right",  # Legende nach rechts verschieben
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )

# -------------------------------------------------
# 2. Plot für "Common Genes", "Intersection" und "All Genes"
# -------------------------------------------------

# a. Erstellen des Sub-Datenframes für Kombinationen
exprs_num_df_combinations <- exprs_num_df %>%
  filter(Exprs.Source %in% c("Common Genes", "Intersection", "All Genes")) %>%
  arrange(SortGroup, desc(`Exprs.Count`))

# b. Anpassen der Faktorlevels für die Reihenfolge im Plot
exprs_num_df_combinations$Exprs.Source <- factor(
  exprs_num_df_combinations$Exprs.Source, 
  levels = rev(exprs_num_df_combinations$Exprs.Source)  # Umgekehrte Reihenfolge für coord_flip()
)

# c. Anpassen der Farbzuordnung für Kombinationen
exprs_num_df_combinations <- exprs_num_df_combinations %>%
  mutate(
    Color_Group = case_when(
      Exprs.Source == "Common Genes"  ~ "darkred",
      Exprs.Source == "Intersection"   ~ "#0c4252",
      Exprs.Source == "All Genes"      ~ "grey",
      TRUE                             ~ "gray"  # Für Sicherheit
    )
  )

color_map_combinations <- c(
  "darkred" = "darkred", 
  "#0c4252" = "#0c4252", 
  "grey" = "grey"
)

legend_labels_combinations <- c(
  "darkred" = "Common Genes",
  "#0c4252" = "Intersection",
  "grey" = "All Genes"
)
# Reihenfolge der Levels in Color_Group anpassen
exprs_num_df_combinations$Color_Group <- factor(
  exprs_num_df_combinations$Color_Group, 
  levels = c("grey", "darkred", "#0c4252")  # Neue Reihenfolge: "All Genes", "Common Genes", "Intersection"
)

# Aktualisierter Plot
ggplot(exprs_num_df_combinations, aes(x = Exprs.Source, y = `Exprs.Count`, fill = Color_Group)) +
  geom_bar(stat = "identity", width = 0.8) +
  geom_text(aes(label = `Exprs.Count`), hjust = -0.2, size = 3) +
  scale_fill_manual(
    values = c("darkred" = "darkred", "#0c4252" = "#0c4252", "grey" = "grey"),
    labels = c("grey" = "All Genes", "darkred" = "Common Genes", "#0c4252" = "Intersection")
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  coord_flip() +
  labs(
    title = "Number of Genes by Combination Type",
    x = "",
    y = "Gene Count",
    fill = "Combination Types"
  ) +
  theme_minimal() +
  theme(
    axis.text.y  = element_text(size = 10),
    axis.text.x  = element_text(size = 10),
    plot.title   = element_text(hjust = 0, size = 14, face = "bold"),  # Titel linksbündig
    legend.position = "right",  # Legende nach rechts verschieben
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )



################################################################################################################
#Intersection of test data genes with common and train intersect genes
common_genes <- colnames(train_exprs_merged_imputed)
intersection <- colnames(train_exprs_merged_intersect)
TCGA_PRAD_genes_in_common <- intersect(TCGA_PRAD_genes, common_genes)
UKD2_genes_in_common <- intersect(UKD2_genes, common_genes)



# Einzigartige Gen-Namen aus allen Vektoren sammeln
all_genes <- unique(c(common_genes, intersection, TCGA_PRAD_genes_in_common, UKD2_genes_in_common))

# Alle Gruppen (y-Achse)
groups <- c("Training Common Genes", "Training Genes Intersection", "TCGA_PRAD", "UKD2")
# Leere Matrix erstellen (Zeilen: Gruppen, Spalten: Alle Gene)
overlap_matrix <- matrix(0, nrow = length(groups), ncol = length(all_genes))
rownames(overlap_matrix) <- groups
colnames(overlap_matrix) <- all_genes

# Matrix mit 1 befüllen, wenn ein Gen in der Gruppe vorkommt
overlap_matrix["Training Common Genes", all_genes %in% common_genes] <- 1
overlap_matrix["Training Genes Intersection", all_genes %in% intersection] <- 1
overlap_matrix["TCGA_PRAD", all_genes %in% TCGA_PRAD_genes_in_common] <- 1
overlap_matrix["UKD2", all_genes %in% UKD2_genes_in_common] <- 1





# Matrix in Data Frame umwandeln
heatmap_data <- melt(overlap_matrix)

# Heatmap plotten
ggplot(heatmap_data, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +  # Farben definieren
  labs(
    title = "Gene Overlap Heatmap",
    x = "Genes",
    y = "Groups",
    fill = "Presence"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12)
  )






# Gruppieren nach Überlappung
overlap_summary <- colSums(overlap_matrix)
overlap_summary <- data.frame(
  Genes = names(overlap_summary),
  OverlapCount = overlap_summary
)



# Gene in funktionelle Gruppen aufteilen (Beispiel: 100-Gene-Gruppen)
gene_groups <- cut(seq_along(all_genes), breaks = 100, labels = FALSE)
overlap_grouped <- matrix(0, nrow = nrow(overlap_matrix), ncol = max(gene_groups))
rownames(overlap_grouped) <- rownames(overlap_matrix)
for (i in seq_len(ncol(overlap_matrix))) {
  overlap_grouped[, gene_groups[i]] <- overlap_grouped[, gene_groups[i]] + overlap_matrix[, i]
}

# Heatmap der zusammengefassten Daten
grouped_heatmap_data <- melt(overlap_grouped)
ggplot(grouped_heatmap_data, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(
    title = "Gene Overlap Heatmap (Grouped)",
    x = "Gene Groups",
    y = "Groups",
    fill = "Presence"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12)
  )



library(plotly)

plot_ly(
  z = overlap_matrix,
  x = colnames(overlap_matrix),
  y = rownames(overlap_matrix),
  type = "heatmap",
  colors = c("white", "blue")
) %>%
  layout(
    xaxis = list(
      title = "Genes", # Titel der x-Achse
      tickvals = NULL, # Keine Ticks anzeigen
      showticklabels = FALSE # Keine Labels anzeigen
    ),
    yaxis = list(
      title = ""
    )
  )





################################################################################################################
#Box Plot month to BCR

# Vorbereitung der Train- und Test-Daten
train_pData_MONTH_TO_BCR <- train_pData[, c("MONTH_TO_BCR", "STUDY", "MONTH_TO_DOD")]
test_pData_MONTH_TO_BCR <- test_pData[, c("MONTH_TO_BCR", "STUDY", "MONTH_TO_DOD")]

# Spezielle Behandlung für Ribolution_ukd2
test_pData_MONTH_TO_BCR[test_pData_MONTH_TO_BCR$STUDY == "Ribolution_ukd2", ]$MONTH_TO_BCR <- 
  test_pData_MONTH_TO_BCR[test_pData_MONTH_TO_BCR$STUDY == "Ribolution_ukd2", ]$MONTH_TO_DOD / 100

# Gruppieren und auswählen
train_pData_MONTH_TO_BCR <- train_pData_MONTH_TO_BCR %>%
  select(MONTH_TO_BCR, STUDY) %>%
  mutate(Group = "Group 1")

test_pData_MONTH_TO_BCR <- test_pData_MONTH_TO_BCR %>%
  select(MONTH_TO_BCR, STUDY) %>%
  mutate(Group = "Group 2")

# Kohorten umbenennen
rename_cohorts <- function(df) {
  df %>%
    mutate(
      STUDY = case_when(
        STUDY == "Atlanta_2014_Long" ~ "Cohort 1",
        STUDY == "Belfast_2018_Jain" ~ "Cohort 2",
        STUDY == "CamCap_2016_Ross_Adams" ~ "Cohort 3",
        STUDY == "CancerMap_2017_Luca" ~ "Cohort 4",
        STUDY == "CPC_GENE_2017_Fraser" ~ "Cohort 5",
        STUDY == "CPGEA_2020_Li" ~ "Cohort 6",
        STUDY == "DKFZ_2018_Gerhauser" ~ "Cohort 7",
        STUDY == "MSKCC_2010_Taylor" ~ "Cohort 8",
        STUDY == "Stockholm_2016_Ross_Adams" ~ "Cohort 9",
        STUDY == "Ribolution_prad" ~ "Cohort 10",
        STUDY == "Ribolution_ukd2" ~ "Cohort 11 (Month to DOD|x100)",
        TRUE ~ STUDY
      )
    )
}

# Kohorten in beiden Datensätzen umbenennen
train_pData_MONTH_TO_BCR <- rename_cohorts(train_pData_MONTH_TO_BCR)
test_pData_MONTH_TO_BCR <- rename_cohorts(test_pData_MONTH_TO_BCR)

# Daten kombinieren
bcr_data <- rbind(train_pData_MONTH_TO_BCR, test_pData_MONTH_TO_BCR)

# Reihenfolge der Kohorten umkehren
bcr_data <- bcr_data %>%
  mutate(
    STUDY = factor(
      STUDY,
      levels = rev(c(
        "Cohort 1",
        "Cohort 2",
        "Cohort 3",
        "Cohort 4",
        "Cohort 5",
        "Cohort 6",
        "Cohort 7",
        "Cohort 8",
        "Cohort 9",
        "Cohort 10",
        "Cohort 11 (Month to DOD|x100)"
      ))
    )
  )


# Boxplot mit begrenzter X-Skala
ggplot(bcr_data, aes(y = STUDY, x = MONTH_TO_BCR, fill = Group)) +
  geom_boxplot(outlier.shape = NA) +  
  geom_jitter(aes(color = Group), width = 0.2, alpha = 0.5) + 
  scale_fill_manual(values = c("Group 1" = "purple", "Group 2" = "orange")) + 
  scale_color_manual(values = c("Group 1" = "darkorchid", "Group 2" = "darkorange")) + 
  labs(
    title = "Distribution of Month to BCR by Cohort and Group",
    x = "Months to BCR",
    y = "Cohort",
    fill = "Group",
    color = "Group"
  ) +
  coord_cartesian(xlim = c(0, 180)) +  
  theme_minimal() +
  theme(axis.text.y = element_text(angle = 0))


##############################
# Aggregated for the groups

# Reihenfolge der Gruppen explizit festlegen
group_data <- group_data %>%
  mutate(Group = factor(Group, levels = c("Group 2", "Group 1")))


# Horizontaler Boxplot erstellen
ggplot(group_data, aes(y = Group, x = MONTH_TO_BCR, fill = Group)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(aes(color = Group), width = 0.2, alpha = 0.5) +
  scale_fill_manual(values = c("Group 1" = "skyblue", "Group 2" = "orange")) +
  scale_color_manual(values = c("Group 1" = "blue", "Group 2" = "darkorange")) +
  labs(
    title = "Distribution of Month to BCR by Group",
    x = "Months to BCR",
    y = "Group",
    fill = "Group",
    color = "Group"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(angle = 0))


################################################################################################################
# Performance Box Plot


# 1. Dummy Daata
deepsurv                <- runif(9, min = 0.6, max = 0.8)
random_survival_forest <- runif(9, min = 0.6, max = 0.8)
catboost               <- runif(9, min = 0.6, max = 0.8)
pasnet                 <- runif(9, min = 0.6, max = 0.8)
priority_lasso         <- runif(9, min = 0.6, max = 0.8)
penalized_cox          <- runif(9, min = 0.6, max = 0.8)
risk_scores            <- c(0.6872, 0.6985, 0.7829, 0.6889, 0.6953, 0.6503, 0.75, 0.6834, 0.6751)

# 2. Create Data frame
model_df <- data.frame(
  DeepSurv                = deepsurv,
  RandomSurvivalForest    = random_survival_forest,
  CatBoost                = catboost,
  PasNet                  = pasnet,
  PriorityLasso           = priority_lasso,
  PenalizedCox            = penalized_cox,
  RiskScores              = risk_scores,
  row.names = paste0("Cohort ", 1:9)
)

# 3) In langes Format (tidy) bringen
model_long <- model_df %>%
  # row.names als eigene Spalte 'Cohort' sichern (damit man sie später ggf. nutzen kann)
  tibble::rownames_to_column(var = "Cohort") %>%
  pivot_longer(
    cols      = -Cohort,
    names_to  = "Model",
    values_to = "Value"
  )

# 4) Vertikaler Boxplot mit custom-Farben
median_risk <- median(subset(model_long, Model == "RiskScores")$Value)

#    RiskScores wird orange, alle anderen Modelle blau
ggplot(model_long, aes(x = Model, y = Value, fill = Model)) +
  geom_boxplot() +
  scale_fill_manual(values = c(
    "DeepSurv"             = "blue",
    "RandomSurvivalForest" = "blue",
    "CatBoost"             = "blue",
    "PasNet"               = "blue",
    "PriorityLasso"        = "blue",
    "PenalizedCox"         = "blue",
    "RiskScores"           = "orange"
  )) +
  # <-- Hier die horizontale Linie mit geom_hline
  geom_hline(
    yintercept = median_risk,
    color = "darkgrey",       # Farbe für die Linie
    linetype = "dashed", # Linientyp
    size = 0.5             # Linienstärke
  ) +
  theme_minimal() +
  labs(
    title = "Vertikaler Boxplot: Score-Farbe vs. Andere",
    x = "Model",
    y = "Wert (z.B. Score)"
  )


################################################################################################################
#Dot Plot with reference for Models vs score baseline

# Referenz-Linie
mean_risk   <- 0.684

# 1) Dummy-Werte für 6 Modelle
models <- c("DeepSurv", 
            "RandomSurvivalForest", 
            "CatBoost", 
            "PasNet", 
            "PriorityLasso", 
            "PenalizedCox")

set.seed(123) # Für Reproduzierbarkeit
values <- runif(length(models), min = 0.70, max = 0.75)

# 2) Data Frame erstellen
df <- data.frame(Model = models, Wert = values)

# 3) Faktor-Level (Reihenfolge) festlegen
df$Model <- factor(df$Model, levels = models)

# 4) Dot-Plot mit Referenzlinie
ggplot(df, aes(y = Model, x = Wert)) +
  
  # (a) Graue Verbindungslinie von jedem Modell-Punkt zur blauen Referenz
  geom_segment(
    aes(yend = Model),     # "Model" aus df
    xend     = mean_risk,  # "mean_risk" als konstante Variable
    linetype = "solid", 
    color    = "gray"
  ) +
  
  # (b) Kurze Vertikallinie am Modell-Wert
  geom_segment(aes(
    x    = Wert, 
    xend = Wert,
    y    = as.numeric(Model) - 0.1,  
    yend = as.numeric(Model) + 0.1
  ),
  linetype = "solid", 
  color    = "gray") +
  
  # (c) Kurze Vertikallinie an der Referenz (blauer Strich)
  geom_segment(aes(
    x    = mean_risk, 
    xend = mean_risk,
    y    = as.numeric(Model) - 0.1,  
    yend = as.numeric(Model) + 0.1
  ),
  linetype = "solid", 
  color    = "gray") +
  
  # (d) Vertikale blaue Referenzlinie
  geom_vline(
    xintercept = mean_risk, 
    color      = "blue"
  ) +
  
  # (e) Zusätzliche (breitere) Vertikallinien an den Punkten
  geom_segment(aes(
    x    = Wert, 
    xend = Wert,
    y    = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color    = "grey", 
  size     = 0.4) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.65, 0.8),             
    breaks = seq(0.60, 0.80, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Model Performances vs. Risk Score Reference",
    x     = "C-Index",
    y     = "Model"
  ) +
  theme_minimal()

#---------------------------------------------------
#---------------------------------------------------





# Referenz-Linie
mean_risk <- 0.684

# Dummy-Werte für 6 Modelle (zwei Werte pro Modell)
models <- c("DeepSurv", "RandomSurvivalForest", "CatBoost", "PasNet", "PriorityLasso", "PenalizedCox")

values_1 <- runif(length(models), min = 0.70, max = 0.75)
values_2 <- runif(length(models), min = 0.70, max = 0.75)

# Data Frame erstellen
df <- data.frame(
  Model = rep(models, each = 2), # Jedes Modell doppelt, einmal pro Wert
  Wert = c(values_1, values_2),  # Beide Werte einfügen
  Gruppe = rep(c("A", "B"), times = length(models)) # Gruppe für Werte (A/B)
)

# Faktor-Level (Reihenfolge) festlegen
df$Model <- factor(df$Model, levels = models)

# Dot-Plot mit Referenzlinie und zwei Punkten pro Modell
library(ggplot2)

ggplot(df, aes(y = Model, x = Wert, group = Gruppe)) +
  
  # Graue Verbindungslinien zur blauen Referenz
  geom_segment(aes(yend = Model), xend = mean_risk, linetype = "solid", color = "gray") +
  
  # Vertikallinien an den Modellwerten
  geom_segment(aes(
    x = Wert, 
    xend = Wert,
    y = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color = "gray", 
  size = 0.4) +
  
  # Blaue Referenzlinie
  geom_vline(xintercept = mean_risk, color = "blue") +
  
  # Punkte für die zwei Werte je Modell
  geom_point(aes(color = Gruppe), size = 3) +
  
  # X-Achsen-Anpassung
  scale_x_continuous(
    limits = c(0.65, 0.8),             
    breaks = seq(0.60, 0.80, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "Model Performances with Two Values vs. Risk Score Reference",
    x = "C-Index",
    y = "Model",
    color = "Group"
  ) +
  theme_minimal()





################################################################################################################
# Bot Plot with base model baseline vs. exprs only



# 1) Baseline-Modell auf p-Daten (Referenzlinie)
baseline_pData_model <- 0.67  # Beispielwert

# 2) Die gleichen 6 Modelle wie zuvor, nur jetzt auf Gen-Daten trainiert
models <- c("DeepSurv", 
            "RandomSurvivalForest", 
            "CatBoost", 
            "PasNet", 
            "PriorityLasso", 
            "PenalizedCox", "Risk Score")

# Dummy-Werte für die 6 Modelle (auf Gen-Daten trainiert)
set.seed(123) # Für Reproduzierbarkeit
values <- runif(length(models), min = 0.65, max = 0.80)

# Data Frame erstellen
df <- data.frame(
  Model = models,
  Wert  = values
)
df[7, 2] <- 0.684
# Reihenfolge (Faktor-Level) beibehalten
df$Model <- factor(df$Model, levels = models)

# 3) Dot-Plot mit Referenzlinie (baseline_pData_model)
ggplot(df, aes(y = Model, x = Wert)) +
  
  # (a) Graue Verbindungslinie vom Gen-Modell-Wert zur pData-Referenz
  geom_segment(
    aes(yend = Model),
    xend     = baseline_pData_model,  # Konstanter Wert
    linetype = "solid", 
    color    = "gray"
  ) +
  
  # (b) Kurze Vertikallinie am Gen-Modell-Wert
  geom_segment(aes(
    x    = Wert, 
    xend = Wert,
    y    = as.numeric(Model) - 0.1,  
    yend = as.numeric(Model) + 0.1
  ),
  linetype = "solid", 
  color    = "gray") +
  
  # (c) Kurze Vertikallinie an der Referenz (baseline_pData_model)
  geom_segment(aes(
    x    = baseline_pData_model,
    xend = baseline_pData_model,
    y    = as.numeric(Model) - 0.1,  
    yend = as.numeric(Model) + 0.1
  ),
  linetype = "solid", 
  color    = "gray") +
  
  # (d) Vertikale Referenzlinie (blau gestrichelt)
  geom_vline(
    xintercept = baseline_pData_model, 
    color      = "blue",
    linetype   = "dashed"
  ) +
  
  # (e) Zusätzliche (breitere) Vertikallinien am Gen-Modell-Wert
  geom_segment(aes(
    x    = Wert, 
    xend = Wert,
    y    = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color    = "grey", 
  size     = 0.4) +
  
  # X-Achse anpassen (z.B. von 0.60 bis 0.85)
  scale_x_continuous(
    limits = c(0.60, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "pData Base Model vs. Exprs Models",
    x     = "C-Index",
    y     = "Model"
  ) +
  theme_minimal()

################################################################################################################
# Bot Plot with base model baseline vs. exprs + pData



# 1) Baseline-Modell auf p-Daten (Referenzlinie)
baseline_pData_model <- 0.67  # Beispielwert

# 2) Die gleichen 6 Modelle wie zuvor, nur jetzt auf Gen-Daten trainiert
models <- c("DeepSurv", 
            "RandomSurvivalForest", 
            "CatBoost", 
            "PenalizedCox")

# Dummy-Werte für die 6 Modelle (auf Gen-Daten trainiert)

values <- runif(length(models), min = 0.65, max = 0.80)

# Data Frame erstellen
df <- data.frame(
  Model = models,
  Wert  = values
)

# Reihenfolge (Faktor-Level) beibehalten
df$Model <- factor(df$Model, levels = models)

# 3) Dot-Plot mit Referenzlinie (baseline_pData_model)
ggplot(df, aes(y = Model, x = Wert)) +
  
  # (a) Graue Verbindungslinie vom Gen-Modell-Wert zur pData-Referenz
  geom_segment(
    aes(yend = Model),
    xend     = baseline_pData_model,  # Konstanter Wert
    linetype = "solid", 
    color    = "gray"
  ) +
  
  # (b) Kurze Vertikallinie am Gen-Modell-Wert
  geom_segment(aes(
    x    = Wert, 
    xend = Wert,
    y    = as.numeric(Model) - 0.1,  
    yend = as.numeric(Model) + 0.1
  ),
  linetype = "solid", 
  color    = "gray") +
  
  # (c) Kurze Vertikallinie an der Referenz (baseline_pData_model)
  geom_segment(aes(
    x    = baseline_pData_model,
    xend = baseline_pData_model,
    y    = as.numeric(Model) - 0.1,  
    yend = as.numeric(Model) + 0.1
  ),
  linetype = "solid", 
  color    = "gray") +
  
  # (d) Vertikale Referenzlinie (blau gestrichelt)
  geom_vline(
    xintercept = baseline_pData_model, 
    color      = "blue",
    linetype   = "dashed"
  ) +
  
  # (e) Zusätzliche (breitere) Vertikallinien am Gen-Modell-Wert
  geom_segment(aes(
    x    = Wert, 
    xend = Wert,
    y    = as.numeric(Model) - 0.2, 
    yend = as.numeric(Model) + 0.2
  ),
  linetype = "solid", 
  color    = "grey", 
  size     = 0.4) +
  
  # X-Achse anpassen (z.B. von 0.60 bis 0.85)
  scale_x_continuous(
    limits = c(0.60, 0.85),             
    breaks = seq(0.60, 0.85, 0.05),   
    labels = scales::label_number(accuracy = 0.001) 
  ) +
  
  # Beschriftungen & Theme
  labs(
    title = "pData Base Model vs. Exprs Models",
    x     = "C-Index",
    y     = "Model"
  ) +
  theme_minimal()





################################################################################################################
# Box-Plot dor model performances w.r.t the used data


# 1) Data 
categories <- c(
  "Clinical Data",
  "Common Genes",
  "Intersection of Genes",
  "Intersection of 
  Genes + Clinical Data",
  "Common Genes + 
  Clinical Data"
)

# 2) Models
models <- c(
  "DeepSurv", 
  "RandomSurvivalForest", 
  "CatBoost", 
  "PasNet", 
  "PriorityLasso", 
  "PenalizedCox"
)

# 3) Dummy data
set.seed(123)  # Damit die Zufallswerte reproduzierbar sind
df <- data.frame(
  Datentyp = rep(categories, each = length(models)),
  Modell   = rep(models, times = length(categories)),
  CIndex   = runif(length(categories) * length(models), min = 0.65, max = 0.75)
)

# 4) Boxplot 
ggplot(df, aes(x = Datentyp, y = CIndex)) +
  geom_boxplot(fill = "skyblue", outlier.shape = NA) +    # Boxplots in skyblue
  geom_jitter(width = 0.2, color = "blue", alpha = 0.7) + # Punkte in Blau
  labs(
    title = "Comparison of C-Index between Input Data Variations",
    x = "Datentyp",
    y = "C-Index"
  ) +
  theme_minimal(base_size = 14)




################################################################################################################
# Negative Bar Plot for imputed Genes


# Definition der Variablen common_genes und intersect_genes
common_genes <- colnames(train_exprs_merged_imputed)[-1]
intersect_genes <- colnames(train_exprs_merged_intersect[-1])

# Initialisieren der DataFrames für train und test
train_pData <- data.frame()
test_pData <- data.frame()

# Initialisieren von missing_genes_df
missing_genes_df <- data.frame(Kohorte = character(), Fehlende_Gene = numeric(), stringsAsFactors = FALSE)

# Loop über train_cohorts
for (cohort_name in names(train_cohorts)) { 
  # Kombinieren der train_pData
  train_pData <- rbind(
    train_pData, 
    train_cohorts[[cohort_name]]$pData[, c("AGE", "STUDY", "TISSUE", "GLEASON_SCORE", "PRE_OPERATIVE_PSA", 
                                           "BCR_STATUS", "MONTH_TO_BCR", "DOD_STATUS", "MONTH_TO_DOD")]
  )
  
  # Berechnung der fehlenden Gene
  missing_genes <- length(setdiff(common_genes, rownames(train_cohorts[[cohort_name]]$exprs)))
  
  # Hinzufügen einer Zeile zu missing_genes_df
  missing_genes_df <- rbind(
    missing_genes_df, 
    data.frame(Kohorte = cohort_name, Fehlende_Gene = missing_genes, stringsAsFactors = FALSE)
  )
}

# Loop über test_cohorts
for (cohort_name in names(test_cohorts)) { 
  # Kombinieren der test_pData
  test_pData <- rbind(
    test_pData, 
    test_cohorts[[cohort_name]]$pData[, c("AGE", "STUDY", "TISSUE", "GLEASON_SCORE", "PRE_OPERATIVE_PSA", 
                                          "BCR_STATUS", "MONTH_TO_BCR", "DOD_STATUS", "MONTH_TO_DOD")]
  )
  
  # Berechnung der fehlenden Gene
  missing_genes <- length(setdiff(common_genes, rownames(test_cohorts[[cohort_name]]$exprs)))
  
  # Hinzufügen einer Zeile zu missing_genes_df
  missing_genes_df <- rbind(
    missing_genes_df, 
    data.frame(Kohorte = cohort_name, Fehlende_Gene = missing_genes, stringsAsFactors = FALSE)
  )
}

# Berechnung der fehlenden Gene für TCGA_PRAD und UKD2
missing_genes_TCGA_PRAD_intersect <- length(setdiff(intersect_genes, rownames(test_cohorts[["TCGA_PRAD"]]$exprs)))
missing_genes_UKD2_intersect <- length(setdiff(intersect_genes, rownames(test_cohorts[["UKD2"]]$exprs)))

# Hinzufügen der Zeilen zu missing_genes_df
missing_genes_df <- rbind(
  missing_genes_df, 
  data.frame(Kohorte = "TCGA_PRAD to Intersect", Fehlende_Gene = missing_genes_TCGA_PRAD_intersect, stringsAsFactors = FALSE)
)
missing_genes_df <- rbind(
  missing_genes_df, 
  data.frame(Kohorte = "UKD2 to Intersect", Fehlende_Gene = missing_genes_UKD2_intersect, stringsAsFactors = FALSE)
)

# Hinzufügen eines Mappings basierend auf den Kohortennamen
rename_map <- c(
  "Atlanta_2014_Long" = "Cohort 1",
  "Belfast_2018_Jain" = "Cohort 2",
  "CamCap_2016_Ross_Adams" = "Cohort 3",
  "CancerMap_2017_Luca" = "Cohort 4",
  "CPC_GENE_2017_Fraser" = "Cohort 5",
  "CPGEA_2020_Li" = "Cohort 6",
  "DKFZ_2018_Gerhauser" = "Cohort 7",
  "MSKCC_2010_Taylor" = "Cohort 8",
  "Stockholm_2016_Ross_Adams" = "Cohort 9",
  "TCGA_PRAD" = "Cohort 10",
  "UKD2" = "Cohort 11",
  "TCGA_PRAD to Intersect" = "Cohort 10 (Inters.)",
  "UKD2 to Intersect" = "Cohort 11 (Inters.)"
)

# Erstellen der Spalte für umbenannte Kohorten
missing_genes_df$Cohort <- rename_map[missing_genes_df$Kohorte]

# Fehlende Werte in Cohort-Namen entfernen
missing_genes_df <- missing_genes_df[!is.na(missing_genes_df$Cohort), ]

# Reihenfolge der Balken festlegen
cohort_order <- c(
  "Cohort 1", "Cohort 2", "Cohort 3", "Cohort 4", "Cohort 5", 
  "Cohort 6", "Cohort 7", "Cohort 8", "Cohort 9", "Cohort 10", "Cohort 10 (Inters.)", 
  "Cohort 11", "Cohort 11 (Inters.)"
)

missing_genes_df$Cohort <- factor(missing_genes_df$Cohort, levels = cohort_order)



# Sicherstellen, dass alle Werte in Cohort definiert sind
missing_genes_df <- missing_genes_df %>%
  filter(Cohort %in% cohort_order) %>%
  mutate(
    ColorGroup = ifelse(
      Cohort %in% c("Cohort 10", "Cohort 11", "Cohort 10 (Inters.)", "Cohort 11 (Inters.)"),
      "Group 2",  # Orange für diese Kohorten
      "Group 1"   # Skyblue für die anderen
    )
  )

missing_genes_df$Fehlende_Gene <- -missing_genes_df$Fehlende_Gene



ggplot(missing_genes_df, aes(x = Cohort, y = Fehlende_Gene, fill = ColorGroup)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(
    values = c("Group 1" = "#ffcd66", "Group 2" = "#00706d"),  # Farben zuweisen
    name = "Group"  # Legendentitel setzen
  ) +
  scale_y_continuous(
    limits = c(-2500, 0),  # Skala bis -2500
    breaks = seq(-2500, 0, by = 500),  # Breakpoints alle 500
    labels = seq(-2500, 0, by = 500)   # Negative Werte anzeigen
  ) +
  labs(
    title = "Genes to Impute by Cohort",
    x = "Cohorts",
    y = "Number of Missing Genes"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0, size = 14, face = "bold")  # Titel linksbündig
  ) +
  geom_text(aes(label = Fehlende_Gene), vjust = 1.5, color = "black")  # Negative Labels




################################################################################################################
# % of Endpoint reached per group
all_pData_BCR_STATUS <- rbind(train_pData[, c("BCR_STATUS", "DOD_STATUS", "STUDY")], (test_pData[, c("BCR_STATUS", "DOD_STATUS","STUDY")]))
all_pData_BCR_STATUS[is.na(all_pData_BCR_STATUS$BCR_STATUS),"BCR_STATUS"] <- all_pData_BCR_STATUS[is.na(all_pData_BCR_STATUS$BCR_STATUS),]$"DOD_STATUS"


# Mapping für Umbenennung
rename_map <- c(
  "Atlanta_2014_Long" = "Cohort 1",
  "Belfast_2018_Jain" = "Cohort 2",
  "CamCap_2016_Ross_Adams" = "Cohort 3",
  "CancerMap_2017_Luca" = "Cohort 4",
  "CPC_GENE_2017_Fraser" = "Cohort 5",
  "CPGEA_2020_Li" = "Cohort 6",
  "DKFZ_2018_Gerhauser" = "Cohort 7",
  "MSKCC_2010_Taylor" = "Cohort 8",
  "Stockholm_2016_Ross_Adams" = "Cohort 9",
  "Ribolution_prad" = "Cohort 10",
  "Ribolution_ukd2" = "Cohort 11 (DOD)"
)

# Umbenennen der Studien
all_pData_BCR_STATUS <- all_pData_BCR_STATUS %>%
  mutate(STUDY = rename_map[STUDY])

# Berechnungen: Anzahl BCR_STATUS == TRUE und Gesamtanzahl pro Studie
result <- all_pData_BCR_STATUS %>%
  group_by(STUDY) %>%
  summarise(
    Total_Patients = n(),
    BCR_True_Count = sum(BCR_STATUS, na.rm = TRUE)
  ) %>%
  mutate(BCR_True_Proportion = BCR_True_Count / Total_Patients)



result <- result %>%
  mutate(
    # Gruppenzuweisung basierend auf den spezifischen Cohorts
    Group = ifelse(STUDY %in% c("Cohort 10", "Cohort 11 (DOD)"), "Group 2", "Group 1"),
    
    # Farbzuweisung basierend auf der Gruppe
    Color = ifelse(Group == "Group 2", "orange", "skyblue"),
    
    # Manuelle Festlegung der Reihenfolge der Faktor-Levels
    STUDY = factor(STUDY, levels = c(
      "Cohort 1", "Cohort 2", "Cohort 3", "Cohort 4", "Cohort 5",
      "Cohort 6", "Cohort 7", "Cohort 8", "Cohort 9",
      "Cohort 10", "Cohort 11 (DOD)"
    ))
  )

# Überprüfe die Faktor-Levels, um sicherzustellen, dass die Reihenfolge korrekt ist
levels(result$STUDY)



ggplot(result, aes(x = STUDY, y = BCR_True_Proportion, fill = Group)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("#ffcd66", "#00706d"), labels = c("Group A", "Group B")) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "% of BCR per cohort",
    x = "Cohort",
    y = "BCR",
    fill = "Group"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),  # Überschrift fett und Schriftgröße 14
    axis.text.x = element_text(angle = 45, hjust = 1)     # X-Achsen-Beschriftung rotieren
  )



################################################################################################################
#Survival Kurve

ggplot(missing_genes_df, aes(x = Cohort, y = MissingGenes, fill = ColorGroup)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("skyblue" = "skyblue", "yellow" = "yellow")) +
  labs(
    x = "Cohorts",
    y = "Number of Missing Genes",
    title = "Genes to Impute by Cohort"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Beispiel-Daten
df <- data.frame(
  Zeit = c(0, 1, 2, 3, 4, 5),
  Modell1 = c(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
  Modell2 = c(1.0, 0.85, 0.65, 0.45, 0.25, 0.0)
)

# Umformen in langes Format
df_long <- pivot_longer(df, cols = c("Modell1", "Modell2"),
                        names_to = "Modell", values_to = "Survival")

ggplot(df_long, aes(x = Zeit, y = Survival, color = Modell)) +
  geom_step(size = 1.2) +  # Treppenfunktionen für Survival-Kurven
  labs(title = "Survival-Kurven für zwei Modelle",
       x = "Zeit",
       y = "Überlebenswahrscheinlichkeit") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))









################################################################################################################




###Korelatoin
# Subset der numerischen Spalten
numerical_columns_exprs <- train_exprs_merged_intersect[, sapply(train_exprs_merged_intersect, is.numeric)]




# 1. Korrelationsmatrix berechnen
cor_matrix <- cor(numerical_columns_exprs)

# 2. Bedingung: Korrelation > 0.9 und nicht diagonal
high_corr <- cor_matrix < (-0.4) & lower.tri(cor_matrix)

# 3. Spalten mit hoher Korrelation zählen
columns_with_high_corr <- which(apply(high_corr, 2, any)) # Spalten finden

# Ausgabe der Anzahl der Spalten
length(columns_with_high_corr)






