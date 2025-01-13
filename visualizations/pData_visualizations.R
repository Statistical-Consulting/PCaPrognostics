library(ggplot2)
library(dplyr)
library(tidyr)
library(reshape2)

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
train_cohorts <- load_cohorts( "PCa_cohorts.Rds")
train_cohorts$Stockholm_2016_Ross_Adams$pData$AGE
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

train_tissue_counts <- table(train_pData$TISSUE)
test_pData$TISSUE[test_pData$TISSUE == "fresh-frozen"] <- "Fresh_frozen"
test_tissue_counts <- table(test_pData$TISSUE)


all_tissue_types <- union(names(train_tissue_counts), names(test_tissue_counts))
train_tissue_counts <- train_tissue_counts[all_tissue_types]
test_tissue_counts <- test_tissue_counts[all_tissue_types]


train_tissue_counts[is.na(train_tissue_counts)] <- 0
test_tissue_counts[is.na(test_tissue_counts)] <- 0


stacked_counts <- rbind(train_tissue_counts, test_tissue_counts)
rownames(stacked_counts) <- c("Train", "Test")


bar_colors <- c("skyblue", "orange")


barplot(
  stacked_counts,
  main = "Number of patients by tissue type (Training and Test Data)",
  xlab = "Tissue Type",
  ylab = "Number of patients",
  col = bar_colors,
  names.arg = c("Fresh Frozen", "FFPE", "Snap Frozen"),
  legend.text = rownames(stacked_counts), # Legende für Train/Test
  args.legend = list(x = "topright") # Position der Legende
)


################################################################################################################
#Boxplot for age of train and test
# Reorder the factor levels of STUDY

train_pData$AGE <- as.numeric(train_pData$AGE )
desired_order <- c(
  "Atlanta_2014_Long", "Belfast_2018_Jain", "CamCap_2016_Ross_Adams", 
  "CancerMap_2017_Luca", "CPC_GENE_2017_Fraser", "CPGEA_2020_Li", 
  "DKFZ_2018_Gerhauser", "MSKCC_2010_Taylor", "Stockholm_2016_Ross_Adams",
  "Ribolution_prad", "Ribolution_ukd2"                                   
)

# Update STUDY levels to the desired order
all_pData_AGE$STUDY <- factor(all_pData_AGE$STUDY, levels = desired_order)

# Ensure AGE is numeric and STUDY has a consistent factor order
all_pData_AGE <- all_pData_AGE %>% 
  mutate(
    AGE = as.numeric(AGE), # Ensure AGE is numeric
    STUDY = factor(STUDY, levels = all_studies) # Order STUDY by Train and Test groups
  )

# Add missing studies with NA for AGE
all_pData_AGE <- all_pData_AGE %>%
  complete(STUDY = all_studies, fill = list(AGE = NA)) # Add missing studies

# Add a new column to distinguish Train and Test cohorts
all_pData_AGE <- all_pData_AGE %>%
  mutate(COHORT_TYPE = case_when(
    STUDY %in% train_studies ~ "Train", # Assign "Train" for Train studies
    STUDY %in% test_studies ~ "Test",   # Assign "Test" for Test studies
    TRUE ~ "Unknown"                    # Default to "Unknown" if not classified
  ))

# Create the box plot
ggplot(all_pData_AGE, aes(x = STUDY, y = AGE, fill = COHORT_TYPE)) +
  geom_boxplot(outlier.shape = NA) +  # Create box plots, without showing outliers
  geom_jitter(aes(color = COHORT_TYPE), width = 0.2, alpha = 0.6) + # Add jittered points
  scale_fill_manual(values = c("Train" = "skyblue", "Test" = "orange", "Unknown" = "gray")) + # Fill colors
  scale_color_manual(values = c("Train" = "blue", "Test" = "darkorange", "Unknown" = "gray")) + # Point colors
  theme_minimal() +
  labs(
    title = "Age Distribution by Study and Cohort Type",
    x = "Study",
    y = "Age",
    fill = "Cohort Type",
    color = "Cohort Type"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1) # Rotate x-axis labels for better readability
  )





################################################################################################################
#Stacked Bar Plot for Gleason



train_gleason_counts <- table(train_pData$GLEASON_SCORE)
test_gleason_counts <- table(test_pData$GLEASON_SCORE)


train_percent <- 100 * train_gleason_counts / sum(train_gleason_counts)
test_percent <- 100 * test_gleason_counts / sum(test_gleason_counts)


all_gleason_scores <- union(names(train_gleason_counts), names(test_gleason_counts))


train_percent <- train_percent[all_gleason_scores]
test_percent <- test_percent[all_gleason_scores]


train_percent[is.na(train_percent)] <- 0
test_percent[is.na(test_percent)] <- 0


gleason_data <- data.frame(
  Gleason_Score = rep(all_gleason_scores, 2),
  Percent = c(train_percent, test_percent),
  Group = rep(c("Train", "Test"), each = length(all_gleason_scores))
)


ggplot(gleason_data, aes(x = Gleason_Score, y = Percent, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", Percent)), 
            position = position_dodge(width = 0.8), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("Train" = "skyblue", "Test" = "orange")) +
  labs(
    title = "Percentage of Patients by Gleason Score (Training and Test Data)",
    x = "Gleason Score",
    y = "Percentage of Patients",
    fill = "Group"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




################################################################################################################
#PSA Box Plot





train_pData$PRE_OPERATIVE_PSA <- as.numeric(train_pData$PRE_OPERATIVE_PSA)
test_pData$PRE_OPERATIVE_PSA <- as.numeric(test_pData$PRE_OPERATIVE_PSA)


psa_data <- rbind(
  train_pData %>%
    select(PRE_OPERATIVE_PSA, STUDY) %>%
    mutate(Group = "Train"),
  test_pData %>%
    select(PRE_OPERATIVE_PSA, STUDY) %>%
    mutate(Group = "Test")
)

# All Data
ggplot(psa_data, aes(y = STUDY, x = PRE_OPERATIVE_PSA, fill = Group)) +
  geom_boxplot(outlier.shape = NA) + 
  geom_jitter(aes(color = Group), width = 0.2, alpha = 0.5) +  
  scale_fill_manual(values = c("Train" = "skyblue", "Test" = "orange")) + 
  scale_color_manual(values = c("Train" = "blue", "Test" = "darkorange")) + 
  labs(
    title = "Distribution of PSA Values by Study and Group",
    x = "PSA Value",
    y = "Study",
    fill = "Group",
    color = "Group"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(angle = 0))


# Limited X Scale
ggplot(psa_data, aes(y = STUDY, x = PRE_OPERATIVE_PSA, fill = Group)) +
  geom_boxplot(outlier.shape = NA) +  
  geom_jitter(aes(color = Group), width = 0.2, alpha = 0.5) + 
  scale_fill_manual(values = c("Train" = "skyblue", "Test" = "orange")) + 
  scale_color_manual(values = c("Train" = "blue", "Test" = "darkorange")) + 
  labs(
    title = "Distribution of PSA Values by Study and Group",
    x = "PSA Value",
    y = "Study",
    fill = "Group",
    color = "Group"
  ) +
  coord_cartesian(xlim = c(0, 100)) +  
  theme_minimal() +
  theme(axis.text.y = element_text(angle = 0))  


sum(is.na(test_pData$MONTH_TO_BCR))
train_cohorts$Stockholm_2016_Ross_Adams$pData$GLEASON_SCORE




################################################################################################################
#Bar Chart Expressions



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
UKD2_genes <- rownames(test_cohorts$UKD2$exprs)
all_test_genes <- unique(c(TCGA_PRAD_genes, UKD2_genes))
all_train_and_test_genes <- unique(c(all_test_genes, colnames(train_exprs_merged_all_genes)))

exprs_num_df <- rbind(exprs_num_df, data.frame(`Exprs Source` = "TCGA_PRAD_genes", `Exprs Count` = nrow(test_cohorts$TCGA_PRAD$exprs)))
exprs_num_df <- rbind(exprs_num_df, data.frame(`Exprs Source` = "UKD2", `Exprs Count` = nrow(test_cohorts$UKD2$exprs)))
exprs_num_df <- exprs_num_df %>%
  arrange(desc(`Exprs.Count`))
exprs_num_df <- rbind(exprs_num_df, data.frame(`Exprs Source` = "Common Genes", `Exprs Count` = ncol(train_exprs_merged_imputed)))
exprs_num_df <- rbind(exprs_num_df, data.frame(`Exprs Source` = "Intersection", `Exprs Count` = ncol(train_exprs_merged_intersect)))
exprs_num_df <- rbind(exprs_num_df, data.frame(`Exprs Source` = "All Training Genes", `Exprs Count` = ncol(train_exprs_merged_all_genes)))
exprs_num_df <- rbind(exprs_num_df, data.frame(`Exprs Source` = "All Test Genes", `Exprs Count` = length(all_test_genes)))
exprs_num_df <- rbind(exprs_num_df, data.frame(`Exprs Source` = "All Genes", `Exprs Count` = length(all_train_and_test_genes)))
                      
                      



# Invert order
exprs_num_df$Exprs.Source <- factor(exprs_num_df$Exprs.Source, levels = rev(exprs_num_df$Exprs.Source))


exprs_num_df <- exprs_num_df %>%
mutate(
  Color_Group = case_when(
    row_number() >=12 ~ "Gray",         
    row_number() == 6 ~ "Orange",   
    row_number() == 1 ~ "Orange",  
    TRUE ~ "Blue"                      
  )
)



color_map <- c("Blue" = "skyblue", "Orange" = "orange", "Gray" = "gray")


legend_labels <- c(
  "Blue" = "Train",   
  "Orange" = "Test ",     
  "Gray" = "Combinatinos"      
)

ggplot(exprs_num_df, aes(x = Exprs.Source, y = Exprs.Count, fill = Color_Group)) +
  geom_bar(stat = "identity", width = 0.8) +  
  geom_text(aes(label = Exprs.Count), hjust = -0.2, size = 3) +  
  scale_fill_manual(
    values = color_map,          
    labels = legend_labels      
  ) +
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
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    plot.title = element_text(hjust = 0.5, size = 14)
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

