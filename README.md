# Consulting Project for Fraunhofer IZI

This repository contains files used during the consulting project for Fraunhofer IZI, including data preprocessing, modeling, and visualization of results.

## Code Architecture

<img width="802" alt="Codegrafik1" src="https://github.com/user-attachments/assets/4541a784-5e3d-4370-bb9f-61b909857a6e" />



## How to Work with This Code

### 1. Data Placement
Before running the code, place the required data files in the `data` folder:
- `Revised_ProstaTrend.Rds` (ProstaTrend-ffpe scores)
- `PCa_cohorts.RDS` (training cohorts)
- `PCa_cohorts_2.RDS` (test cohorts)

### 2. Data Preprocessing
The preprocessing step runs before the rest of the code and generates the necessary CSV files for modeling.

- **Main preprocessing:**  
  Run `preprocessing.R` to preprocess the training cohorts.
- **Additional preprocessing (if required):**  
  Run `preprocessing_2.R` to apply further preprocessing to test cohorts and ProstaTrend-FFPE scores.
- **Dimensionality reduction:**
  1. Run the `generate_autoencoder.ipynb` in Google Colab (link in 'pretrained_models_ae/generate_autoencoder.ipynb')
  2. Download the 'csv', 'csv_eval' and 'models'-folders from this notebook into the 'pretrained_models_ae'-folder of this repository

### 3. Running the Models
There are two types of model implementations in this repository. Some models are fully implemented within the repository’s local structure, whereas others only run on Google Colab to thus take advantage of GPU resources. To use these models:
1. Open the respective Google Colab notebook.
2. Upload the necessary files (for tuning, training, or evaluation) into the Colab `content` pane.
3. Execute the required code chunks according to the instructions provided in the notebook’s comments.
In addition some models are implemented in Python, some in R.

##### Models implemented in Python:
1. Nested resampling, model tuning and final model training in the `<model_name>_modelling.py`-files:
    - To load the preferred dataset, adapt the `DATA_CONFIG`:
        `DATA_CONFIG = {
        'use_pca': False, # Experimental feautre, does PCA on the gene data; not recommended to use during modelling process
        'pca_threshold': 0.85, # Only relevant if use_pca == True
        'gene_type': ('intersection', 'common_genes', 'all_genes'), # gene data to be loaded 
        'use_imputed': True, # wether imputed data is to be returend or data with NA`s for missing values
        'select_random' : False, # Experimental feature, selects a random subset of the gene data; not recommended to use during modelling process
        'use_cohorts': False, # Wether to return a dict of seperate cohort csvs; not combinable with modelling process
        'requires_ohenc' : False, # Wether categorical data requires One-Hot-encoding; Only relevant if `clinical covs` is specified
        'only_pData': False, # Wether to only return clinical data
        'clinical_covs' : ["AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA'] # clinical variables to be used; Remove if no clini. Data is wanted
        }`
    - To use the preferred modelling config, adapt the `MODEL_CONFIG`:
        `MODEL_CONFIG = {
        'params_cv':{ # param grid for hyperparameter tuning; Needs `model__`-prefix
        'model__iterations': [2],
        'model__learning_rate': [0.1],
        'model__depth': [10],
        'model__min_data_in_leaf': [10],
        },
        'refit': True, # Wether a final model is to be tuned and fitted
        'do_nested_resampling': True,  # Wether nested resampling should be done
        'path' : RESULTS_DIR, # path to save the results to, ideally `results` within the model folder
        'fname_cv' : 'test' # filename for results (both model and nested resampling results)
        }`
2. Analysis of results via the `<model_name>_analysis.py`-files: Make sure that a `results/model` (containing final models) and a `results/results (containing .csv-files from nested resampling) folder exists within model folder
3. Implemented models:
    - GBoost `models/cat_boost`: Modelling runs locally
    - DeepSurv `models/deep_surv`: Modelling runs only runs in provided Google Colab Notebooks
    - CoxPN `models/cox_pas_net`:
      1. Run `create_pathways.R` to create pathway mask
      2. Modelling runs locally (not recommended) or again in provided Google Colab Notebooks
    
##### Models implemnted in R
1. Modelling Process in the `<model_name>_modelling.R`-files
    - To load the wanted data set, set:
use_aenc = TRUE # if latent space from AE is to be used
use_inter = FALSE # if gene data in general is to be used
use_exprs = FALSE # if intersection data is to be used --> if FALSE & use_inter then imputed/common genes are used
use_pData = FALSE # if clinical data is used
vars_pData = c("AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA')
3. Analysis of results via the `<model_name>_analysis.R`-files: Make sure that a `results/model` (containing final models) and a `results/results (containing .csv-files from nested resampling) folder exists within model folder
4. Implemented models:
    - CoxPH `models/pen_cox`: Modelling runs locally
    - RSF `models/rsf`: Modelling runs locally
    - PrioLasso `models/prio_lasso`:
     1. Execute the `create_blocks.py`, save the resulting block structure into the `prio_lasso/`-folder
     2. Modify the paths for `df_blockwise_data` and `df_blockwise_indcs` in `priority_lasso_modelling.r` and `priority_lasso_analysis.R`
     3. Modelling runs locally

---

### 💡 Notes:
- Ensure that all required dependencies are installed before running the code.
- Follow the inline comments in the scripts and notebooks for additional guidance.

---

📌 *For further details or issues, feel free to create an issue in this repository.*

