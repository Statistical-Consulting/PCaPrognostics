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
There are two types of model implementations in this repository:

#### **📌 Models Implemented Within This Repository**
Some models are fully implemented within the repository’s local structure. These models automatically access the preprocessed CSV files.
1. Models implemented in Python:
2. Models implemnted in R:
   - CoxPH `models/pen_cox`:
     1. For model fitting and nested resampling run `pen_cox_modelling.R`
     2. For results analysis run `pen_cox_analysis.R`
   - RSF `models/rsf`:
     1. For model fitting and nested resampling run `rsf_modelling.R`
     2. For results analysis run `rsf_analysis.R`
   - PrioLasso `models/prio_lasso`:
     1. Execute the `create_blocks.py, save the resulting block structure into the `prio_lasso/`-folder
     2. Modify the paths for `df_blockwise_data` and `df_blockwise_indcs` in `priority_lasso_modelling.r` and `priority_lasso_analysis.R`
     3. For model fitting and nested resampling run `priority_lasso_modelling.R`
     4. For results analysis run `priority_lasso_analysis.R`

To use them:
1. Define the datasets to be used as input.
2. Configure the model hyperparameter grid for tuning or specify the exact parameters for training.
3. The results from nested resampling or the trained model are automatically stored.

📌 *TBD: How to evaluate new data on a model*

#### **📌 Models Implemented in Google Colab (DeepSurv & Cox-PASNet)**
DeepSurv and Cox-PASNet are not included in this repository’s folder structure. Instead, they are implemented in Google Colab notebooks to take advantage of GPU performance.

To use these models:
1. Open the respective Google Colab notebook.
2. Upload the necessary files (for tuning, training, or evaluation) into the Colab `content` pane.
3. Execute the required code chunks according to the instructions provided in the notebook’s comments.

---

### 💡 Notes:
- Ensure that all required dependencies are installed before running the code.
- Follow the inline comments in the scripts and notebooks for additional guidance.

---

📌 *For further details or issues, feel free to create an issue in this repository.*

