import os
import pandas as pd
import pickle
import re


def extract_values(row):
    """
    Extracts 'test_cohort' and 'test_score' values from a given row string.
    Args:
        row (str): A string containing key-value pairs in a structured format.
    
    Returns:
        tuple: (test_cohort, test_score) where:
            - test_cohort (str or None): Extracted test cohort name.
            - test_score (float or None): Extracted test score, converted to float.
    """
    if not isinstance(row, str):
        return None, None 
    
    test_cohort_match = re.search(r"'test_cohort':\s*'([^']+)'", row)
    test_score_match = re.search(r"'test_score':\s*([\d\.]+)", row)  
    
    test_cohort = test_cohort_match.group(1) if test_cohort_match else None
    test_score = float(test_score_match.group(1)) if test_score_match else None
    if test_score is None: 
        test_score_match = re.search(r"'test_score':\s*array\(([\d.]+)", row)
        test_score = float(test_score_match.group(1)) if test_score_match else None
    return test_cohort, test_score


def load_split_results(results_path, model_name = None):
    """
    Loads and processes cross-validation result files from a directory.
 
    Args:
        results_path (str): Path to the directory containing CSV files.
        model_name (str): Name of model class
    
    Returns:
        pd.DataFrame: A combined DataFrame containing processed results
    """
    csv_files = [os.path.join(results_path, file) for file in os.listdir(results_path) if file.endswith('.csv')]
    
    combined_data = []
    for file in csv_files: 
        df = pd.read_csv(file, index_col=0)
        df['model_class'] = model_name
        df['model'] = os.path.basename(file).replace(".csv", "")
        df[['test_cohort', 'ci']] = df['fold_results'].apply(lambda x: pd.Series(extract_values(x)))
                
        contains_pData = bool(re.search(r"pData", file, re.IGNORECASE))
        contains_intersection = bool(re.search(r"inter|intersection", file, re.IGNORECASE))
        contains_imputed = bool(re.search(r"imp|imputed|common", file, re.IGNORECASE))
        contains_aenc = bool(re.search(r"aenc|auto|autoenc", file, re.IGNORECASE))
        contains_scores = bool(re.search(r"score|scores", file, re.IGNORECASE))    
        
        components = [
            "pData" if contains_pData else "",
            "Intersection" if contains_intersection else "",
            "Imputed" if contains_imputed else "",
            "AutoEncoder" if contains_aenc else "",
            "Scores" if contains_scores else ""
        ]

        dataset = "_".join(filter(None, components)) 
        df['dataset'] = dataset   
        df.drop(['fold_results', 'mean_score', 'std_score'], axis=1, inplace=True) 
        combined_data.append(df)
    
    df_final = pd.concat(combined_data)
    return df_final


def load_all_results(results_path): 
    """
    Loads and aggregates all cross-validation result files from a directory.
    
    Args:
        results_path (str): Path to the directory containing CSV files.
    
    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    csv_files = [os.path.join(results_path, file) for file in os.listdir(results_path) if file.endswith('.csv')]
    combined_data = pd.concat(
        [
            pd.read_csv(file).assign(model=os.path.basename(file).replace(".csv", "")) for file in csv_files
        ],
        ignore_index=True 
    )
    combined_data = combined_data.loc[:, ['model', 'mean_score' ,'std_score']]
    combined_data = combined_data.groupby('model', as_index=False).agg(mean=('mean_score', 'mean'), sd = ('std_score', 'mean'))
    return combined_data


def combine_results(results_nstd, results_test = None):
    """
    Merges nested resampling results with test results.

    Args:
        results_nstd (pd.DataFrame): DataFrame containing nested resampling results.
        results_test (pd.DataFrame or None): DataFrame containing test results or None.

    Returns:
        pd.DataFrame: merged DataFrame
    """
    if results_test is not None: 
        combined_results = results_nstd.merge(results_test, how = "left", left_on = 'model', right_on = 'model')
    else: 
        combined_results = results_nstd
        combined_results.loc[: , 'ci_coh1'] = None
        combined_results.loc[: , 'ci_coh2'] = None
    return(combined_results)


def load_feat_imp(model):
    """
    Loads and processes feature importance values from a saved model.
    Args:
        model : saved model object.
    
    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'feature': Name of the feature.
            - 'value': Importance score of the feature.
    """
    imps = model.model.get_feature_importance()
    
    df = pd.DataFrame({
    'feature': model.model.feature_names_,
    'value': imps
    })
    
    df = df.sort_values(by = "value", ascending=False)
    df = df[df.loc[: , 'value'] > 0]
    return df


def feat_imp_all_models(model_path, model_name, DATA_CONFIG): 
    """
    Computes and aggregates feature importance values for all models in a given directory.
    Args:
        model_path (str): Path to the directory containing trained model files (.pkl).
        model_name (str): Name of the model type (e.g., CatBoost, XGBoost, etc.).
        DATA_CONFIG (dict): Dictionary containing data configuration settings.

    Returns:
        pd.DataFrame: A DataFrame containing feature importances
    """
    files = os.listdir(model_path)
    imps_list = []
    
    for file in files:
        contains_pData = bool(re.search(r"pData", file, re.IGNORECASE))
        contains_intersection = bool(re.search(r"inter|int|intersection", file, re.IGNORECASE))
        contains_imputed = bool(re.search(r"imp|imputed|common", file, re.IGNORECASE))
        contains_aenc = bool(re.search(r"aenc|auto|autoenc|autoencoder", file, re.IGNORECASE))
        contains_scores = bool(re.search(r"score|scores", file, re.IGNORECASE))
        
        # Load data based on file type
        if contains_intersection:
            DATA_CONFIG['gene_type'] = 'intersection'
        elif contains_imputed:
            DATA_CONFIG['gene_type'] = 'common_genes'
        elif contains_aenc:
            DATA_CONFIG['gene_type'] = 'intersection'
        elif contains_scores: 
            DATA_CONFIG['gene_type'] = 'scores'
        if contains_pData:
            DATA_CONFIG['clinical_covs'] = ["AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA']
        if contains_pData and not contains_intersection and not contains_imputed and not contains_aenc and not contains_scores: 
            DATA_CONFIG['only_pData'] = True
            DATA_CONFIG['gene_type'] = None
            
        model = load_model(os.path.join(model_path, file))  
        
        if contains_aenc: 
            pass
        else: 
            components = [
                "pData" if contains_pData else "",
                "Intersection" if contains_intersection else "",
                "Imputed" if contains_imputed else "",
                "AutoEncoder" if contains_aenc else "",
                "Scores" if contains_scores else ""
            ]

            # Join non-empty components into a single string with a separator (e.g., "_")
            dataset = "_".join(filter(None, components))         
            imps = load_feat_imp(model)
            imps.loc[:, 'model_class'] =  model_name
            imps.loc[:, 'dataset'] = dataset
            imps_list.append(imps)
        
    df = pd.concat(imps_list, axis = 0)
    return df


def load_model(model_path): 
    """
    Loads a trained model
    
    Args:
        model_path (str): Path to the saved model file (pickle format).
    
    Returns:
        object: The trained model instance.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# def extract_scores_from_files(folder_path):
#     """
#     Reads all files in a given folder, extracts the 'mean_score' and 'std_score' 
#     columns from the first row of each file, and saves them into a new DataFrame.

#     Args:
#         folder_path (str): Path to the folder containing the files.

#     Returns:
#         pd.DataFrame: A DataFrame with columns ['Model', 'mean_score', 'std_score'].
#     """
#     # Initialize an empty list to store the extracted data
#     data = []

#     # Iterate through all files in the folder
#     for file_name in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file_name)

#         # Check if the file is a valid CSV file
#         if os.path.isfile(file_path) and file_name.endswith('.csv'):
#             try:
#                 # Read the file into a DataFrame
#                 df = pd.read_csv(file_path)

#                 # Extract the 'mean_score' and 'std_score' from the first row
#                 if 'mean_score' in df.columns and 'std_score' in df.columns:
#                     mean_score = df.loc[0, 'mean_score']
#                     std_score = df.loc[0, 'std_score']

#                     # Append the data to the list
#                     data.append({'Model': file_name, 'mean_score': mean_score, 'std_score': std_score})
#             except Exception as e:
#                 print(f"Error reading file {file_name}: {e}")

#     # Create a DataFrame from the extracted data
#     result_df = pd.DataFrame(data)

#     return result_df

