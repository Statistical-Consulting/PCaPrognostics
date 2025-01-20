import os
import pandas as pd
import pickle
import re


def extract_values(row):
    if not isinstance(row, str):
        return None, None  # Handle non-string cases
    
    # Regex patterns
    test_cohort_match = re.search(r"'test_cohort':\s*'([^']+)'", row)
    test_score_match = re.search(r"'test_score':\s*([\d\.]+)", row)  # Numeric value
    
    # Extract matched values
    test_cohort = test_cohort_match.group(1) if test_cohort_match else None
    test_score = float(test_score_match.group(1)) if test_score_match else None
    if test_score is None: 
        test_score_match = re.search(r"'test_score':\s*array\(([\d.]+)", row)
        test_score = float(test_score_match.group(1)) if test_score_match else None

    
    return test_cohort, test_score

def load_split_results(results_path): 
    csv_files = [os.path.join(results_path, file) for file in os.listdir(results_path) if file.endswith('.csv')]
    
    combined_data = []
    for file in csv_files: 
        df = pd.read_csv(file, index_col=0)
        df['model'] =os.path.basename(file).replace("_cv.csv", "")
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

        # Join non-empty components into a single string with a separator (e.g., "_")
        dataset = "_".join(filter(None, components)) 
        df['dataset'] = dataset   
        
        df.drop(['fold_results', 'mean_score', 'std_score'], axis=1, inplace=True)
         
        combined_data.append(df)
    
    df_final = pd.concat(combined_data)
    return df_final

def load_all_results(results_path): 
    # Get a list of all CSV files in the "results" folder
    csv_files = [os.path.join(results_path, file) for file in os.listdir(results_path) if file.endswith('.csv')]
    # Read all CSV files, add a "name" column, and combine them into one dataframe
    combined_data = pd.concat(
        [
            # Read each CSV file and add a "name" column with the file name
            pd.read_csv(file).assign(model=os.path.basename(file).replace("_cv.csv", "")) for file in csv_files
        ],
        ignore_index=True  # Reset the index in the combined dataframe
    )
    combined_data = combined_data.loc[:, ['model', 'mean_score' ,'std_score']]
    combined_data = combined_data.groupby('model', as_index=False).agg(mean=('mean_score', 'mean'), sd = ('std_score', 'mean'))
    # View the combined data
    return combined_data

# Not necessary due to different sd structure
# def aggregate_results(results):
#     results_aggr = results.groupby('model', as_index=False).agg(mean=('ci', 'mean'), sd=('ci', 'std'))
#     return results_aggr


# TODO: Ergbebnisse aus Test und Nested reampling kombiniernen
def combine_results(results_nstd, results_test):
    if results_test is not None: 
        combined_results = results_nstd.merge(results_test, how = "left", left_on = 'model', right_on = 'model')
    else: 
        combined_results = results_nstd
        combined_results.loc[: , 'ci_coh1'] = None
        combined_results.loc[: , 'ci_coh2'] = None
    return(combined_results)


# TODO: Dataframe erstellen: Spalte 1: Name des Feautres, Spalte 2: Wert
# -------------------- functions to load feat. imp from model
def load_feat_imp(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Cat boost specific
    #print(model)
    # bei den Modellen die keine eigene Modellklasse von uns haben, muss man gucken wie der library interne Aufruf ist
    imps = model.model.get_feature_importance()
    
    df = pd.DataFrame({
    'feature': model.model.feature_names_,
    'value': imps
    })
    
    df = df.sort_values(by = "value", ascending=False)
    df = df[df.loc[: , 'value'] > 0]
    
    return df

def load_model(model_path): 
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    return model

def extract_scores_from_files(folder_path):
    """
    Reads all files in a given folder, extracts the 'mean_score' and 'std_score' 
    columns from the first row of each file, and saves them into a new DataFrame.

    Args:
        folder_path (str): Path to the folder containing the files.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Model', 'mean_score', 'std_score'].
    """
    # Initialize an empty list to store the extracted data
    data = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a valid CSV file
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            try:
                # Read the file into a DataFrame
                df = pd.read_csv(file_path)

                # Extract the 'mean_score' and 'std_score' from the first row
                if 'mean_score' in df.columns and 'std_score' in df.columns:
                    mean_score = df.loc[0, 'mean_score']
                    std_score = df.loc[0, 'std_score']

                    # Append the data to the list
                    data.append({'Model': file_name, 'mean_score': mean_score, 'std_score': std_score})
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

    # Create a DataFrame from the extracted data
    result_df = pd.DataFrame(data)

    return result_df

