import os
import pandas as pd

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

