import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.data_path = os.path.join(base_path, 'data')
        self.cohort_data_path = os.path.join(self.data_path, 'cohort_data')
        self.merged_data_path = os.path.join(self.data_path, 'merged_data')

    def load_csv_files(self, directory):
        """Loads all CSV files from a directory into a dictionary of DataFrames."""
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        return {f: pd.read_csv(os.path.join(directory, f), index_col=0) for f in csv_files}

    def load_cohort_data(self):
        """Loads cohort data including expression and pData files."""
        # Load expression data from cohort directory
        self.exprs_data = self.load_csv_files(os.path.join(self.cohort_data_path, 'exprs'))

        # Load original and imputed pData
        pdata_path = os.path.join(self.cohort_data_path, 'pData')
        self.pdata_original = self.load_csv_files(os.path.join(pdata_path, 'original'))
        self.pdata_imputed = self.load_csv_files(os.path.join(pdata_path, 'imputed'))

    def load_merged_data(self):
        """Loads merged data including expression and pData files."""
        # Load merged expression data from the "merged_data" directory
        exprs_path = os.path.join(self.merged_data_path, 'exprs')
        self.all_genes_data = self.load_csv_files(os.path.join(exprs_path, 'all_genes'))
        self.common_genes_data = self.load_csv_files(os.path.join(exprs_path, 'common_genes'))
        self.intersection_data = self.load_csv_files(os.path.join(exprs_path, 'intersection'))

        # Load merged original and imputed pData
        pdata_path = os.path.join(self.merged_data_path, 'pData')
        self.merged_pdata_original = self.load_csv_files(os.path.join(pdata_path, 'original'))
        self.merged_pdata_imputed = self.load_csv_files(os.path.join(pdata_path, 'imputed'))

    def load_all_data(self):
        """Loads all available data: cohort data and merged data."""
        self.load_cohort_data()  # Load cohort-specific data
        self.load_merged_data()  # Load merged data

    def get_data_summary(self):
        """Returns a summary of the loaded data as a dictionary."""
        return {
            "Expression files": len(self.exprs_data),
            "Original pData files": len(self.pdata_original),
            "Imputed pData files": len(self.pdata_imputed),
            "All Genes files": len(self.all_genes_data),
            "Common Genes files": len(self.common_genes_data),
            "Intersection files": len(self.intersection_data),
            "Merged original pData files": len(self.merged_pdata_original),
            "Merged imputed pData files": len(self.merged_pdata_imputed)
        }

    def get_cohort_data(self, cohort_name, use_imputed=True):
        """Get expression and pData for a specific cohort"""
        exprs_file = f"{cohort_name}.csv"
        pdata_file = f"{cohort_name}.csv"

        exprs = self.exprs_data.get(exprs_file)
        pdata = self.pdata_imputed.get(pdata_file) if use_imputed else self.pdata_original.get(pdata_file)

        return exprs, pdata

    def get_merged_data(self, gene_type='intersection', use_imputed=True):
        """
        Get merged expression and pData

        Parameters:
        -----------
        gene_type : str
            One of 'all_genes', 'common_genes', or 'intersection'
        use_imputed : bool
            Whether to use imputed pData
        """
        if gene_type == 'all_genes':
            exprs = self.all_genes_data.get('all_genes.csv')
        elif gene_type == 'common_genes':
            exprs = self.common_genes_data.get('common_genes_knn_imputed.csv' if use_imputed else 'common_genes.csv')
        else:  # intersection
            exprs = self.intersection_data.get('exprs_intersect.csv')

        pdata = self.merged_pdata_imputed.get('merged_imputed_pData.csv') if use_imputed else \
            self.merged_pdata_original.get('merged_original_pData.csv')

        return exprs, pdata

    def prepare_survival_data(self, pdata):
        """Convert pData to survival format"""
        time = pdata['MONTH_TO_BCR'].values
        event = pdata['BCR_STATUS'].values.astype(bool)

        y = np.zeros(len(time), dtype={'names': ('time', 'status'),
                                       'formats': ('f8', 'bool')})
        y['time'] = time
        y['status'] = event

        return y