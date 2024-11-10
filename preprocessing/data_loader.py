import os
import pandas as pd
import numpy as np
from sksurv.util import Surv


class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.data_path = os.path.join(base_path, 'data')
        self.cohort_data_path = os.path.join(self.data_path, 'cohort_data')
        self.merged_data_path = os.path.join(self.data_path, 'merged_data')

        # Initialize data containers
        self.exprs_data = None
        self.pdata_original = None
        self.pdata_imputed = None
        self.all_genes_data = None
        self.common_genes_data = None
        self.intersection_data = None
        self.merged_pdata_original = None
        self.merged_pdata_imputed = None

        self._verify_paths()
        self.load_all_data()

    def _verify_paths(self):
        """Verify that all required data directories exist"""
        required_paths = [
            self.data_path,
            self.cohort_data_path,
            self.merged_data_path,
            os.path.join(self.cohort_data_path, 'exprs'),
            os.path.join(self.cohort_data_path, 'pData', 'original'),
            os.path.join(self.cohort_data_path, 'pData', 'imputed'),
            os.path.join(self.merged_data_path, 'exprs', 'all_genes'),
            os.path.join(self.merged_data_path, 'exprs', 'common_genes'),
            os.path.join(self.merged_data_path, 'exprs', 'intersection'),
            os.path.join(self.merged_data_path, 'pData', 'original'),
            os.path.join(self.merged_data_path, 'pData', 'imputed')
        ]

        missing_paths = [p for p in required_paths if not os.path.exists(p)]
        if missing_paths:
            raise FileNotFoundError(f"Missing directories: {missing_paths}")

    def load_csv_files(self, directory):
        """Load all CSV files from directory into dictionary"""
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        return {
            f: pd.read_csv(
                os.path.join(directory, f),
                index_col=0,
                engine='python'  # More robust CSV parsing
            ) for f in csv_files
        }

    def load_all_data(self):
        """Load all available data"""
        # Load expression data
        self.exprs_data = self.load_csv_files(
            os.path.join(self.cohort_data_path, 'exprs')
        )

        # Load pData
        pdata_path = os.path.join(self.cohort_data_path, 'pData')
        self.pdata_original = self.load_csv_files(
            os.path.join(pdata_path, 'original')
        )
        self.pdata_imputed = self.load_csv_files(
            os.path.join(pdata_path, 'imputed')
        )

        # Load merged data
        exprs_path = os.path.join(self.merged_data_path, 'exprs')
        self.all_genes_data = self.load_csv_files(
            os.path.join(exprs_path, 'all_genes')
        )
        self.common_genes_data = self.load_csv_files(
            os.path.join(exprs_path, 'common_genes')
        )
        self.intersection_data = self.load_csv_files(
            os.path.join(exprs_path, 'intersection')
        )

        # Load merged pData
        merged_pdata_path = os.path.join(self.merged_data_path, 'pData')
        self.merged_pdata_original = self.load_csv_files(
            os.path.join(merged_pdata_path, 'original')
        )
        self.merged_pdata_imputed = self.load_csv_files(
            os.path.join(merged_pdata_path, 'imputed')
        )

    def get_merged_data(self, gene_type='intersection', use_imputed=True):
        """Get merged expression and pData"""
        if gene_type == 'all_genes':
            exprs = self.all_genes_data['all_genes.csv']
        elif gene_type == 'common_genes':
            exprs = self.common_genes_data[
                'common_genes_knn_imputed.csv' if use_imputed else 'common_genes.csv'
            ]
        else:  # intersection
            exprs = self.intersection_data['exprs_intersect.csv']

        pdata = self.merged_pdata_imputed['merged_imputed_pData.csv'] \
            if use_imputed else self.merged_pdata_original['merged_original_pData.csv']

        return exprs, pdata

    def prepare_survival_data(self, pdata):

        # Convert status to boolean
        status = pdata['BCR_STATUS'].astype(bool).values

        # Convert time to float
        time = pdata['MONTH_TO_BCR'].astype(float).values

        # Create structured array using scikit-survival utility
        y = Surv.from_arrays(
            event=status,
            time=time,
            name_event='status',  # Explicitly use 'status'
            name_time='time'
        )

        return y
