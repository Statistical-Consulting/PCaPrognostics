import os
import numpy as np
from preprocessing.data_loader import DataLoader
from preprocessing.dimension_reduction import PCADimensionReduction
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class DataContainer:
    """Container for data loading and preprocessing"""

    def __init__(self, data_config=None, project_root=None):

        # Default configuration
        self.config = {
            'use_pca': False,
            'pca_threshold': 0.95,
            'gene_type': 'intersection',
            'use_imputed': True,
            'validation_split': 0.2,
            'use_cohorts': True, 
            'clinical_covs' : None
        }

        # Update with provided config
        if data_config is not None:
            self.config = data_config

        self.project_root = project_root
        self.pca = None
        self.groups = None
    
        
    def load_test_data(self, cohort = None):
        """
        Load merged test data.
        Optionally, filter data for a specific cohort.
        Also extract test group labels and optionally merge clinical data.
        """
        loader = DataLoader(self.project_root)
        # get merged test data
        X, pdata = loader.get_merged_test_data(gene_type=self.config['gene_type'],
                use_imputed=self.config['use_imputed'])
        
        if cohort is not None: 
            X = X[X.index.str.startswith(cohort)]
            pdata = pdata[pdata.index.str.startswith(cohort)]
        y = loader.prepare_survival_data(pdata)
        # Extract test group labels from the index using regex splitting
        import re
        def extract_first_part_with_number(s):
            match = re.search(r'\d', s)  # Find first number
            if match:
                return s[:match.start() + 1]  # Include the number in the first part
            return s

        # Extract test group labels
        test_groups = np.array([re.split(r'(\d)', idx, maxsplit=1)[:2] for idx in X.index])
        self.test_groups = np.array([''.join(part) if len(part) > 1 else part[0] for part in test_groups])

        # If clinical covariates are specified, process and merge them with gene data
        if self.config.get('clinical_covs', None) is not None:
            logger.info('Found clinical data specification')
            ### TODO: Remove ------------------------------------------
            pdata['AGE'] = pd.to_numeric(pdata['AGE'], errors='coerce')
            
            
            clin_data = pdata.loc[:, self.config['clinical_covs']] 
            clin_data.index = pdata.index
            X.index = pdata.index
            cat_cols = clin_data.select_dtypes(exclude=['number']).columns
            num_cols = clin_data.select_dtypes(exclude=['object']).columns
            clin_data_cat = clin_data.loc[:, cat_cols]
            # If one-hot encoding is not required, set a default for TISSUE column
            if self.config.get('requires_ohenc', True) is False:
                clin_data_cat.loc[ : , 'TISSUE'] = 'Fresh_frozen'
            # If one-hot encoding is required, manually create dummy columns for TISSUE
            if self.config.get('requires_ohenc', True) is True:
                print(clin_data_cat.columns)
                clin_data_cat.loc[: ,'TISSUE_FFPE'] = 0
                clin_data_cat.loc[: ,'TISSUE_Fresh_frozen'] = 1
                clin_data_cat.loc[: ,'TISSUE_Snap_frozen'] = 0
                print(clin_data_cat.columns)

                clin_data_cat.drop(['TISSUE'], axis = 1, inplace=True)
            # Get numerical clinical data
            clin_data_num = clin_data.loc[:, num_cols]
            
            if self.config.get('only_pData', False) is not False: 
                logger.info('Only uses pData')
                X = pd.concat([clin_data_cat, clin_data_num], axis = 1)
            else:
                X = pd.concat([clin_data_cat, clin_data_num, X], axis = 1)

        logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
        

    def load_data(self):
        """
        Load and preprocess training data.
        Applies PCA if configured, selects random genes if required,
        and merges clinical covariates if specified.
        """
        try:
            if self.project_root is None:
                raise ValueError("project_root must be provided")

            logger.info("Loading data...")

            # Load data
            loader = DataLoader(self.project_root)
            # Load merged data (X: gene expression features, pdata: patient metadata)
            X, pdata = loader.get_merged_data(
                gene_type=self.config['gene_type'],
                use_imputed=self.config['use_imputed']
            )

            # Convert to survival format
            y = loader.prepare_survival_data(pdata)

            # Extract cohort information
            self.groups = np.array([idx.split('.')[0] for idx in X.index])
            
            # Apply PCA if configured
            if self.config['use_pca']:
                logger.info("Applying PCA...")
                self.pca = PCADimensionReduction(
                    variance_threshold=self.config['pca_threshold']
                )
                X = self.pca.fit_transform(X)
            
            if self.config['select_random']: 
               logger.info("Selecting random subsets of genes...")
               X = X.sample(frac = self.config['random_frac'], axis = 1)

            # Process clinical covariates if provided in config
            if self.config.get('clinical_covs', None) is not None:
                logger.info('Found clinical data specification')
                ### TODO: Remove ------------------------------------------
                pdata['AGE'] = pd.to_numeric(pdata['AGE'], errors='coerce')
                
                
                clin_data = pdata.loc[:, self.config['clinical_covs']] 
                cat_cols = clin_data.select_dtypes(exclude=['number']).columns
                print(pdata.info())
                num_cols = clin_data.select_dtypes(exclude=['object']).columns
                clin_data_cat = clin_data.loc[:, cat_cols]
                # Perform one-hot encoding if required
                if self.config.get('requires_ohenc', True) is True:
                    ohc = OneHotEncoder()
                    clin_data_cat = ohc.fit_transform(clin_data_cat)
                    clin_data_cat = pd.DataFrame.sparse.from_spmatrix(clin_data_cat, columns=ohc.get_feature_names_out()).set_index(X.index)
                clin_data_num = clin_data.loc[:, num_cols]
                # Merge clinical data with gene data depending on configuration
                if self.config.get('only_pData', False) is not False: 
                    logger.info('Only uses pData')
                    X = pd.concat([clin_data_cat, clin_data_num], axis = 1)
                else:
                    X = pd.concat([clin_data_cat, clin_data_num, X], axis = 1)

            logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        
    def load_val_data(self):
        """
        Load data for validation purposes.
        Currently returns merged data and patient metadata.
        """
        loader = DataLoader(self.project_root)
        # TODO: adapt for testing cohorts
        X, pdata = loader.get_merged_data(
                gene_type=self.config['gene_type'],
                use_imputed=self.config['use_imputed']
            )
        y = loader.prepare_survival_data(pdata)

        return X, pdata 
        
        

    def get_groups(self):
        """Return cohort labels"""
        return self.groups

    
    def get_test_groups(self):
        """Return cohort labels"""
        return self.test_groups

    def get_train_val_split(self, X, y):
        """
        Create a train/validation split.
        If 'use_cohorts' is enabled and group labels are available, perform a cohort-based split.
        Otherwise, perform a random split based on the validation_split parameter.
        """
        try:
            if self.config['use_cohorts'] and self.groups is not None:
                # Cohort-based split
                unique_cohorts = np.unique(self.groups)
                val_cohort = np.random.choice(unique_cohorts)
                val_mask = self.groups == val_cohort
                train_mask = ~val_mask

                logger.info(f"Using cohort {val_cohort} for validation")
            else:
                # Random split
                indices = np.random.permutation(len(X))
                split = int(len(X) * (1 - self.config['validation_split']))
                train_mask = indices[:split]
                val_mask = indices[split:]

                logger.info(f"Using random {self.config['validation_split'] * 100}% validation split")

            # Use iloc for integer-based indexing
            X_train = X.iloc[train_mask]
            y_train = y[train_mask]
            X_val = X.iloc[val_mask]
            y_val = y[val_mask]

            logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            return X_train, y_train, X_val, y_val

        except Exception as e:
            logger.error(f"Error creating train/val split: {str(e)}")
            raise