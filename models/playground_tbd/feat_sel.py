import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup paths
PROJECT_ROOT =  os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Setup directories
MODEL_DIR = os.path.join(os.getcwd(), 'model')
RESULTS_DIR = os.path.join(os.getcwd(), 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Imports
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from utils.feature_selection import FoldAwareSelectFromModel, FoldAwareAE
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
#from utils.feature_selection import FoldAwareSelectFromModel, FoldAwareAE
from models.modelling_process import ModellingProcess
import joblib  # Assuming pretrained models are saved as .pkl files


# Data configuration
DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.95,
    'gene_type': 'intersection',
    'use_imputed': True,
    'use_cohorts': False, 
    'select_random' : False, 
    'requires_ohenc' : True, 
    'only_pData': False, 
    'clinical_covs' : ["AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA']

}

# Model configuration
MODEL_CONFIG = {
    'params_cv': {
        'model__n_estimators': [1],
        'model__min_samples_split': [6], 
        'model__max_features': ['sqrt'],
        'model__bootstrap' : [False], 
        'model__n_jobs': [-1], 
        'model__random_state': [1234], 
        'model__low_memory' : [True] 
    },
    'refit': False, 
    'do_nested_resampling': True, 
    'path' : RESULTS_DIR, 
    'fname_cv' : 'test'}

mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT)

from sklearn.compose import ColumnTransformer

# Create the dynamic model selector
dynamic_selector = FoldAwareSelectFromModel(estimator=GradientBoostingSurvivalAnalysis(), threshold = "mean")
#dynamic_selector = SelectFromModel(pretrained_gb)
pdata_cols = ['TISSUE_FFPE', 'TISSUE_Fresh_frozen', 'TISSUE_Snap_frozen', 'AGE',
       'GLEASON_SCORE', 'PRE_OPERATIVE_PSA']
exprs_cols =  list(set(mp.X.columns) - set(pdata_cols))

ae = FoldAwareAE()
preprocessor = ColumnTransformer(
    transformers=[
        ('feature_selection', ae, exprs_cols),  # Apply feature selection
        ('other_features', 'passthrough', pdata_cols)         # Pass through other columns
    ]
)


# Define the pipeline
pipe_steps = [
    ('preprocessor', preprocessor),
    ('model', RandomSurvivalForest(n_estimators = 1))]

mp.do_modelling(pipe_steps, MODEL_CONFIG)
