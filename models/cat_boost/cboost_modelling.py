import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from catboost import CatBoostRegressor, Pool

# Setup paths for Windows
PROJECT_ROOT =  os.getcwd()

# Set up path for Mac
# PROJECT_ROOT = str(Path(os.getcwd()).parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    

# Setup directories
RESULTS_DIR = os.path.join(Path(__file__).parent.resolve(), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Imports
from preprocessing.data_container import DataContainer
from utils.evaluation import cindex_score
from models.modelling_process import ModellingProcess
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from utils.evaluation import EarlyStoppingMonitor
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from models.cat_boost_model import CatBoostModel
from utils.feature_selection import FoldAwareSelectFromModel
from sklearn.compose import ColumnTransformer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.85,
    'gene_type': 'intersection',
    'use_imputed': True,
    'select_random' : False, 
    'use_cohorts': False, 
    'requires_ohenc' : False, 
    # Auf True wenn NUR pDaten verwendet werden sollen
    'only_pData': False,
    # Nur benötigt wenn pDaten mitgefitten werden sollen
    # 'clinical_covs' : ["AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA']
}

mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT) 

# pipe steps für modelle ohne pDaten
# pipe_steps = [('model', CatBoostModel(cat_features = None))]

# pipe steps für modelle mit pDaten
# pipe_steps = [('model', CatBoostModel())]


# # Model configuration
MODEL_CONFIG = {
    'params_cv': {
        'model__iterations': [500],
        'model__learning_rate': [0.1],
        'model__depth': [3, 5, 10],
        'model__min_data_in_leaf': [3, 5, 10],
        'model__nan_mode' : ["Forbidden"], 
        'model__rsm' : [None, 0.1]
        },
    'refit': False, 
    'do_nested_resampling': True, 
    'path' : RESULTS_DIR, 
    # TODO: WICHTIG: Ändern pro Modell; Dateiname des Modells
    'fname_cv' : 'cboost_autoencoder_paper'}


# MODEL_CONFIG = {
#     'params_cv': {
#         'model__iterations': [500],
#         'model__learning_rate': [0.1],
#         'model__depth': [3, 5],
#         'model__min_data_in_leaf': [1, 3, 5, 10],
#         'model__nan_mode' : ["Forbidden"], 
#         'model__rsm' : [0.2]
#         },
#     'refit': True, 
#     'do_nested_resampling': True, 
#     'path' : RESULTS_DIR, 
#     # TODO: WICHTIG: Ändern pro Modell; Dateiname des Modells
#     'fname_cv' : 'cboost_pData'}


from sklearn.compose import ColumnTransformer
from utils.feature_selection import FoldAwareSelectFromModel, FoldAwareAE


# Create the dynamic model selector
#dynamic_selector = FoldAwareSelectFromModel(estimator=GradientBoostingSurvivalAnalysis(), threshold = "mean")
#dynamic_selector = SelectFromModel(pretrained_gb)
#pdata_cols = ['TISSUE', 'AGE',
#       'GLEASON_SCORE', 'PRE_OPERATIVE_PSA']
pdata_cols = []
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
    ('model', CatBoostModel(from_autoenc_exprs= True, cat_features=None))]

mp.do_modelling(pipe_steps, MODEL_CONFIG)

#nstd_res_result = mp.do_modelling(pipe_steps, MODEL_CONFIG)
#import torch
#mp.cmplt_pipeline.to(torch.device('cpu'))
#torch.save(mp.cmplt_pipeline, os.path.join(PROJECT_ROOT, RESULTS_DIR, 'pipe', "cboost_autoencoder_pData_paper.pth"))