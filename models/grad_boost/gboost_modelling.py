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
RESULTS_DIR = os.path.join(Path(__file__).parent.resolve(), 'results_new')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Imports
import utils.evaluation
from preprocessing.data_container import DataContainer
from utils.evaluation import cindex_score
from models.modelling_process import ModellingProcess
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from utils.evaluation import EarlyStoppingMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#set early stopping monitor 
monitor = EarlyStoppingMonitor(10, 5)

# Data configuration
DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.85,
    'gene_type': 'intersection',
    'use_imputed': True,
    'select_random' : False, 
    'use_cohorts': False
}

# Model configuration
# MODEL_CONFIG = {
#     'params_cv': {
#         'model__n_estimators': [500],
#         'model__learning_rate': [0.1],
#         'model__max_depth': [3, 5],
#         'model__min_samples_split': [5, 10],
#         'model__min_samples_leaf': [3, 5],
#         'model__subsample': [0.9],
#         'model__max_features': ['sqrt'], 
#         'model__n_iter_no_change' : [10], 
#         'model__validation_fraction' : [0.1]
#     },
#     'refit': True, 
#     'do_nested_resampling': False, 
#     #'monitor' : monitor, 
#     'path' : RESULTS_DIR, 
#     'fname_cv' : 'test'}

# # Model configuration
MODEL_CONFIG = {
    'params_cv': {
        'model__n_estimators': [500],
        'model__learning_rate': [0.1],
        'model__max_depth': [3, 5, 10, 15],
        'model__min_samples_split': [2, 4, 10, 24],
        'model__min_samples_leaf': [3, 5, 10],
        'model__subsample': [0.9],
        'model__max_features': ['sqrt', 'log2'], 
        #'model__n_iter_no_change' : [5, 10], 
        #'model__validation_fraction' : [0.1]
    },
    'refit': True, 
    'do_nested_resampling': True, 
    'monitor' : monitor, 
    'path' : RESULTS_DIR, 
    'fname_cv' : 'test'}

# validation_fraction=0.1 as a mean to inclued early stopping
gb_pipeline_steps = [('model', GradientBoostingSurvivalAnalysis())]


mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT)

nstd_res_result = mp.do_modelling(gb_pipeline_steps, MODEL_CONFIG)
#mp.save_results(RESULTS_DIR, 'gb_intersect_imp_done', model = mp.cmplt_model, cv_results = mp.resampling_cmplt, pipe = mp.cmplt_pipeline)