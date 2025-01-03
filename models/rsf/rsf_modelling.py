import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Setup directories
MODEL_DIR = os.path.join(Path(__file__).parent.resolve(), 'model')
RESULTS_DIR = os.path.join(Path(__file__).parent.resolve(), 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Imports
from preprocessing.data_container import DataContainer
from utils.evaluation import cindex_score
from models.modelling_process import ModellingProcess
from sksurv.ensemble import RandomSurvivalForest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data configuration
DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.95,
    'gene_type': 'common_genes',
    'use_imputed': True,
    'use_cohorts': False,
    'select_random': False, 
    'requires_ohenc' : True, 
    'clinical_covs' : ['AGE', 'TISSUE', 'GLEASON_SCORE', 'PATH_T_STAGE']
}

# Model configuration
MODEL_CONFIG = {
    'params_cv': {
        'model__n_estimators': [50, 60, 70],
        'model__min_samples_split': [12],
        'model__max_features': ['sqrt', 'log2'],
        'model__bootstrap': [True],
        'model__max_samples' : [0.4], 
        #'model__n_jobs': [-1],
        'model__random_state': [1234],
        #'model__low_memory': [True], 
        'model__warm_start' : [True]
    },
    'refit': True,
    'do_nested_resampling': False,
    'path': RESULTS_DIR,
    'fname_cv': 'results_common'
}

rsf_pipeline_steps = [('model', RandomSurvivalForest())]

# Initialize and run modelling process
mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT)
nstd_res_result = mp.do_modelling(rsf_pipeline_steps, MODEL_CONFIG)

# Optional: Save results
# if nstd_res_result is not None:
#     mp.save_results(
#         path=RESULTS_DIR,
#         fname='rsf_model',
#         model=nstd_res_result[1],
#         cv_results=nstd_res_result[0],
#         pipe=nstd_res_result[2]
#     )