import os
import sys
from pathlib import Path
import logging

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
from models.modelling_process import ModellingProcess
from models.cat_boost_model import CatBoostModel
from sklearn.compose import ColumnTransformer
from utils.feature_selection import FoldAwareAE


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------- Example for Intersection Data with clinical data
DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.85,
    'gene_type': 'intersection',
    'use_imputed': True,
    'select_random' : False, 
    'use_cohorts': False, 
    'requires_ohenc' : False, 
    'only_pData': False,
    'clinical_covs' : ["AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA'] # --> remove if no clini. Data is wanted
}

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
#     'fname_cv' : 'cboost_inter_pData'}

mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT) 

# Model wo. clin data
# pipe_steps = [('model', CatBoostModel(cat_features = None))] # --> use if no pData is wanted

# Model with clin data
pipe_steps = [('model', CatBoostModel())]

# -------------------------------------- Example Config
MODEL_CONFIG = {
    'params_cv': {
        'model__iterations': [2],
        'model__learning_rate': [0.1],
        'model__depth': [10],
        'model__min_data_in_leaf': [10],
        'model__nan_mode' : ["Forbidden"], 
        'model__rsm' : [0.1]
        },
    'refit': True , 
    'do_nested_resampling': True, 
    'path' : RESULTS_DIR, 
    # TODO: WICHTIG: Ändern pro Modell; Dateiname des Modells
    'fname_cv' : 'test1'}

mp.do_modelling(pipe_steps, MODEL_CONFIG)
# ---------------------------------------------------------- Example for Autoencoder wo. clin. data
DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.85,
    'gene_type': 'intersection',
    'use_imputed': True,
    'select_random' : False, 
    'use_cohorts': False, 
    'requires_ohenc' : False, 
    'only_pData': False,
    # 'clinical_covs' : ["AGE", "TISSUE", "GLEASON_SCORE", 'PRE_OPERATIVE_PSA'] # --> Decomment if clin. Data is wantend
}

mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT) 

# MODEL_CONFIG = {
#     'params_cv': {
#         'model__iterations': [500],
#         'model__learning_rate': [0.1],
#         'model__depth': [3, 5, 10],
#         'model__min_data_in_leaf': [3, 5, 10],
#         'model__nan_mode' : ["Forbidden"], 
#         'model__rsm' : [None, 0.1]
#         },
#     'refit': True , 
#     'do_nested_resampling': True, 
#     'path' : RESULTS_DIR, 
#     'fname_cv' : 'cboost_autoencoder_paper'}

# Clinical data columns 
# pdata_cols = ['TISSUE', 'AGE', 'GLEASON_SCORE', 'PRE_OPERATIVE_PSA'] # --> decomment if pData is wanted
pdata_cols = []
exprs_cols =  list(set(mp.X.columns) - set(pdata_cols))
exprs_cols = sorted(exprs_cols)

ae = FoldAwareAE()
preprocessor = ColumnTransformer(
    transformers=[
        ('feature_selection', ae, exprs_cols),  
        ('other_features', 'passthrough', pdata_cols)         
    ]
)

# Define the pipeline using FoldAwareAE to adequatly respect splits
pipe_steps = [
    ('preprocessor', preprocessor),
    ('model', CatBoostModel(from_autoenc_exprs= True, cat_features=None))]

mp.do_modelling(pipe_steps, MODEL_CONFIG)