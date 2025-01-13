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

# Imports
from models.modelling_process import ModellingProcess
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from utils.evaluation import EarlyStoppingMonitor
from sklearn.base import clone
import joblib  

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def remove_cohort(X, y, c):
    X['c'] = X.index.to_series().str.split('.').str[0]
    indices = X[X['c'] != c].index
    numeric_indices = [X.index.get_loc(idx) for idx in indices]
    filtered_X = X.loc[indices].drop(['c'], axis = 1)
    filtered_y = y[numeric_indices]
    return filtered_X, filtered_y
    

def fit_feature_sel(estimator, X, y, monitor, current_c = ''): 

    cohorts = X.index.to_series().str.split('.').str[0].unique()
    print(current_c)
    for c in cohorts: 
        print(c)
        estimator_c = clone(estimator)
        X_tmp, y_tmp = remove_cohort(X, y, c)
        print(type(X_tmp))
        
        estimator_c.fit(X_tmp, y_tmp, monitor = monitor)
        print(estimator_c)
        if current_c == '': 
            c_path = c + '.pkl'
            joblib.dump(estimator_c, c_path)
        else: 
            c_path = current_c + '_' + c + '.pkl'
            joblib.dump(estimator_c, c_path)


monitor = EarlyStoppingMonitor(10, 5)

# Data configuration
DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.85,
    'gene_type': 'intersection',
    'use_imputed': False,
    'select_random' : False, 
    'use_cohorts': False
}

mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT)

# Autoencoder
est = GradientBoostingSurvivalAnalysis(n_estimators=500, learning_rate=0.1, random_state=1234, max_depth = 3, max_features = 'sqrt', min_samples_leaf= 10, min_samples_split =  4, subsample = 0.8)

# outer round of feature sel
X, y = mp.X, mp.y
#fit_feature_sel(est, X, y, monitor)
cohorts = X.index.to_series().str.split('.').str[0].unique()

print("Outer round done!")

# inner round of feature sel (to cohorts missing)
for c in cohorts: 
    X_tmp, y_tmp = remove_cohort(X, y, c)
    fit_feature_sel(est, X_tmp, y_tmp, monitor, c)

print("Inner round done")

# complete pretrained model 
est_final = clone(est)
est_final.fit(X, y, monitor = monitor)
joblib.dump(est_final, 'pretrnd_cmplt.pkl')