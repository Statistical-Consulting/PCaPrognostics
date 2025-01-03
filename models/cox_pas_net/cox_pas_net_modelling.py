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
print(PROJECT_ROOT)
# Setup directories
RESULTS_DIR = os.path.join(Path(__file__).parent.resolve(), 'results_new')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Imports
from preprocessing.data_container import DataContainer
from utils.evaluation import cindex_score
from models.modelling_process import ModellingProcess
from models.cox_pas_net_model import Cox_PASNet_Model
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DATA_CONFIG = {
    'use_pca': False,
    'pca_threshold': 0.95,
    'gene_type': 'intersect',
    'use_imputed': True,
    'use_cohorts': False,
    'select_random': False, 
    'clinical_covs' : ['AGE']
}

dtype = torch.FloatTensor
''' Net Settings'''
In_Nodes = 13214 ###number of genes
Pathway_Nodes = 143 ###number of pathways
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 10 ###number of hidden nodes in the last hidden layer
''' Initialize '''
Learning_Rate = 0.01
L2 = 0
num_epochs = 3000 ###for grid search
Num_EPOCHS = 15 ###for training
###sub-network setup
Dropout_Rate = [0.1, 0.1]
''' load data and pathway '''
pathway_mask = pd.read_csv("models/cox_pas_net/pathway_mask.csv", index_col = 0)


# Model configuration
MODEL_CONFIG = {
    'params_cv': {
        'model__Learning_Rate': [0.01],
        'model__L2': [0], 
        'model__Num_Epochs': [1]
        },
    'refit': False, 
    'do_nested_resampling': True, 
    'path' : RESULTS_DIR, 
    'fname_cv' : 'results_intersect'}

pipeline_steps = [('model', Cox_PASNet_Model(pathway_mask=pathway_mask, clin_covs=['AGE']))]

mp = ModellingProcess()
mp.prepare_data(DATA_CONFIG, PROJECT_ROOT)
mp.do_modelling(pipeline_steps, MODEL_CONFIG)