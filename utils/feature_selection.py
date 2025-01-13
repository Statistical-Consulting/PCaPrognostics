from sklearn.feature_selection import SelectFromModel, SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
import joblib  
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import torch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split
# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self, inp_dim, latent_dim=20):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, inp_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Get latent representation
        latent = self.encoder(x)

        # Reconstruct input
        reconstructed = self.decoder(latent)

        # Reshape to original dimensions
        return reconstructed


class FoldAwareSelectFromModel(SelectFromModel, SelectorMixin):
    def __init__(self, estimator, threshold = "median"):
        self.all_cohorts = ['Atlanta_2014_Long', 'Belfast_2018_Jain', 'CamCap_2016_Ross_Adams',
 'CancerMap_2017_Luca', 'CPC_GENE_2017_Fraser', 'CPGEA_2020_Li',
 'DKFZ_2018_Gerhauser', 'MSKCC_2010_Taylor', 'Stockholm_2016_Ross_Adams']
        super().__init__(estimator=estimator, threshold=threshold)
        self.estimator = estimator
        #self.threshold_ = threshold

    
    @_fit_context(
    # SelectFromModel.estimator is not validated yet
    prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **fit_params):
        # Example logic to choose a model based on the data split
        root = os.path.dirname(os.path.dirname(__file__))
        root = os.path.join(root, 'pretrnd_models')
        #if root not in sys.path:
        #    sys.path.append(root)
        print(root)
        cohort_names = X.index.to_series().str.split('.').str[0]
        # Get unique cohort names
        unique_cohort_names = cohort_names.unique()
        model_path = ''
        for c in self.all_cohorts: 
            if c not in unique_cohort_names: 
                if len(model_path) > 0: 
                    model_path +=  "_"
                model_path += c  
        if model_path == '': 
            model_path = 'pretrnd_cmplt'
        model_path = os.path.join(root, model_path)
        print(model_path)
        self.estimator= joblib.load(model_path + '.pkl')  
        #super().fit(X, y, **fit_params)  # No need to fit as models are pretrained
        return self
    

class FoldAwareAE(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.all_cohorts = ['Atlanta_2014_Long', 'Belfast_2018_Jain', 'CamCap_2016_Ross_Adams',
 'CancerMap_2017_Luca', 'CPC_GENE_2017_Fraser', 'CPGEA_2020_Li',
 'DKFZ_2018_Gerhauser', 'MSKCC_2010_Taylor', 'Stockholm_2016_Ross_Adams']
        self.estimator = None
        #self.threshold_ = threshold

    
    #@_fit_context(
    # SelectFromModel.estimator is not validated yet
    #prefer_skip_nested_validation=False
    #)
    def fit(self, X, y=None):
        # Example logic to choose a model based on the data split
        #self.estimator= joblib.load(model_path + '.pkl')  
        #super().fit(X, y, **fit_params)  # No need to fit as models are pretrained
        #print("new transform --------------------------------")
        # Ensure the estimator is fitted
        root = os.path.dirname(os.path.dirname(__file__))
        root = os.path.join(root, 'pretrnd_models_ae', 'models', 'models')
        #if root not in sys.path:
        #    sys.path.append(root)
        #print(X.info())
        cohort_names = X.index.to_series().str.split('.').str[0]
        # Get unique cohort names
        unique_cohort_names = cohort_names.unique()
        print(unique_cohort_names)
        model_path = ''
        for c in self.all_cohorts: 
            #print('--------------')
            #print(c)
            if c not in unique_cohort_names: 
                if len(model_path) > 0: 
                    model_path +=  "_"
                model_path += c 
            #print('Model path') 
            #print(model_path)
        if model_path == '': 
            model_path = 'pretrnd_cmplt'
        model_path = os.path.join(root, model_path)
        #self.estimator= joblib.load(model_path + '.pkl')  
        #self.estimator = torch.load(model_path, map_location=torch.device("cpu"))
        
        self.model = Autoencoder(inp_dim=len(X.columns))

        # Load the saved state_dict
        self.model.load_state_dict(torch.load(model_path + '.pth'))

        # Set the model to evaluation mode if needed
        self.model.eval()

        #self.X_rdcd = pd.read_csv(model_path + '.csv', index_col=0)
        #self.X_rdcd = pd.DataFrame(self.X_rdcd)
        return self
    
    def transform(self, X, y = None):
        # Create AE s.t. this is the case
        X_t = torch.FloatTensor(X.values).to(device)
        #X = TensorDataset(X)
        ls = self.model.encoder(X_t).detach().cpu().numpy()
        ls = pd.DataFrame(ls, index=X.index)
        return ls
    
    def fit_transform(self, X, y=None, **fit_params):
        print("fit_transform")
        self.fit(X, y, **fit_params)
        print(X.info())
        return self.transform(X)