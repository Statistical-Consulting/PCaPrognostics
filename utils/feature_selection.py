from sklearn.feature_selection import SelectFromModel, SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
import joblib  
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from typing import Callable

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
The Autoencoder class code is based on the autoencoder implementation
in this repository https://github.com/phcavelar/pathwayae.
"""
class MLP(nn.Module):
    def __init__(
            self,
            input_dim:int,
            hidden_dims:list[int],
            output_dim:int,
            nonlinearity:Callable,
            dropout_rate:float=0.5,
            bias:bool=True,
            ):
        super().__init__()
        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [output_dim]

        self.layers = nn.ModuleList([nn.Linear(d_in, d_out, bias=bias) for d_in, d_out in zip(in_dims, out_dims)])
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.dropout(self.nonlinearity(layer(x)))
        return self.layers[-1](x)

    def layer_activations(self, x:torch.Tensor) -> list[torch.Tensor]:
        # To allow for activation normalisation
        activations = [x]
        for layer in self.layers[:-1]:
            activations.append(self.dropout(self.nonlinearity(layer(activations[-1]))))
        return activations[1:] + [self.layers[-1](activations[-1])]

class NopLayer(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs,
            ):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x

    def update_temperature(self,*args,**kwargs) -> None:
        pass

    def layer_activations(self,*args,**kwargs) -> list[torch.Tensor]:
        return []

class Autoencoder(nn.Module):
    def __init__(
            self,
            input_dim:int=None,
            hidden_dims:list[int]=[128],
            encoding_dim:int=64,
            nonlinearity=F.relu,
            final_nonlinearity=lambda x:x,
            dropout_rate:float=0.5,
            bias:bool=True,
            ):
        super().__init__()
        if input_dim is None:
            raise ValueError("Must specify input dimension before initialising the model")
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]

        self.encoder = MLP(input_dim, hidden_dims, encoding_dim, nonlinearity, dropout_rate, bias)
        self.decoder = MLP(encoding_dim, hidden_dims[-1::-1], input_dim, nonlinearity, dropout_rate, bias)
        self.final_nonlinearity = final_nonlinearity

    def encode(self,x:torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self,x:torch.Tensor) -> torch.Tensor:
        return self.final_nonlinearity(self.decoder(x))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def layer_activations(self,x:torch.Tensor) -> list[torch.Tensor]:
        # To allow for activation normalisation
        encoder_activations = self.encoder.layer_activations(x)
        decoder_activations = self.decoder.layer_activations(encoder_activations[-1])
        return encoder_activations + decoder_activations

    def get_feature_importance_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            feature_importance_matrix = self.encoder.layers[0].weight.T
            for layer in self.encoder.layers[1:]:
                feature_importance_matrix = torch.matmul(feature_importance_matrix, layer.weight.T)
        return feature_importance_matrix.detach()
 
    

class FoldAwareAE(BaseEstimator, TransformerMixin):
    """
    Custom transformer class, based on the transformer class from scikit-learn API. 
    Integrates an Autoencoder model for feature transformation and 
    loads a pretrained autoencoder model based on the cohorts present in the input dataset.
    Transforms the input data into its corresponding latent representations using the encoder of the autoencoder.
    
    Attributes:
        all_cohorts (list): List of all known cohort names.
        model (Autoencoder): Instance of the autoencoder model.
        testing (bool): Whether AE is used during testing or training.
    """
    def __init__(self, testing = False):
        self.all_cohorts = ['Atlanta_2014_Long', 'Belfast_2018_Jain', 'CamCap_2016_Ross_Adams',
                            'CancerMap_2017_Luca', 'CPC_GENE_2017_Fraser', 'CPGEA_2020_Li',
                            'DKFZ_2018_Gerhauser', 'MSKCC_2010_Taylor', 'Stockholm_2016_Ross_Adams']
        self.model = None
        self.testing = testing

    def fit(self, X, y=None):
        """
        Dynamically loads a pretrained autoencoder model based on the cohorts present in the dataset.
        
        Args:
            X (DataFrame): Input dataset with cohort info in the index.
            y (ignored): Included for compatibility with scikit-learn.
        
        Returns:
            self: The fitted instance of the FoldAwareAE class.
        """
        root = os.path.dirname(os.path.dirname(__file__))
        root = os.path.join(root, 'pretrnd_models_ae', 'models')
        if self.testing is False:
            cohort_names = X.index.to_series().str.split('.').str[0]
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
        else: 
            model_path = 'pretrnd_cmplt'
            model_path = os.path.join(root, model_path)
        
        self.model = Autoencoder(input_dim=len(X.columns))
        self.model.load_state_dict(torch.load(model_path + '.pth', map_location=torch.device('cpu')))
        self.model.eval()
        return self
    
    def transform(self, X, y = None):
        """
        Transforms the input data into its corresponding latent representation using the encoder.
        Implemented to adhere to scikit-learn API. 
        
        Args:
            X (DataFrame): Input dataset to be transformed.
            y (ignored): Included for compatibility with scikit-learn.
        
        Returns:
            DataFrame: Latent representation of the input data with original index of X.
        """
        X_t = torch.FloatTensor(X.values).to(device)
        ls = self.model.encoder(X_t).detach().cpu().numpy()
        ls = pd.DataFrame(ls, index=X.index)
        return ls
    
    def fit_transform(self, X, y=None, **fit_params):
        """
        Combines the fit and transform steps. Implemented to adhere to scikit-learn API. 
        
        Args:
            X (DataFrame): Input dataset to be fitted and transformed.
            y (ignored): This parameter is included for compatibility with scikit-learn.
            **fit_params: Additional parameters for the fit method.
        
        Returns:
            DataFrame: Latent representation of the input data after fitting + transforming.
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)