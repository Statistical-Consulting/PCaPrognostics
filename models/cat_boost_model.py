import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from catboost import CatBoostRegressor, Pool


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



class CatBoostModel(BaseEstimator, RegressorMixin): 
    def __init__(self, cat_features = ['TISSUE'], 
                 iterations = None, loss_function = "Cox", eval_metric = "Cox", early_stopping_rounds = 5, 
                 rsm = 0.1, depth = None, min_data_in_leaf = None, learning_rate = 0.1): 
        super(CatBoostModel, self).__init__()
        self.cat_features = cat_features
        self.is_fitted_ = False
        self.iterations=iterations
        self.loss_function=loss_function
        self.eval_metric=eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.bootstrap_type='Bernoulli'
        self.boosting_type = 'Plain'
        self.rsm = rsm 
        self.depth = depth
        self.min_data_in_leaf = min_data_in_leaf
        self.learning_rate = learning_rate
        
        
    def _prepare_data(self, X, y):
        y = pd.DataFrame(y)
        if self.loss_function == 'Cox': 
            y['label'] = np.where(y['status'], y['time'], - y['time'])
            y_fin = y['label']
        # TODO: Include other loss
        else: 
            y['y_lower'] = y['time']
            y['y_upper'] = np.where(y['status'], y['time'], -1)
            y_fin = y.loc[:,['y_lower','y_upper']]
        
        for col in self.cat_features:
            X.loc[:, col] = X.loc[:,col].astype('category')
        
        #data = pd.concat([X, y], dim = 1)
        #print(data.info())
        return X, y_fin
    
    def fit(self, X, y): 
        # early stopping mit 0.1 des training sets
        X, y = self._prepare_data(X, y)

        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1)

        #train_pool = Pool(train[features], label=train['label'], cat_features=cat_features)
        #val_pool = Pool(val[features], label=test['label'], cat_features=cat_features)

        self.model = CatBoostRegressor(iterations=self.iterations,
                        loss_function=self.loss_function,
                        depth = self.depth, 
                        eval_metric=self.eval_metric,
                        learning_rate=self.learning_rate, 
                        early_stopping_rounds = self.early_stopping_rounds,
                        bootstrap_type=self.bootstrap_type, 
                        boosting_type=self.boosting_type,
                        min_data_in_leaf = self.min_data_in_leaf,
                        rsm = self.rsm, 
                        cat_features=self.cat_features, 
                        random_seed=1234)
        
        self.model.fit(X = train_X, y = train_y, eval_set= (val_X, val_y), verbose = False)
        self.is_fitted_ = True
        return self

    def predict(self, X): 
        check_is_fitted(self, 'is_fitted_')
        train_y_pred = self.model.predict(X)
        return train_y_pred
    
    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        preds = self.predict(X)
        ci = concordance_index(y['time'], -preds, y['status'])
        return ci
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def clone(self): 
        super(self).clone()
    
    def get_feature_importance(self, X, y): 
        X, y = self._prepare_data(X, y)
        check_is_fitted(self, 'is_fitted_')
        data = Pool(X, label=y, cat_features=self.cat_features)
        imp = self.model.get_feature_importance(data=data)
        return imp

