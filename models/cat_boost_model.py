import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

class CatBoostModel(BaseEstimator, RegressorMixin): 
    """
    A custom CatBoost-based survival analysis model compatible with scikit-learn.

    Args:
        cat_features (list, optional): List of categorical feature column names. Default is ['TISSUE'].
        iterations (int, optional): Number of boosting iterations. Default is None.
        loss_function (str, optional): Loss function to use. Default is "Cox" for survival analysis.
        eval_metric (str, optional): Evaluation metric for model validation. Default is "Cox".
        early_stopping_rounds (int, optional): Number of rounds for early stopping. Default is 5.
        rsm (float, optional): Random subspace method fraction. Default is 0.1.
        depth (int, optional): Depth of trees in the model. Default is None.
        min_data_in_leaf (int, optional): Minimum number of data points in a leaf. Default is None.
        learning_rate (float, optional): Learning rate for boosting. Default is 0.1.
        from_autoenc_pdata (bool, optional): Whether the input data comes from an autoencoder with clinical and gene data. Default is False.
        from_autoenc_exprs (bool, optional): Whether the input data comes from an autoencoder with no clincal and only gene data. Default is False.
    
    Attributes:
        model (CatBoostRegressor): Trained CatBoost model.
        is_fitted_ (bool): Indicator if the model has been trained.
    """

    def __init__(self, cat_features = ['TISSUE'], 
                 iterations = None, loss_function = "Cox", eval_metric = "Cox", early_stopping_rounds = 5, 
                 rsm = 0.1, depth = None, min_data_in_leaf = None, learning_rate = 0.1, from_autoenc_pdata = False, from_autoenc_exprs = False): 
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
        self.from_autoenc_pdata = from_autoenc_pdata
        self.from_autoenc_exprs = from_autoenc_exprs
        
        
    def _prepare_data(self, X, y):
        """
        Prepares the dataset for training based on https://github.com/catboost/tutorials/blob/master/regression/survival.ipynb
        Args:
            X (pd.DataFrame): Feature matrix.
            y (structured np.ndarray): Survival labels with 'time' and 'status' fields.

        Returns:
            tuple: Feature matrix X and labels y.
        """
        y = pd.DataFrame(y)
        if self.loss_function == 'Cox': 
            y['label'] = np.where(y['status'], y['time'], - y['time'])
            y_fin = y['label']
        else: 
            y['y_lower'] = y['time']
            y['y_upper'] = np.where(y['status'], y['time'], -1)
            y_fin = y.loc[:,['y_lower','y_upper']]
        
        if (self.from_autoenc_pdata): 
            columns = [str(i) for i in range(64)] + ['TISSUE', 'AGE', 'GLEASON_SCORE', 'PRE_OPERATIVE_PSA']
            X = pd.DataFrame(X, columns=columns)
        if (self.from_autoenc_exprs): 
            columns = [str(i) for i in range(64)]
            X = pd.DataFrame(X, columns=columns)
        
        if self.cat_features is not None: 
            for col in self.cat_features:
                X.loc[:, col] = X.loc[:,col].astype('category')
        

        return X, y_fin
    
    def fit(self, X, y): 
        X, y = self._prepare_data(X, y)

        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1)

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

