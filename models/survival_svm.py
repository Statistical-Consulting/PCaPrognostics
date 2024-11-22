import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def prepare_survival_data(pdata, exprs):
    """Clean and prepare survival data"""
    mask = pdata['MONTH_TO_BCR'] > 0
    pdata_clean = pdata[mask]
    exprs_clean = exprs.loc[pdata_clean.index]

    status = pdata_clean['BCR_STATUS'].astype(bool).values
    time = pdata_clean['MONTH_TO_BCR'].astype(float).values
    y = Surv.from_arrays(status, time)

    return exprs_clean, y, mask


class SurvivalSVMWrapper(BaseEstimator):
    """Wrapper for FastSurvivalSVM with sklearn compatibility"""

    def __init__(self, alpha=1.0, max_iter=100, rank_ratio=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.rank_ratio = rank_ratio
        self.model = None

    def fit(self, X, y):
        self.model = FastSurvivalSVM(
            alpha=self.alpha,
            max_iter=self.max_iter,
            rank_ratio=self.rank_ratio,
            random_state=42
        )
        self.model.fit(X, y)
        return self

    def score(self, X, y):
        pred = self.predict(X)
        event_field = 'status' if 'status' in y.dtype.names else 'event'
        return concordance_index_censored(
            y[event_field],
            y['time'],
            -pred)[0]

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'rank_ratio': self.rank_ratio,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def train_survival_svm(X, y, param_grid, groups=None):
    """Train SurvivalSVM with nested cross-validation"""
    from utils.resampling import nested_resampling

    # Create pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SurvivalSVMWrapper())
    ])

    # Perform nested resampling with pipeline
    results = nested_resampling(
        estimator=pipe,
        X=X,
        y=y,
        groups=groups,
        param_grid=param_grid
    )

    # Train final model
    best_params = results['fold_results'][-1]['best_params']
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SurvivalSVMWrapper(**{k.replace('model__', ''): v
                                        for k, v in best_params.items()}))
    ])
    final_pipeline.fit(X, y)

    return {
        'best_params': best_params,
        'mean_score': results['mean_score'],
        'std_score': results['std_score'],
        'final_model': final_pipeline,
        'cv_results': results
    }


def create_pipeline_and_param_grid():
    """Create pipeline and parameter grid"""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SurvivalSVMWrapper())
    ])

    param_grid = {
        'model__alpha': [0.1],
        'model__max_iter': [100],
        'model__rank_ratio': [0.5]
    }

    return pipe, param_grid