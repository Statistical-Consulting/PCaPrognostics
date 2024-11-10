
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from utils.evaluation import cindex_score
from utils.resampling import NestedResamplingCV


class BaseSurvivalModel(BaseEstimator):
    def __init__(self, config=None):
        self.config = config
        self.is_fitted = False
        self.model = None
        self.pipe_best_mod = None  # Nur für Pipeline-basierte Modelle (RSF)
        self.preprocessing = None
        self.cv_results_ = None

    def _validate_survival_data(self, y):
        # Diese Methode kann bleiben wie sie ist
        pass

    def _create_param_grid(self, params_cv, model_prefix=None):
        """Convert parameter grid to sklearn format"""
        if not isinstance(params_cv, dict):
            raise ValueError("params_cv must be a dictionary")
        # Optional prefix für Pipeline-basierte Modelle
        if model_prefix:
            return {f'{model_prefix}__{key}': value for key, value in params_cv.items()}
        return params_cv

    def fit_model(self, X, y, groups=None, fname=None, path=None,
                  pipeline_steps=None, params_cv=None,
                  use_cohort_cv=True, n_splits_inner=5,
                  random_search=False, n_iter=100,
                  early_stopping=False, patience=10,
                  parallel=False, refit=True,
                  **model_specific_params):
        """
        Generic fit method that handles both pipeline-based (RSF)
        and direct models (DeepSurv)
        """
        self._validate_survival_data(y)

        if hasattr(self, 'use_pipeline') and self.use_pipeline:
            # Pipeline-basiertes Training (RSF)
            if pipeline_steps is None:
                raise ValueError("pipeline_steps required for pipeline-based models")

            if params_cv is not None:
                param_grid = self._create_param_grid(params_cv, model_prefix='rsf')
            else:
                param_grid = None

            pipe = Pipeline(pipeline_steps)

            if param_grid is not None:
                cv = NestedResamplingCV(
                    n_splits_inner=n_splits_inner,
                    use_cohort_cv=use_cohort_cv,
                    random_state=42,
                    n_jobs=-1 if parallel else None
                )

                cv_results = cv.fit(
                    estimator=pipe,
                    X=X,
                    y=y,
                    groups=groups,
                    param_grid=param_grid,
                    scoring=cindex_score
                )

                self.cv_results_ = cv_results

                if refit:
                    best_params = cv_results['fold_results'][0]['best_params']
                    pipe.set_params(**best_params)
                    pipe.fit(X, y)
                    self.pipe_best_mod = pipe
                    self.model = pipe.named_steps['rsf']
                    self.is_fitted = True

            else:
                pipe.fit(X, y)
                self.pipe_best_mod = pipe
                self.model = pipe.named_steps['rsf']
                self.is_fitted = True

        else:
            # Direktes Training (DeepSurv)
            self.model = self._fit_direct(
                X=X,
                y=y,
                groups=groups,
                validation_data=model_specific_params.get('validation_data'),
                early_stopping=early_stopping,
                patience=patience,
                **model_specific_params
            )
            self.is_fitted = True

        # Save results
        if fname is not None and path is not None:
            if hasattr(self, 'cv_results_'):
                self.save_cv_results(self.cv_results_, path, fname)
            self.save_model(path, fname)

        return self

    def _fit_direct(self, X, y, **kwargs):
        """
        Implementiert in konkreten Modellklassen (DeepSurv)
        """
        raise NotImplementedError

    def predict(self, X):
        """Generic predict method"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        if hasattr(self, 'use_pipeline') and self.use_pipeline:
            return self.pipe_best_mod.predict(X)
        else:
            return self.model.predict(X)

    def save_model(self, path, fname):
        """Save fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        os.makedirs(os.path.join(path, 'model'), exist_ok=True)
        os.makedirs(os.path.join(path, 'results'), exist_ok=True)

        # Pipeline-basierte Modelle (RSF)
        if hasattr(self, 'use_pipeline') and self.use_pipeline:
            with open(os.path.join(path, 'model', f"{fname}.pkl"), 'wb') as f:
                pickle.dump(self.pipe_best_mod, f)
        # Direkte Modelle (DeepSurv)
        else:
            self._save_direct_model(path, fname)

        # CV results speichern
        if hasattr(self, 'cv_results_'):
            with open(os.path.join(path, 'results', f"{fname}_cv_results.pkl"), 'wb') as f:
                pickle.dump(self.cv_results_, f)

    def _save_direct_model(self, path, fname):
        """
        Implementiert in konkreten Modellklassen (DeepSurv)
        """
        raise NotImplementedError

    def save_cv_results(self, cv_results, path, fname):
        """Save cross-validation results"""
        os.makedirs(path, exist_ok=True)
        pd.DataFrame(cv_results).to_csv(
            os.path.join(path, f"{fname}_cv_results.csv")
        )