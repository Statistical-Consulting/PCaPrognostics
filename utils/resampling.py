"""
Resampling Module für Cross-Validation und Hyperparameter Tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.base import clone
import logging

logger = logging.getLogger(__name__)

class NestedResamplingCV:
    def __init__(self,
                 n_splits_inner=5,
                 use_cohort_cv=True,
                 random_state=42,
                 n_jobs=None):
        """
        Initialize NestedResamplingCV

        Parameters
        ----------
        n_splits_inner : int
            Number of splits for inner CV (if not cohort-based)
        use_cohort_cv : bool
            Whether to use cohort-based CV for inner loop
        random_state : int
            Random state for reproducibility
        n_jobs : int or None
            Number of parallel jobs (-1 for all cores)
        """
        self.n_splits_inner = n_splits_inner
        self.use_cohort_cv = use_cohort_cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.outer_cv = LeaveOneGroupOut()
        logging.basicConfig(level=logging.INFO)

    def _get_survival_subset(self, y, indices):
        """Extract survival data subset while preserving structure"""
        subset = np.empty(len(indices), dtype=y.dtype)
        # Check which field name is used (status or event)
        event_field = 'status' if 'status' in y.dtype.names else 'event'
        subset[event_field] = y[event_field][indices]
        subset['time'] = y['time'][indices]
        return subset

    def _inner_grid_search(self, estimator, X, y, groups, param_grid, inner_cv, scoring):
        """Führt Grid Search für innere CV durch."""
        from itertools import product

        # Remove pipeline prefix ('rsf__') from parameters
        cleaned_param_grid = {}
        for param_name, param_values in param_grid.items():
            # Remove 'rsf__' prefix if present
            clean_name = param_name.split('__')[-1]
            cleaned_param_grid[clean_name] = param_values

        # Convert param_grid dict to list of all combinations
        param_names = sorted(cleaned_param_grid)
        param_values = [cleaned_param_grid[name] for name in param_names]
        param_combinations = [dict(zip(param_names, v))
                              for v in product(*param_values)]

        logger.info(f"Starting inner grid search with {len(param_combinations)} parameter combinations")

        best_score = float('-inf')
        best_params = None
        cv_results = []

        # Für jede Parameterkombination
        for params in param_combinations:
            logger.debug(f"Evaluating parameters: {params}")
            curr_scores = []

            # Innere CV
            for inner_train, inner_val in inner_cv.split(X, groups=groups):
                # Get training and validation sets
                X_inner_train = X.iloc[inner_train]
                X_val = X.iloc[inner_val]
                y_inner_train = self._get_survival_subset(y, inner_train)
                y_val = self._get_survival_subset(y, inner_val)

                # Clone estimator und setze Parameter
                est = clone(estimator)
                # Get the RSF estimator from pipeline
                if hasattr(est, 'named_steps'):
                    rsf_est = est.named_steps['rsf']
                    rsf_est.set_params(**params)
                else:
                    est.set_params(**params)

                # Fit und evaluate
                est.fit(X_inner_train, y_inner_train)
                pred = est.predict(X_val)
                score = scoring(y_val, pred)
                curr_scores.append(score)

            # Mittlerer Score für diese Parameter
            mean_score = np.mean(curr_scores)
            std_score = np.std(curr_scores)

            logger.debug(f"Mean CV score: {mean_score:.3f} ± {std_score:.3f}")

            cv_results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': curr_scores
            })

            # Update best parameters wenn besser
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                logger.info(f"New best score: {best_score:.3f} with params: {best_params}")

        # Fitte bestes Modell auf allen Trainingsdaten
        best_model = clone(estimator)
        if hasattr(best_model, 'named_steps'):
            best_model.named_steps['rsf'].set_params(**best_params)
        else:
            best_model.set_params(**best_params)
        best_model.fit(X, y)

        # Add pipeline prefix back to parameters for consistency
        best_params_with_prefix = {f'rsf__{k}': v for k, v in best_params.items()}

        return {
            'best_model': best_model,
            'best_params': best_params_with_prefix,
            'best_score': best_score,
            'cv_results': cv_results
        }

    def _evaluate_best_model(self, best_model, X_test, y_test, scoring):
        """Evaluiert bestes Modell auf Test-Daten."""
        pred = best_model.predict(X_test)
        test_score = scoring(y_test, pred)
        return test_score

    def _aggregate_results(self, results):
        """Aggregiert Ergebnisse der nested CV."""
        # Berechne mean und std über alle outer folds
        scores = [res['test_score'] for res in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        logger.info(f"Aggregated results:")
        logger.info(f"Mean score: {mean_score:.3f} ± {std_score:.3f}")
        logger.info(f"Individual scores: {scores}")

        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_results': results
        }

    def fit(self, estimator, X, y, groups, param_grid, scoring):
        """
        Führt nested Cross-Validation durch.

        Parameters
        ----------
        estimator : estimator object
            A scikit-learn estimator with fit and predict methods
        X : pd.DataFrame
            Training data
        y : structured array
            Survival data with time and status/event fields
        groups : array-like
            Group labels for CV splitting
        param_grid : list of dict
            Parameter grid for search
        scoring : callable
            Scoring function

        Returns
        -------
        dict
            Results of nested CV
        """
        logger.info("Starting nested cross-validation...")
        logger.info(f"Data shape: X={X.shape}, groups={len(np.unique(groups))} unique")

        outer_results = []

        # Äußere CV (Leave-One-Cohort-Out)
        for i, (train_idx, test_idx) in enumerate(self.outer_cv.split(X, y, groups)):
            logger.info(f"\nOuter fold {i+1}")

            # Get training and test sets
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = self._get_survival_subset(y, train_idx)
            y_test = self._get_survival_subset(y, test_idx)
            train_groups = groups[train_idx] if groups is not None else None

            # Test cohort info
            test_cohort = groups[test_idx][0] if groups is not None else None
            logger.info(f"Test cohort: {test_cohort}")

            # Inner CV
            if self.use_cohort_cv:
                inner_cv = LeaveOneGroupOut()
            else:
                inner_cv = KFold(
                    n_splits=self.n_splits_inner,
                    shuffle=True,
                    random_state=self.random_state
                )

            # Grid Search auf Training Daten
            inner_cv_results = self._inner_grid_search(
                estimator=estimator,
                X=X_train,
                y=y_train,
                groups=train_groups,
                param_grid=param_grid,
                inner_cv=inner_cv,
                scoring=scoring
            )

            # Beste Parameter auf Test-Kohorte evaluieren
            test_score = self._evaluate_best_model(
                best_model=inner_cv_results['best_model'],
                X_test=X_test,
                y_test=y_test,
                scoring=scoring
            )

            logger.info(f"Best parameters: {inner_cv_results['best_params']}")
            logger.info(f"Test score: {test_score:.3f}")

            # Ergebnisse speichern
            outer_results.append({
                'test_cohort': test_cohort,
                'test_score': test_score,
                'best_params': inner_cv_results['best_params'],
                'inner_cv_results': inner_cv_results['cv_results']
            })

        return self._aggregate_results(outer_results)

    def get_summary(self):
        """Returns a summary of the CV results if available"""
        if not hasattr(self, 'cv_results_'):
            return "No CV results available. Run fit() first."

        summary = pd.DataFrame([{
            'mean_score': self.cv_results_['mean_score'],
            'std_score': self.cv_results_['std_score'],
            'n_folds': len(self.cv_results_['fold_results'])
        }])

        return summary
    def get_summary(self):
        """Erstellt Zusammenfassung der CV Ergebnisse."""
        # Implementation der Zusammenfassung
        pass