"""
Resampling Module für Cross-Validation und Hyperparameter Tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
from itertools import product
import logging
from utils.evaluation import cindex_score


logger = logging.getLogger(__name__)


def _get_survival_subset(y, indices):
    """Extract survival data subset while preserving structure"""
    subset = np.empty(len(indices), dtype=y.dtype)
    event_field = 'status' if 'status' in y.dtype.names else 'event'
    subset[event_field] = y[event_field][indices]
    subset['time'] = y['time'][indices]
    return subset

def _aggregate_results(results):
    """Aggregates nested CV results."""
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

def nested_resampling(estimator, X, y, groups, param_grid, monitor = None, ss = GridSearchCV, outer_cv = LeaveOneGroupOut(), inner_cv = LeaveOneGroupOut(), scoring = None):
    logger.info("Starting nested resampling...")
    logger.info(f"Data shape: X={X.shape}, groups={len(np.unique(groups))} unique")

    outer_results = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        logger.info(f"\nOuter fold {i+1}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = _get_survival_subset(y, train_idx)
        y_test = _get_survival_subset(y, test_idx)
        train_groups = groups[train_idx] if groups is not None else None

        test_cohort = groups[test_idx][0] if groups is not None else None
        logger.info(f"Test cohort: {test_cohort}")
        
        inner_gcv = ss(estimator, param_grid, cv = inner_cv, refit = True, n_jobs=-1, verbose = 2)
        if monitor is not None:
            inner_results = inner_gcv.fit(X_train, y_train, groups = train_groups, model__monitor = monitor)
            logger.info(f'number of iterations early stopping: {inner_results.best_estimator_.named_steps['model'].n_estimators_}')
        else: 
            inner_results = inner_gcv.fit(X_train, y_train, groups = train_groups)
        
        inner_cv_results = inner_results.cv_results_
        inner_best_params = inner_results.best_params_
        
        outer_model = inner_results.best_estimator_.named_steps['model']
        test_score = outer_model.score(X_test, y_test)

        logger.info(f"Best parameters: {inner_best_params}")
        logger.info(f"Test score: {test_score:.3f}")

        outer_results.append({
            'test_cohort': test_cohort,
            'test_score': test_score,
            'best_params': inner_best_params,
            'inner_cv_results': inner_cv_results
        })

    return _aggregate_results(outer_results)
 

# class NestedResamplingCV:
#     def __init__(self,
#                  n_splits_inner=5,
#                  use_cohort_cv=True,
#                  random_state=42,
#                  n_jobs=None):
#         """
#         Initialize NestedResamplingCV)
#         """
#         self.n_splits_inner = n_splits_inner
#         self.use_cohort_cv = use_cohort_cv
#         self.random_state = random_state
#         self.n_jobs = n_jobs
#         self.outer_cv = LeaveOneGroupOut()

#     def _get_survival_subset(self, y, indices):
#         """Extract survival data subset while preserving structure"""
#         subset = np.empty(len(indices), dtype=y.dtype)
#         event_field = 'status' if 'status' in y.dtype.names else 'event'
#         subset[event_field] = y[event_field][indices]
#         subset['time'] = y['time'][indices]
#         return subset

#     def _inner_grid_search(self, estimator, X, y, groups, param_grid, inner_cv, scoring):
#         """Inner grid search implementation."""
#         # Parameters are already correctly prefixed by BaseSurvivalModel
#         param_names = sorted(param_grid)
#         param_values = [param_grid[name] for name in param_names]
#         param_combinations = [dict(zip(param_names, v))
#                             for v in product(*param_values)]

#         logger.info(f"Starting inner grid search with {len(param_combinations)} parameter combinations")

#         best_score = float('-inf')
#         best_params = None
#         cv_results = []

#         for params in param_combinations:
#             logger.debug(f"Evaluating parameters: {params}")
#             curr_scores = []

#             for inner_train, inner_val in inner_cv.split(X, groups=groups):
#                 X_inner_train = X.iloc[inner_train]
#                 X_val = X.iloc[inner_val]
#                 y_inner_train = self._get_survival_subset(y, inner_train)
#                 y_val = self._get_survival_subset(y, inner_val)

#                 est = clone(estimator)
#                 est.set_params(**params)

#                 est.fit(X_inner_train, y_inner_train)
#                 pred = est.predict(X_val)
#                 score = scoring(y_val, pred)
#                 curr_scores.append(score)

#             mean_score = np.mean(curr_scores)
#             std_score = np.std(curr_scores)

#             logger.debug(f"Mean CV score: {mean_score:.3f} ± {std_score:.3f}")

#             cv_results.append({
#                 'params': params,
#                 'mean_score': mean_score,
#                 'std_score': std_score,
#                 'scores': curr_scores
#             })

#             if mean_score > best_score:
#                 best_score = mean_score
#                 best_params = params
#                 logger.info(f"New best score: {best_score:.3f} with params: {best_params}")

#         best_model = clone(estimator)
#         best_model.set_params(**best_params)
#         best_model.fit(X, y)

#         return {
#             'best_model': best_model,
#             'best_params': best_params,
#             'best_score': best_score,
#             'cv_results': cv_results
#         }

#     def _evaluate_best_model(self, best_model, X_test, y_test, scoring):
#         """Evaluates best model on test data."""
#         pred = best_model.predict(X_test)
#         test_score = scoring(y_test, pred)
#         return test_score

#     def _aggregate_results(self, results):
#         """Aggregates nested CV results."""
#         scores = [res['test_score'] for res in results]
#         mean_score = np.mean(scores)
#         std_score = np.std(scores)

#         logger.info(f"Aggregated results:")
#         logger.info(f"Mean score: {mean_score:.3f} ± {std_score:.3f}")
#         logger.info(f"Individual scores: {scores}")

#         return {
#             'mean_score': mean_score,
#             'std_score': std_score,
#             'fold_results': results
#         }

#     def fit(self, estimator, X, y, groups, param_grid, scoring):

#         logger.info("Starting nested cross-validation...")
#         logger.info(f"Data shape: X={X.shape}, groups={len(np.unique(groups))} unique")

#         outer_results = []

#         for i, (train_idx, test_idx) in enumerate(self.outer_cv.split(X, y, groups)):
#             logger.info(f"\nOuter fold {i+1}")

#             X_train = X.iloc[train_idx]
#             X_test = X.iloc[test_idx]
#             y_train = self._get_survival_subset(y, train_idx)
#             y_test = self._get_survival_subset(y, test_idx)
#             train_groups = groups[train_idx] if groups is not None else None

#             test_cohort = groups[test_idx][0] if groups is not None else None
#             logger.info(f"Test cohort: {test_cohort}")

#             inner_cv = LeaveOneGroupOut() if self.use_cohort_cv else KFold(
#                 n_splits=self.n_splits_inner,
#                 shuffle=True,
#                 random_state=self.random_state
#             )

#             inner_cv_results = self._inner_grid_search(
#                 estimator=estimator,
#                 X=X_train,
#                 y=y_train,
#                 groups=train_groups,
#                 param_grid=param_grid,
#                 inner_cv=inner_cv,
#                 scoring=scoring
#             )

#             test_score = self._evaluate_best_model(
#                 best_model=inner_cv_results['best_model'],
#                 X_test=X_test,
#                 y_test=y_test,
#                 scoring=scoring
#             )

#             logger.info(f"Best parameters: {inner_cv_results['best_params']}")
#             logger.info(f"Test score: {test_score:.3f}")

#             outer_results.append({
#                 'test_cohort': test_cohort,
#                 'test_score': test_score,
#                 'best_params': inner_cv_results['best_params'],
#                 'inner_cv_results': inner_cv_results['cv_results']
#             })

#         return self._aggregate_results(outer_results)


# class DeepSurvNestedCV:
#     """Nested Cross-Validation implementation specific for DeepSurv"""

#     def __init__(self, n_splits_inner=5, use_cohort_cv=True, random_state=42):
#         self.n_splits_inner = n_splits_inner
#         self.use_cohort_cv = use_cohort_cv
#         self.random_state = random_state
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     def _prepare_data_batch(self, X, y, indices):
#         """Prepare data batch for training"""
#         if isinstance(X, pd.DataFrame):
#             X = X.iloc[indices].values
#         else:
#             X = X[indices]

#         # Get correct subset of survival data
#         time = y['time'][indices]
#         event_field = 'status' if 'status' in y.dtype.names else 'event'
#         events = y[event_field][indices]

#         return (
#             torch.FloatTensor(X).to(self.device),
#             torch.FloatTensor(time).to(self.device),
#             torch.FloatTensor(events).to(self.device)
#         )

#     def _train_fold(self, model, X_train, y_train, X_val, y_val, params):
#         """Train model on one fold"""
#         # Reset/Initialize model
#         model.init_network(X_train.shape[1])
#         model.set_params(**params)

#         n_epochs = params.get('num_epochs', 100)
#         batch_size = max(params.get('batch_size', 32), 4)  # Mindestens Batch-Size 4
#         patience = params.get('patience', 10)

#         best_val_loss = float('inf')
#         no_improve = 0
#         best_weights = None

#         n_samples = len(X_train)
#         n_batches = (n_samples - 1) // batch_size + 1

#         # Wenn zu wenig Samples, kompletten Datensatz als Batch verwenden
#         if n_samples < 4:
#             batch_size = n_samples
#             n_batches = 1

#         for epoch in range(n_epochs):
#             # Training
#             model.network.train()
#             total_loss = 0

#             # Generate random indices for shuffling
#             indices = np.random.permutation(n_samples)

#             for i in range(n_batches):
#                 start_idx = i * batch_size
#                 end_idx = min((i + 1) * batch_size, n_samples)
#                 batch_indices = indices[start_idx:end_idx]

#                 if len(batch_indices) < 2:  # Skip zu kleine Batches
#                     continue

#                 X_batch, time_batch, event_batch = self._prepare_data_batch(
#                     X_train, y_train, batch_indices
#                 )

#                 loss = model._train_step(X_batch, time_batch, event_batch)
#                 total_loss += loss

#             # Validation - hier kompletten Validierungsdatensatz verwenden
#             model.network.eval()
#             with torch.no_grad():
#                 X_val_tensor, time_val_tensor, event_val_tensor = self._prepare_data_batch(
#                     X_val, y_val, np.arange(len(X_val))
#                 )
#                 val_loss = model._negative_log_likelihood(
#                     model.network(X_val_tensor),
#                     time_val_tensor,
#                     event_val_tensor
#                 )

#             # Early stopping check
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_weights = model.network.state_dict().copy()
#                 no_improve = 0
#             else:
#                 no_improve += 1

#             if no_improve >= patience:
#                 break

#         # Load best weights
#         model.network.load_state_dict(best_weights)
#         model.is_fitted = True

#         # Calculate validation score (c-index)
#         val_pred = model.predict(X_val)
#         val_score = cindex_score(y_val, val_pred)

#         return val_score, model

#     def fit(self, estimator, X, y, groups, param_grid):
#         """Perform nested cross-validation"""
#         logger.info("Starting nested cross-validation for DeepSurv...")

#         # Setup CV iterators
#         if self.use_cohort_cv and groups is not None:
#             outer_cv = LeaveOneGroupOut()
#             inner_cv = LeaveOneGroupOut()
#         else:
#             outer_cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
#             inner_cv = KFold(n_splits=self.n_splits_inner, shuffle=True,
#                              random_state=self.random_state)

#         # Generate parameter combinations
#         param_combinations = [dict(zip(param_grid.keys(), v))
#                               for v in product(*param_grid.values())]

#         results = []
#         for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, groups=groups)):
#             logger.info(f"Outer fold {fold_idx + 1}")

#             X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             y_train = y[train_idx]
#             y_test = y[test_idx]

#             if groups is not None:
#                 train_groups = groups[train_idx]
#             else:
#                 train_groups = None

#             # Inner CV for hyperparameter tuning
#             inner_scores = []
#             for params in param_combinations:
#                 fold_scores = []

#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train,
#                                                                      groups=train_groups):
#                     X_inner_train = X_train.iloc[inner_train_idx]
#                     X_inner_val = X_train.iloc[inner_val_idx]
#                     y_inner_train = y_train[inner_train_idx]
#                     y_inner_val = y_train[inner_val_idx]

#                     # Train on fold
#                     inner_model = clone(estimator)
#                     score, trained_model = self._train_fold(
#                         inner_model,
#                         X_inner_train,
#                         y_inner_train,
#                         X_inner_val,
#                         y_inner_val,
#                         params
#                     )
#                     fold_scores.append(score)

#                 inner_scores.append({
#                     'params': params,
#                     'mean_score': np.mean(fold_scores),
#                     'std_score': np.std(fold_scores)
#                 })

#             # Find best parameters
#             best_score_idx = np.argmax([s['mean_score'] for s in inner_scores])
#             best_params = inner_scores[best_score_idx]['params']

#             # Train final model for this fold
#             final_model = clone(estimator)
#             _, trained_model = self._train_fold(
#                 final_model,
#                 X_train,
#                 y_train,
#                 X_test,
#                 y_test,
#                 best_params
#             )

#             # Evaluate on test set
#             test_pred = trained_model.predict(X_test)
#             test_score = cindex_score(y_test, test_pred)

#             results.append({
#                 'fold': fold_idx,
#                 'test_score': test_score,
#                 'best_params': best_params,
#                 'inner_cv_results': inner_scores
#             })

#             logger.info(f"Fold {fold_idx + 1} complete - "
#                         f"Test c-index: {test_score:.3f}")

#         return {
#             'cv_results': results,
#             'mean_score': np.mean([r['test_score'] for r in results]),
#             'std_score': np.std([r['test_score'] for r in results])
#         }