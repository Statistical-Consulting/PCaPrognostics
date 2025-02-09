import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
import logging


logger = logging.getLogger(__name__)


def _get_survival_subset(y, indices):
    """
    Extracts a subset of the survival dataset

    Args:
        y (np.ndarray): Structured array containing survival data with fields 'time' and 'status' (or 'event').
        indices: Indices of the subset.

    Returns:
        np.ndarray: A structured array containing the prognostic endpoint of survival data
    """
    subset = np.empty(len(indices), dtype=y.dtype)
    event_field = 'status' if 'status' in y.dtype.names else 'event'
    subset[event_field] = y[event_field][indices]
    subset['time'] = y['time'][indices]
    return subset


def _aggregate_results(results):
    """
    Aggregates nested cross-validation results.

    Args:
        results (list of dict): A list of dictionaries containing results from each CV fold.

    Returns:
        dict: A dict with aggr. infos
    """
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
    """
    Performs nested resampling using Leave-One-Group-Out.

    Args:
        estimator (sklearn estimator/sklearn pipeline): The base model/pipeline 
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Survival outcome data.
        groups (array): Group labels for each sample
        param_grid (dict): Hyperparameter grid for the inner cross-validation.
        monitor (optional): Monitoring parameter for early stopping (default: None).
        ss (class): The search strategy class (default: GridSearchCV).
        outer_cv (sklearn CV splitter): Outer cross-validation splitter (default: LeaveOneGroupOut).
        inner_cv (sklearn CV splitter): Inner cross-validation splitter (default: LeaveOneGroupOut).
        scoring (optional): Scoring function for model evaluation (default: None).

    Returns:
        dict: Aggregated results from nested cross-validation.
    """
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
        
        inner_gcv = ss(estimator, param_grid, cv = inner_cv, refit = True, n_jobs=4, verbose = 2)
        if monitor is not None:
            inner_results = inner_gcv.fit(X_train, y_train, groups = train_groups, model__monitor = monitor)
            logger.info(
                f"Number of iterations early stopping: {inner_results.best_estimator_.named_steps['model'].n_estimators_}")

        else: 
            inner_results = inner_gcv.fit(X_train, y_train, groups = train_groups)
        
        inner_cv_results = inner_results.cv_results_
        inner_best_params = inner_results.best_params_
        
        outer_model = inner_results.best_estimator_
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