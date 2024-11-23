import numpy as np
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold, LeaveOneGroupOut
import pandas as pd

class EarlyStoppingMonitor:
    def __init__(self, window_size, max_iter_without_improvement):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        # continue training for first self.window_size iterations
        if iteration < self.window_size:
            return False

        # compute average improvement in last self.window_size iterations.
        # oob_improvement_ is the different in negative log partial likelihood
        # between the previous and current iteration.
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])

        if improvement > 1e-6:
            self._best_step = iteration
            return False  # continue fitting

        # stop fitting if there was no improvement
        # in last max_iter_without_improvement iterations
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement


def cindex_score(y_true, y_pred):
    """Calculate concordance index for survival data"""
    # Get correct event field name
    event_field = 'status' if 'status' in y_true.dtype.names else 'event'
    time = y_true['time']
    event = y_true[event_field]
    return concordance_index(time, -y_pred, event)


def nested_cv_score(estimator, X, y, groups=None,
                    n_splits_outer=5, n_splits_inner=5,
                    param_grid=None, scoring=cindex_score):
    """Perform nested cross-validation"""

    # Setup CV iterators
    if groups is not None:
        outer_cv = LeaveOneGroupOut()
    else:
        outer_cv = KFold(n_splits=n_splits_outer, shuffle=True)

    inner_cv = KFold(n_splits=n_splits_inner, shuffle=True)

    # Storage for results
    outer_scores = []
    cv_results = []

    # Outer loop
    for train_idx, test_idx in outer_cv.split(X, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner loop for parameter tuning
        inner_scores = []
        for params in param_grid:
            estimator.set_params(**params)
            scores = []
            for inner_train, inner_val in inner_cv.split(X_train):
                X_inner_train = X_train.iloc[inner_train]
                y_inner_train = y_train[inner_train]
                X_val = X_train.iloc[inner_val]
                y_val = y_train[inner_val]

                estimator.fit(X_inner_train, y_inner_train)
                pred = estimator.predict(X_val)
                score = scoring(y_val, pred)
                scores.append(score)

            inner_scores.append({
                'params': params,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            })

        # Find best parameters
        best_score_idx = np.argmax([s['mean_score'] for s in inner_scores])
        best_params = inner_scores[best_score_idx]['params']

        # Train on full training set with best parameters
        estimator.set_params(**best_params)
        estimator.fit(X_train, y_train)

        # Evaluate on test set
        pred = estimator.predict(X_test)
        test_score = scoring(y_test, pred)

        outer_scores.append(test_score)
        cv_results.append({
            'test_score': test_score,
            'best_params': best_params,
            'inner_cv_results': inner_scores
        })

    return {
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'cv_results': cv_results
    }


def save_cv_results(results, path, model_name):
    """Save cross-validation results to CSV"""
    # Flatten nested results
    flat_results = []
    for fold, res in enumerate(results['cv_results']):
        flat_results.append({
            'fold': fold,
            'test_score': res['test_score'],
            'best_params': str(res['best_params']),
            'model': model_name
        })

    df = pd.DataFrame(flat_results)
    df.to_csv(path, index=False)