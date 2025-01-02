import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_survival_curves(model, X_test, y_test, groups=None, n_curves=5):
    np.random.seed(2134)
    """Plot predicted survival curves for selected samples"""
    if hasattr(model, 'predict_survival_function'):
        plt.figure(figsize=(10, 6))

        # Select samples to plot
        if n_curves < len(X_test):
            idx = np.random.choice(len(X_test), n_curves, replace=False)
            X_subset = X_test.iloc[idx]
            y_subset = y_test[idx]
        else:
            X_subset = X_test
            y_subset = y_test

        # Plot predicted curves
        for i, (_, x) in enumerate(X_subset.iterrows()):
            surv_funcs = model.predict_survival_function(x.values.reshape(1, -1))
            time_points = surv_funcs[0].x  # Get time points from survival function
            survival_probs = surv_funcs[0].y  # Get probabilities from survival function
            plt.plot(time_points, survival_probs, label=f'Patient {i + 1}')

            # Plot actual event if it occurred
            if y_subset[i]['status']:
                plt.scatter(y_subset[i]['time'], 0, color='red', marker='x')

        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Predicted Survival Curves')
        plt.grid(True)
        plt.legend()

    else:
        print("Model does not support survival function prediction")


def plot_cv_results(results_df, metric='test_score'):
    """Plot cross-validation results comparison"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='model', y=metric)
    plt.xticks(rotation=45)
    plt.title(f'Cross-validation {metric} by Model')
    plt.tight_layout()


def plot_feature_importance(importance_df, top_n=20):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    importance_df.head(top_n).plot(kind='barh')
    plt.title(f'Top {top_n} Important Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()