"""
utils_hyperopt.py
=================
Hyperparameter search grids and a RandomizedSearchCV wrapper
for Logistic Regression, Random Forest, XGBoost, and LightGBM.

Usage (in a script or notebook)
--------------------------------
    from utils_hyperopt import get_param_grids, run_hyperopt
    from sklearn.ensemble import RandomForestClassifier

    grid = get_param_grids()["rf"]
    best_model, best_params, cv_results = run_hyperopt(
        RandomForestClassifier(n_jobs=-1, random_state=42),
        X_train, y_train,
        param_grid=grid,
        n_iter=20,
        cv=3,
        scoring="f1",   # or "f1_macro" for multi-class
    )
"""

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform


def get_param_grids() -> dict:
    """Return per-model hyperparameter search spaces."""
    return {
        "lr": {
            "C": loguniform(1e-3, 1e3),
            "max_iter": [200, 500, 1000],
            "solver": ["lbfgs", "saga"],
            "class_weight": [None, "balanced"],
        },
        "rf": {
            "n_estimators": randint(100, 501),
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 5),
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced"],
        },
        "xgb": {
            "n_estimators": randint(100, 501),
            "max_depth": randint(3, 10),
            "learning_rate": loguniform(1e-3, 0.3),
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": loguniform(1e-4, 10),
            "reg_lambda": loguniform(1e-4, 10),
        },
        "lgbm": {
            "n_estimators": randint(100, 501),
            "max_depth": randint(3, 10),
            "learning_rate": loguniform(1e-3, 0.3),
            "num_leaves": randint(20, 150),
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": loguniform(1e-4, 10),
            "reg_lambda": loguniform(1e-4, 10),
            "class_weight": [None, "balanced"],
        },
    }


def run_hyperopt(
    estimator,
    X_train,
    y_train,
    param_grid: dict,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1",
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1,
):
    """
    Run RandomizedSearchCV and return the best estimator,
    best params, and the full cv_results_ dataframe.

    Parameters
    ----------
    estimator      : sklearn-compatible estimator (unfitted)
    X_train        : feature matrix
    y_train        : target vector
    param_grid     : dict of distributions / lists (from get_param_grids())
    n_iter         : number of random samples
    cv             : number of CV folds
    scoring        : sklearn scoring string ('f1', 'f1_macro', 'accuracy', …)
    n_jobs         : parallel jobs (-1 = all cores)
    random_state   : seed for reproducibility
    verbose        : verbosity level

    Returns
    -------
    best_estimator : fitted estimator with best params
    best_params    : dict of best hyperparameter values
    cv_results     : pandas DataFrame of all search results
    """
    import pandas as pd

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return search.best_estimator_, search.best_params_, cv_results


if __name__ == "__main__":
    print("Available model keys:", list(get_param_grids().keys()))
