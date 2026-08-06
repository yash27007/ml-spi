import os
import sys
import dill
from math import prod
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from spi.exception import CustomException

def save_object(file_path: str, obj) -> None:
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    except Exception as error:
        raise CustomException(error, sys) from error


def evaluate_model(X_test, y_test, models: dict) -> dict:
    """Evaluate already-fitted regressors on held-out test data."""
    try:
        report = {}

        for model_name, model in models.items():
            y_test_pred = model.predict(X_test)
            report[model_name] = {
                "r2_score": float(r2_score(y_test, y_test_pred)),
                "mae": float(mean_absolute_error(y_test, y_test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            }

        return report
    except Exception as error:
        raise CustomException(error, sys) from error


def tune_models(
    X_train,
    y_train,
    models: dict,
    param_distributions: dict,
    cv: int = 5,
    n_iter: int = 10,
) -> tuple[dict, dict]:
    """Tune each regressor with cross-validation using only training data.

    RandomizedSearchCV refits each returned estimator on the full training set.
    The held-out test set is intentionally not accepted by this function.
    """
    try:
        tuned_models = {}
        tuning_report = {}
        cross_validation = KFold(n_splits=cv, shuffle=True, random_state=42)

        for model_name, model in models.items():
            search_space_size = prod(
                len(values) for values in param_distributions[model_name].values()
            )
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions[model_name],
                n_iter=min(n_iter, search_space_size),
                scoring="r2",
                cv=cross_validation,
                n_jobs=-1,
                random_state=42,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)

            tuned_models[model_name] = search.best_estimator_
            tuning_report[model_name] = {
                "cv_r2_score": float(search.best_score_),
                "best_params": search.best_params_,
            }

        return tuned_models, tuning_report
    except Exception as error:
        raise CustomException(error, sys) from error
