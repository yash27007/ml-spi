import os
import sys
import dill
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from spi.exception import CustomException

def save_object(file_path: str, obj) -> None:
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    except Exception as error:
        raise CustomException(error, sys) from error


def evaluate_model(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """Fit each regressor and return its test-set evaluation metrics by name."""
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            report[model_name] = {
                "r2_score": float(r2_score(y_test, y_test_pred)),
                "mae": float(mean_absolute_error(y_test, y_test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            }

        return report
    except Exception as error:
        raise CustomException(error, sys) from error
