import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from spi.exception import CustomException
from spi.logger import logging
from spi.utils import evaluate_model, save_object, tune_models

logger = logging.getLogger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    hyperparameter_tuning_cv: int = 5
    hyperparameter_tuning_n_iter: int = 10

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logger.info('Splitting training and test input data')
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(random_state=42, n_jobs=1),
                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False, random_state=42, thread_count=1
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            param_distributions = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", 1.0],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [2, 3, 5],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.8, 1.0],
                },
                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [False, True],
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                    "leaf_size": [20, 30, 40],
                },
                "XGB Regressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 4, 6],
                    "min_child_weight": [1, 3, 5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_alpha": [0.0, 0.01, 0.1],
                    "reg_lambda": [1.0, 2.0, 5.0],
                },
                "CatBoost Regressor": {
                    "iterations": [200, 400, 600],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8],
                    "l2_leaf_reg": [1, 3, 5, 7],
                    "random_strength": [0, 1, 2],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"],
                },
            }

            logger.info(
                "Tuning %d models with %d-fold cross-validation and %d sampled settings each",
                len(models),
                self.model_trainer_config.hyperparameter_tuning_cv,
                self.model_trainer_config.hyperparameter_tuning_n_iter,
            )
            tuned_models, tuning_report = tune_models(
                X_train=X_train,
                y_train=y_train,
                models=models,
                param_distributions=param_distributions,
                cv=self.model_trainer_config.hyperparameter_tuning_cv,
                n_iter=self.model_trainer_config.hyperparameter_tuning_n_iter,
            )
            for model_name, tuning_result in tuning_report.items():
                logger.info(
                    "%s CV R2=%.3f; best parameters=%s",
                    model_name,
                    tuning_result["cv_r2_score"],
                    tuning_result["best_params"],
                )

            best_model_name = max(
                tuning_report,
                key=lambda model_name: tuning_report[model_name]["cv_r2_score"],
            )
            best_model = tuned_models[best_model_name]
            logger.info("Evaluating selected model '%s' on held-out test data", best_model_name)
            model_report = evaluate_model(
                X_test=X_test,
                y_test=y_test,
                models={best_model_name: best_model},
            )
            best_model_metrics = model_report[best_model_name]
            best_model_score = best_model_metrics["r2_score"]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logger.info(
                "Saved best model '%s': R2=%.3f, MAE=%.3f, RMSE=%.3f",
                best_model_name,
                best_model_score,
                best_model_metrics["mae"],
                best_model_metrics["rmse"],
            )

            return best_model_score

        except Exception as error:
            logger.exception('Unable to initiate model training')
            raise CustomException(error, sys) from error
