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
from spi.utils import evaluate_model, save_object

logger = logging.getLogger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logger.info('Split training and test input data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(random_state=42),
                "CatBoost Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_name = max(
                model_report,
                key=lambda model_name: model_report[model_name]["r2_score"],
            )
            best_model_score = model_report[best_model_name]["r2_score"]
            best_model = models[best_model_name]

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
                model_report[best_model_name]["mae"],
                model_report[best_model_name]["rmse"],
            )

            return best_model_score

        except Exception as error:
            logger.exception('Unable to initiate model training')
            raise CustomException(error,sys) from error
