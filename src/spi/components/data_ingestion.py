from dataclasses import dataclass
import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from spi.exception import CustomException
from spi.components.data_transform import DataTransformation
from spi.components.model_trainer import ModelTrainerConfig
from spi.components.model_trainer import ModelTrainer
logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logger.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Train-test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info("Data ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as error:
            logger.exception("Data ingestion failed")
            raise CustomException(error, sys) from error

if __name__ == "__main__":
    from spi.logger import configure_logging
    configure_logging()

    obj = DataIngestion()
    train_data,test_data =  obj.inititate_data_ingestion()

    data_transformation = DataTransformation()
    train_array,test_array,preprocessor_path=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_array=train_array,test_array=test_array,preprocessor_path=preprocessor_path)
    print(r2_score)
    