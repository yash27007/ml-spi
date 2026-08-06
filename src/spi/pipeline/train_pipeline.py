import logging
import sys

from spi.components.data_ingestion import DataIngestion
from spi.components.data_transform import DataTransformation
from spi.components.model_trainer import ModelTrainer
from spi.exception import CustomException
from spi.logger import configure_logging


logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Run the full training workflow from raw data to persisted model."""

    def start_training(self) -> float:
        try:
            logger.info("Starting training pipeline")

            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.inititate_data_ingestion()

            data_transformation = DataTransformation()
            train_array, test_array, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_path=train_data_path,
                    test_path=test_data_path,
                )
            )

            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_array=train_array,
                test_array=test_array,
                preprocessor_path=preprocessor_path,
            )

            logger.info("Training pipeline completed with R2 score %.3f", model_score)
            return float(model_score)
        except Exception as error:
            logger.exception("Training pipeline failed")
            raise CustomException(error, sys) from error


def main() -> None:
    configure_logging()
    score = TrainingPipeline().start_training()
    logger.info("Final training score: %.3f", score)


if __name__ == "__main__":
    main()
