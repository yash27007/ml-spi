from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

from spi.exception import CustomException
from spi.utils import load_object


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class PredictPipeline:
    """Load persisted artifacts and predict a student's math score."""

    def __init__(self):
        self.model_path = ARTIFACTS_DIR / "model.pkl"
        self.preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_object(str(self.preprocessor_path))
            model = load_object(str(self.model_path))
            transformed_features = preprocessor.transform(features)
            return model.predict(transformed_features)
        except Exception as error:
            raise CustomException(error, sys) from error


@dataclass
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Return one row with the exact feature names used for training."""
        return pd.DataFrame(
            [
                {
                    "gender": self.gender,
                    "race_ethnicity": self.race_ethnicity,
                    "parental_level_of_education": self.parental_level_of_education,
                    "lunch": self.lunch,
                    "test_preparation_course": self.test_preparation_course,
                    "reading_score": self.reading_score,
                    "writing_score": self.writing_score,
                }
            ]
        )
