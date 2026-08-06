import sys
import os
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from spi.exception import CustomException
from spi.utils import save_object
logger = logging.getLogger(__name__)

@dataclass
class DataTransformationConfig:
    # The fitted preprocessor is saved so that training and future predictions
    # use exactly the same data-preparation rules.  During fit_transform() on
    # the training set, it learns values such as numerical medians, category
    # mappings for one-hot encoding, and scaling statistics.  Calling fit() on
    # new prediction data would learn different values and make model inputs
    # inconsistent with the data used to train the model.
    #
    # A .pkl file is used because dill serializes the complete fitted sklearn
    # ColumnTransformer object, including its pipelines and learned parameters.
    # It can later be loaded and used with preprocessor.transform(new_data),
    # without fitting it again.
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logger.info(f'Categorical columns: {categorical_columns}')
            logger.info(f'Numerical columns: {numerical_columns}')
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as error:
            logger.exception("Unable to transform data")
            raise CustomException(error,sys) from error


    def initiate_data_transformation(self,train_path:str,test_path:str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info('Read train and test data completed')
            logger.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns=['writing_score','reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logger.info(f'Applying preprocessing object on training dataframe and testing dataframe.')
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            # np.c_ is NumPy shorthand for joining arrays column-wise.
            train_arr = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logger.info(f'Saved preprocessing object')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as error:
            logger.exception('Unable to initialte data transformation')
            raise CustomException(error,sys) from error
