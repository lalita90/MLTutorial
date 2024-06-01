import sys
from dataclasses import dataclass
import logging
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import os
# Set up logging to save messages to a file
logging.basicConfig(filename='/Users/lmeena/Desktop/ML_Lalita/MLTutorial/logs/example_tf.log', level=logging.INFO)

# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory of 'src' to the Python path
src_parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(src_parent_dir)

# Import logger from the 'src' folder
from logger import logging as custom_logging
from exception import CustomException

from utils import save_object
custom_logging.info("Starting data transformation")
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        #try:
        numerical_columns = ["writing_score", "reading_score"]
        categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())

            ]
        )

        cat_pipeline=Pipeline(

            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]

        )
        custom_logging.info("created pipeline")
        custom_logging.info(f"Categorical columns: {categorical_columns}")
        custom_logging.info(f"Numerical columns: {numerical_columns}")

        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipelines",cat_pipeline,categorical_columns)

            ]


        )

        return preprocessor

        #except Exception as e:
         #   raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        #try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)

        custom_logging.info("Read train and test data completed")

        custom_logging.info("Obtaining preprocessing object")

        preprocessing_obj=self.get_data_transformer_object()

        target_column_name="math_score"
        numerical_columns = ["writing_score", "reading_score"]

        input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
        target_feature_train_df=train_df[target_column_name]

        input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
        target_feature_test_df=test_df[target_column_name]

        custom_logging.info(
            f"Applying preprocessing object on training dataframe and testing dataframe."
        )

        input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        custom_logging.info(f"Saved preprocessing object.")

        save_object(

            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj

        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )
        #except Exception as e:
         #   raise CustomException(e, sys)
