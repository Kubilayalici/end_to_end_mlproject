import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import grab_col_names, save_object
from src.exception import CustomException
from src.logger import logging



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self, df:pd.DataFrame):
        try:
            cat_cols, num_cols, cat_but_car = grab_col_names(df)
            
            logging.info(f"Kategorik kolonlar: {cat_cols}")
            logging.info(f"Numerik kolonlar: {num_cols}")
            logging.info(f"Kategorik ama kardinal kolonlar: {cat_but_car}")
            
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            logging.info('Numerical columns scaling completed')
            logging.info('Categorical columns encoding completed')

            preprocessor = ColumnTransformer(transformers=[
                ("num", numerical_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols)
            ])



            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed!")
            logging.info("Obtaining preprocessing object")

            target_column_name = "G3"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            logging.info('Applying preprocessing object on training and testing data')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
