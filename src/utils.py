import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  # Sadece klasör yolunu al
        os.makedirs(dir_path, exist_ok=True)   # Klasörü oluştur

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)




def check_dataframe(dataframe, head=5):
    print("################ First 5 rows ###########")
    print(dataframe.head(head))
    print("############# Dataframe Shape #############")
    print(dataframe.shape)
    print("############ Null Values #############")
    print(dataframe.isnull().sum())
    print("################ Statistical Values #############")
    print(dataframe.describe())
    print("################# Last 5 rows##########")
    print(dataframe.tail(head))


def grab_col_names(dataframe, cat_th = 8, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes=="O"]
    num_but_cat =[col for col in dataframe.columns if 
                  dataframe[col].nunique()<cat_th and dataframe[col].dtypes != "O"]
    cat_but_car =[col for col in dataframe.columns if
                  dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

def evaluate_models(X_train, y_train, X_test,y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred= model.predict(X_train)
            y_test_pred= model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{list(models.keys())[i]} - Train R2: {train_model_score:.4f}, Test R2: {test_model_score:.4f}")

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)