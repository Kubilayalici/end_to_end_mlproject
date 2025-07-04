import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
                "Linear_Regression":LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                        
                        "Linear_Regression": {},
                        
                        "Lasso": {
                            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                            "max_iter": [1000, 5000, 10000]
                        },

                        "Ridge": {
                            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                            "solver": ["auto", "svd", "cholesky", "lsqr"]
                        },

                        "K-Neighbors Regressor": {
                            "n_neighbors": [3, 5, 7, 9, 11],
                            "weights": ["uniform", "distance"],
                            "metric": ["euclidean", "manhattan", "minkowski"]
                        },

                        "Decision Tree": {
                            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                            "max_depth": [None, 5, 10, 20, 50],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        },

                        "Random Forest Regressor": {
                            "n_estimators": [50, 100, 200],
                            "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        },

                        "XGBRegressor": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "max_depth": [3, 5, 7, 10],
                            "subsample": [0.5, 0.7, 1.0],
                            "colsample_bytree": [0.5, 0.7, 1.0]
                        },

                        "CatBoosting Regressor": {
                            "iterations": [100, 200, 500],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "depth": [4, 6, 8, 10],
                            "l2_leaf_reg": [1, 3, 5, 7, 9]
                        },

                        "AdaBoost Regressor": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                            "loss": ["linear", "square", "exponential"]
                        }
                    }


            model_report: dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models= models, params = params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)


