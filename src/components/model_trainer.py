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
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','modle.pkl')

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split Training and Test input data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models)
            #To get best model score :
            best_model_score = max(sorted(model_report.values()))
            #to get best Model Name from dict :
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            #If performance of all models is very poor :
            if best_model_score < 0.6:
                raise CustomException("No best Model Found")
            logging.info("Best Training and test model found")

            #Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            #Check Accuracy :
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)

