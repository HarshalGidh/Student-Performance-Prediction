import sys
import os
import dill

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

#For Data Transformation :
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
#For Model trainer:
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i] #Extract a model per use 
            model.fit(X_train,y_train) #Train Model 
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            #Model Perdormance ;
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
