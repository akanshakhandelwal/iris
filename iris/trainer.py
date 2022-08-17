import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional, Union
from pathlib import Path
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgbm

class Trainer(BaseModel):
     
   train: Optional[pd.DataFrame]
   x:Optional[pd.DataFrame]
   y:Optional[np.ndarray]

   class Config:
        arbitrary_types_allowed = True

   def data_setup(self):
        self.train = load_iris()
        self.x = self.train.data
        self.y= self.train.target
        logger.add("optunalogs.log")
        return self
    
   def model_train(self):
        self.data_setup()
        fold=1
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=8)
        for train, validation in kfold.split(self.x,self.y):
             X_train = self.x[train]
             X_val = self.x[validation]
             y_train = self.y[train]
             y_val = self.y[validation]
             model = RandomForestClassifier()
             model.fit(X_train, y_train)
             val_pred = model.predict(X_val)
             val_accuracy = accuracy_score(y_val, val_pred)
             print(fold)
             print("Accuracy", val_accuracy)
             val_confusion_matrix = confusion_matrix(y_val, val_pred)
             print(val_confusion_matrix)
             fold = fold+1
             
        return self
      
   def objective(self,trial):
  
      rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
      classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)
      score = cross_val_score(classifier_obj, self.x, self.y, n_jobs=-1, cv=3)
      accuracy = score.mean()
      logger.info(accuracy)
            
      return accuracy
   
   def model_train_optuna(self):
      self.data_setup()
      study = optuna.create_study(direction="maximize")
      study.optimize(self.objective, n_trials=100)
      logger.info(study.best_trial)
      return self
     

   def model_stacking(self):
      self.data_setup()
      classifier1 = KNeighborsClassifier(n_neighbors=1)
      classifier2 = RandomForestClassifier(random_state=1)
      logistic_regression = LogisticRegression()
      stacking_classifier = StackingClassifier(classifiers=[classifier1, classifier2], 
                          meta_classifier=logistic_regression)  
      params = {'kneighborsclassifier__n_neighbors': [1, 5],
               'randomforestclassifier__n_estimators': [10, 50],
               }
      grid = GridSearchCV(estimator=stacking_classifier, 
                    param_grid=params, 
                    cv=5,
                    refit=True)
      grid.fit(self.x, self.y)
      print('Best parameters: %s' % grid.best_params_)
      print('Accuracy: %.2f' % grid.best_score_)

   def model_stacking_optuna(self):
      self.data_setup()
      study = optuna.create_study(direction="maximize")
      study.optimize(self.objective_optuna, n_trials=10)
    
   def objective_optuna(self,trial):
      

      classifier1 = CatBoostClassifier(n_estimators=8000)
      classifier2 = lgbm.LGBMClassifier(objective='binary',
                                      boosting_type='gbdt',
                                      num_leaves=6,
                                      max_depth=2)
     
      params = {
            'max_depth': trial.suggest_int("max_depth", 1, 10, 1),
            'gamma': trial.suggest_float('gamma', 0,1),
            'reg_alpha' : trial.suggest_float('reg_alpha', 0,50),
            'reg_lambda' : trial.suggest_float('reg_lambda', 10,100),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0,1),
            'min_child_weight' : trial.suggest_float('min_child_weight', 0, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0, .15),
            'random_state': 5,
            'n_estimators' : 8000,
            'max_bin' : trial.suggest_int('max_bin', 200, 550, 1),
            'objective': 'binary:logistic',
            'use_label_encoder':False}
         
      stacked_model = StackingClassifier(classifiers=[classifier1, classifier2], 
                             meta_classifier=XGBClassifier(params)
                             )
    
    
      stacked_model.fit(self.x, self.y)
      pred = stacked_model.predict(self.x)
      
      accuracy = accuracy_score(self.y,pred)
         
      print(accuracy)
      
      return accuracy