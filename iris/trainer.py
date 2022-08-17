import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional, Union
from pathlib import Path
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class Trainer(BaseModel):
     
   train: Optional[pd.DataFrame]

   class Config:
        arbitrary_types_allowed = True

   def data_setup(self):
        self.train = load_iris()
        return self
    
   def model_train(self):
        self.data_setup()
        X = self.train.data
        y = self.train.target
        fold=1
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=8)
        for train, validation in kfold.split(X, y):
             X_train = X[train]
             X_val = X[validation]
             y_train = y[train]
             y_val = y[validation]
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

      x = self.train.data
      y = self.train.target    
      rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
      classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)
      score = cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
      accuracy = score.mean()
            
      return accuracy
   
   def model_train_optuna(self):
      self.data_setup()
      study = optuna.create_study(direction="maximize")
      study.optimize(self.objective, n_trials=100)
      print(study.best_trial)


   
   

   
