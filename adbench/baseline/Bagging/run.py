from sklearn.linear_model import LogisticRegression
from adbench.myutils import Utils
from adbench.baseline.Bagging.fit import fit
from adbench.baseline.Bagging.model import bagging
from adbench.baseline.Bagging.preprocess import preprocessor

import torch
import math

class Bagging():
    '''
    
    '''
    def __init__(self, seed:int, model_name:str='VAE'):
        self.seed = seed
        self.utils = Utils()
        self.model_name = model_name

    def fit(self, X_train, y_train, preprocess:bool=False):
        self.preprocessor = preprocessor(normalization_scheme=None) if preprocess else None
        # Preprocess data
        if preprocess:
            X_train = self.preprocessor.fit_transform(X_train)
        
        self.model = bagging(contamination=0.1)
        # fitting
        self.model = self.model.fit(X_train, y_train)
        return self

    def predict_score(self, X):
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        score = self.model.decision_function(X)
        return score