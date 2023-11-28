import itertools
import math
import numpy as np
#from adbench.baseline.PyOD import IForest
#from adbench.baseline.PyOD import CBLOF

from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF

class bagging():

    def __init__(self, contamination:float=0.05, preprocess:bool=True) -> None:
        self.name = 'Bagging' # name
        # Bagging specific components
        self.contamination = contamination
        self.models = [IForest(contamination=contamination), CBLOF(contamination=contamination)]
        # TESTING
        self.check_corr = False
        self.robust_scaling = False
        
        
    def fit(self, X_train, y_train, epochs:int=6000, lr=1e-4, wd=0e-6):
        for clf in self.models:
            clf.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X):
        prediction = self.decision_function(X)
        prediction = prediction / prediction.max() #  linearly convert to probabilities of being positive (anomalous)
        return np.concatenate((1-prediction, prediction), axis=-1)
        
    def decision_function(self, X):
        scores = []
        for clf in self.models:
            scores.append(clf.decision_function(X))
        score = np.maximum(scores[0], scores[1])
        return score.reshape(-1, 1)   #  reconstruction error
    
        