from sklearn.linear_model import LogisticRegression
from adbench.myutils import Utils
from adbench.baseline.AADOCAE.fit import fit
from adbench.baseline.AADOCAE.model import aadocae
from adbench.baseline.AADOCAE.preprocess import preprocessor

import torch
import math

class AADOCAE():
    '''
    You should define the following fit and predict_score function
    Here we use the LogisticRegression as an example
    '''
    def __init__(self, seed:int=0, model_name:str='AADOCAE'):
        self.seed = seed
        self.utils = Utils()
        self.model_name = model_name
        # CUDA acceleration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X_train, y_train, preprocess:bool=False):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.preprocessor = preprocessor(normalization_scheme=None) if preprocess else None
        # Preprocess data
        if preprocess:
            X_train = self.preprocessor.fit_transform(X_train)
        # Initialization
        latent_dim = min(max(math.ceil(math.log2(X_train.shape[0])), 8), X_train.shape[-1]//2)
        latent_dim = min(16, X_train.shape[-1]//2+1)
        latent_dim = min(16, X_train.shape[-1])
        layer_config = [[512, 512, latent_dim], [latent_dim, 64, 64, 64, 64]]
        #layer_config = [[512, 512, 16, latent_dim], [latent_dim, 64, 64, 64, 64]]
        #layer_config = [[128, 128, 128, 128, latent_dim], [latent_dim, 128, 128, 128, 128]]
        
        with torch.device(self.device):
            # hyper-parameters
            wd=1e-6
            alpha=1e-0
            self.model = aadocae(num_feature=X_train.shape[-1], latent_dim=latent_dim, layer_config=layer_config, alpha=alpha)
            # fitting
            self.model = self.model.fit(X_train=X_train, y_train=y_train, epochs=int(3e3), lr=1e-4, wd=wd)
            print("Latent dim= {}. Weight Decay = {:.6f}".format(latent_dim, wd))
            print("Using " + self.device)
        return self

    def predict_score(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # Preprocess data
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        score = self.model.decision_function(X)
        return score.cpu().detach().numpy()
    
    def evaluate_model(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # Preprocess data
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        
        return self.model.evaluate_model(X, y)
        