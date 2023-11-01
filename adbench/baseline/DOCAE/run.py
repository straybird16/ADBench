from sklearn.linear_model import LogisticRegression
from adbench.myutils import Utils
from adbench.baseline.DOCAE.fit import fit
from adbench.baseline.DOCAE.model import docae
from adbench.baseline.DOCAE.preprocess import preprocessor

import torch
import math

class DOCAE():
    '''
    You should define the following fit and predict_score function
    Here we use the LogisticRegression as an example
    '''
    def __init__(self, seed:int, model_name:str='VAE'):
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
        latent_dim = min(16, X_train.shape[-1]//2)
        layer_config = [[512, 512, latent_dim], [latent_dim, 64, 64, 64, 64]]
        
        with torch.device(self.device):
            # hyper-parameters
            wd=0e-2
            alpha=1e-3
            self.model = docae(num_feature=X_train.shape[-1], latent_dim=latent_dim, layer_config=layer_config, alpha=alpha)
            # fitting
            self.model = self.model.fit(X_train=X_train, y_train=y_train, epochs=int(1e4), lr=1e-4, wd=wd)
            print("Weight Decay = {:.6f}".format(wd))
        return self

    def predict_score(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # Preprocess data
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        score = self.model.decision_function(X)
        return score.cpu().detach().numpy()