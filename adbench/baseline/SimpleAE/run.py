from sklearn.linear_model import LogisticRegression
from adbench.myutils import Utils
from adbench.baseline.SimpleAE.fit import fit
from adbench.baseline.SimpleAE.model import ae
from adbench.baseline.SimpleAE.preprocess import preprocessor
import math
import torch

class SimpleAE():
    '''
    You should define the following fit and predict_score function
    Here we use the LogisticRegression as an example
    '''
    def __init__(self, seed:int=0, model_name:str='SimpleAE'):
        self.seed = seed
        self.utils = Utils()
        self.model_name = model_name
        # CUDA acceleration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'

    def fit(self, X_train, y_train, preprocess:bool=False):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.preprocessor = preprocessor() if preprocess else None
        # Preprocess data
        if preprocess:
            X_train = self.preprocessor.fit_transform(X_train)
        # Initialization
        #latent_dim = min(max(math.ceil(math.log2(X_train.shape[0])), 8), X_train.shape[-1]//2)
        latent_dim = min(16, X_train.shape[-1])
        #layer_config = [[512, 512, latent_dim], [latent_dim, 64, 64, 64, 64]]
        layer_config = [[512, 512, latent_dim], [latent_dim, 128, 128, 64, 64]]

        # fitting
        with torch.device(self.device):
            # hyper-parameters
            lr, wd=1e-4, 1e-6
            self.model = ae(num_feature=X_train.shape[-1], latent_dim=latent_dim, layer_config=layer_config)
            # fitting
            self.model = self.model.fit(X_train=X_train, y_train=y_train, epochs=int(5e3), lr=lr, wd=wd)
            print("Latent dim= {}. Weight Decay = {:.6f}".format(latent_dim, wd))
            print("Using " + self.device)
        return self

    def predict_score(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # Preprocess data
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        score = self.model.decision_function(X)
        return score