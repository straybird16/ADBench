from sklearn.linear_model import LogisticRegression
from adbench.myutils import Utils
from adbench.baseline.VAE.fit import fit
from adbench.baseline.VAE.model import vae
from adbench.baseline.VAE.preprocess import preprocessor
import math

class VAE():
    '''
    You should define the following fit and predict_score function
    Here we use the LogisticRegression as an example
    '''
    def __init__(self, seed:int, model_name:str='VAE'):
        self.seed = seed
        self.utils = Utils()
        self.model_name = model_name

    def fit(self, X_train, y_train, preprocess:bool=True):
        self.preprocessor = preprocessor(normalization_scheme=None) if preprocess else None
        # Preprocess data
        if preprocess:
            X_train = self.preprocessor.fit_transform(X_train)
        # Initialization
        latent_dim = min(max(math.ceil(math.log2(X_train.shape[0])), 8), X_train.shape[-1]//2)
        self.model = vae(num_feature=X_train.shape[-1], latent_dim=latent_dim)
        #self.model = ae(num_feature=X_train.shape[-1], latent_dim=X_train.shape[-1]//2)
        #self.model = ae(num_feature=X_train.shape[-1], latent_dim=math.ceil(math.log2(X_train.shape[0])))

        # fitting
        self.model = fit(X_train=X_train, y_train=y_train, model=self.model)

        return self

    def predict_score(self, X):
        # Preprocess data
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        score = self.model.decision_function(X)
        return score