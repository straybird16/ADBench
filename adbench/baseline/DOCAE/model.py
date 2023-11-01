import torch
import torch.nn as nn
import itertools
import math
import numpy as np
from adbench.baseline.SimpleAE.model import ae
from torchsummary import summary

class docae(ae):

    def __init__(self, num_feature, output_feature=0, contamination:float=0.05, center:float=0, R:float=1, alpha:float=1e-3, latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', layer_config=None, preprocess:bool=True) -> None:
        super().__init__(num_feature=num_feature, output_feature=output_feature, contamination=contamination, latent_dim=latent_dim, hidden_dim=hidden_dim, activation=activation, initialization=initialization,layer_config=layer_config,preprocess=preprocess)
        self.name = 'DOCAE' # name
        # DOCAE specific components
        self.v = contamination
        self.center = torch.tensor(center)
        self.alpha = alpha
        self.R = nn.Parameter(torch.tensor(R, dtype=float, requires_grad=True))
        self.error = 0
        # TESTING
        self.check_corr = False
        self.robust_scaling = False
        
        
    def fit(self, X_train, y_train, epochs:int=6000, lr=1e-4, wd=0e-6):
        # mini-batch size
        batch_size = math.ceil(X_train.shape[0]/10)
        batch_size = math.ceil(X_train.shape[0]/2)
        grad_limit = 1e3
        
        #X_train = torch.tensor(X_train, dtype=torch.float32)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss(reduction='sum')
        #summary(self, (X_train.shape[0], X_train.shape[1]), batch_size)
        
        for epoch in range(epochs):
            loss, l = 0, X_train.shape[0]
            
            permutation = np.random.permutation(l)
            for i in range(0, l, batch_size):
                    
                batch_idc = permutation[i:i+batch_size]
                batch_X = X_train[batch_idc,]
                #batch_X = torch.tensor(batch_X, dtype=torch.float32)
                batch_Y = batch_X
                
                optimizer.zero_grad()
                # compute reconstructions
                outputs = self.forward(batch_X) 
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_Y)
                if self.error:
                    train_loss += self.error
                train_loss.backward(retain_graph=False)
                loss += train_loss.item()
                # compute accumulated gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_limit)
                # perform parameter update based on current gradients
                optimizer.step()
                
            loss /= l
            # print training process for each 100 epochs
            if (epoch + 1)%1000 == 0 or epoch == 0:
                print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        
        
        # compute the mean and standard deviation of tarining samples
        #X = torch.tensor(X_train, dtype=torch.float32)
        X = X_train
        error = torch.sum((self.forward(X) - X)**2, dim=-1)#.detach().numpy()
        self.error_mu_, self.error_std_ = error.mean(), error.std()
        self.error_median_, self.error_range_ = error.quantile(q=0.5), error.quantile(q=0.8) - error.quantile(q=0.2)
        self.latent_error_mu_, self.latent_error_sigma_ = self.instance_wise_error.mean(), self.instance_wise_error.std()
        self.latent_error_median_, self.latent_error_range_ = self.instance_wise_error.quantile(q=0.5), self.instance_wise_error.quantile(q=0.8) - self.instance_wise_error.quantile(q=0.2)
        print("self.error_median={:.4f}, self.error_range={:.4f}, self.latent_error_median_={:.4f}, self.latent_error_range_={:.4f}".format(self.error_median_, self.error_range_, self.latent_error_median_, self.latent_error_range_))
        print("Model result scaling scheme: ", self.robust_scaling)
        return self
    
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        prediction = self.decision_function(X)
        prediction = prediction / prediction.max() #  linearly convert to probabilities of being positive (anomalous)
        return np.concatenate((1-prediction, prediction), axis=-1)
        
    def decision_function(self, X):
        #X = torch.tensor(X, dtype=torch.float32)
        reconstruction_error = torch.sum((self.forward(X) - X)**2, dim=-1)
        score = reconstruction_error#.detach().numpy()
        # normalize errors
        if self.robust_scaling:
            score = (score - self.error_median_) / self.error_range_
            normalized_latent_error = (self.instance_wise_error - self.latent_error_median_) / self.latent_error_range_
        else:
            score = (score - self.error_mu_) / self.error_std_ # normalize
            normalized_latent_error = (self.instance_wise_error - self.latent_error_mu_) / self.latent_error_sigma_
        score =  torch.maximum(score, normalized_latent_error)
        print("self.error_mu={:.4f}, self.error_std={:.4f}, self.latent_error_mu_={:.4f}, self.latent_error_sigma_={:.4f}; alpha = {:.4f}".format(self.error_mu_, self.error_std_, self.latent_error_mu_, self.latent_error_sigma_, self.alpha))
        return score.reshape(-1, 1)   #  reconstruction error
    
    def _encode(self, X):
        X = self.encoding_layer(X)
        center = self.center.expand(1, X.shape[-1])
        self.instance_wise_error = torch.sum((X - center)**2, dim=-1)
        #self.instance_wise_error = (distance-self.R**2) * (distance > self.R**2)
        #self.error = self.R**2 + torch.mean(self.instance_wise_error)
        self.error = torch.mean(self.instance_wise_error)
        self.error *= self.alpha
        if self.check_corr:
            self.error += torch.mean(torch.corrcoef(X.t()))
        return X
    
    def _decode(self, X):
        return self.decoding_layer(X)
    
    def forward(self, X):
        return self._decode(self._encode(X))
        