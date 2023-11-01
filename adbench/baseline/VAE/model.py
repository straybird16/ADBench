import torch
import torch.nn as nn
import itertools
import math
import numpy as np
from adbench.baseline.SimpleAE.model import ae

class vae(ae):

    def __init__(self, num_feature, output_feature=0, contamination:float=0.01, latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', sigma:float=1e-0, layer_config=None, preprocess:bool=True) -> None:
        super().__init__(num_feature=num_feature, output_feature=output_feature, contamination=contamination, latent_dim=latent_dim, hidden_dim=hidden_dim, activation=activation, initialization=initialization,layer_config=layer_config,preprocess=preprocess)
        self.name = 'VAE' # name
        # VAE specific components
        self.sigma = sigma
        self.generate_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.generate_log_var = nn.Linear(self.latent_dim, self.latent_dim)
        self._init_weights(self.generate_mean)
        self._init_weights(self.generate_log_var)
        self.KLD=0
            
    def fit(self, X_train, y_train, epochs:int=6000, lr=1e-4, wd=0e-6):
        
        batch_size = math.ceil(X_train.shape[0]/10)
        grad_limit = 1e3
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss(reduction='sum')
        
        for epoch in range(epochs):
            loss, l = 0, X_train.shape[0]
            permutation = np.random.permutation(l)
            for i in range(0, l, batch_size):
                    
                batch_idc = permutation[i:i+batch_size]
                batch_X = X_train[batch_idc,]
                batch_X = torch.tensor(batch_X, dtype=torch.float32)
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
            if (epoch + 1)%1000 == 0  or epoch == 0:
                print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        
        # compute the mean and standard deviation of tarining samples
        X = torch.tensor(X_train, dtype=torch.float32)
        error = torch.sum((self.forward(X) - X)**2, dim=-1).detach().numpy()
        self.error_mu_, self.error_std_ = error.mean(), error.std()
        self.KLD_mu_, self.KLD_sigma_ = self.KLD.detach().numpy(), self.row_wise_KLD.std().detach().numpy()
        return self
    
    def predict_proba(self, X):
        #X = torch.tensor(X, dtype=torch.float32)
        #prediction = torch.sum((self.forward(X) - X)**2, dim=-1)   #  reconstruction error
        prediction = self.decision_function(X)   #  reconstruction error
        prediction = prediction / prediction.max() #  linearly convert to probabilities of being positive (anomaly)
        return np.concatenate((1-prediction, prediction), axis=-1)
        
    def decision_function(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        reconstruction_error = torch.sum((self.forward(X) - X)**2, dim=-1)
        score = reconstruction_error.detach().numpy()
        score = (score - self.error_mu_) / self.error_std_ * (self.row_wise_KLD.detach().numpy() - self.KLD_mu_) / self.KLD_sigma_
        print("self.error_mu={}, self.error_std={}, self.KLD_mu={}, self.KLD_sigma={}".format(self.error_mu_, self.error_std_, self.KLD_mu_, self.KLD_sigma_))
        return score.reshape(-1, 1)   #  reconstruction error
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std)
        self.row_wise_KLD = torch.mean(-0.5*(1 + torch.log(var) - mu**2 - var), dim=-1)
        self.KLD = self.row_wise_KLD.mean()
        self.error=self.sigma * self.KLD
        return sample
    
    def forward(self, X):
        X = self.encoding_layer(X)
        mu, var = self.generate_mean(X), torch.exp(nn.Tanh()(self.generate_log_var(X)))
        X = self.reparameterize(mu, var)
        return self.decoding_layer(X)
        