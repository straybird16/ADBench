import torch
import torch.nn as nn
import itertools
import math
import numpy as np

class ae(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)
        if not isinstance(module, nn.Linear):
            return
        if self.initialization == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif self.initialization == 'xavier_normal':
            torch.nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(nonlinearity='linear'))
        elif self.initialization == 'ones':
            torch.nn.init.ones_(module.weight)
        elif self.initialization == 'zeros':
            torch.nn.init.zeros_(module.weight)
        else:
            raise NotImplementedError("Initialization not defined: '{}'".format(self.initialization))
        if module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, num_feature, output_feature=0, contamination:float=0.01, latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', layer_config=None, preprocess:bool=True) -> None:
        super().__init__()
        self.name = 'AE' # name
        self.num_feature, self.latent_dim, self.hidden_dim, self.initialization = num_feature, latent_dim, hidden_dim, initialization # arguments
        self.output_feature = num_feature if output_feature == 0 else output_feature
        self.hd = hidden_dim
        self.contamination, self.preprocess = contamination, preprocess
        self.error = None # extra error terms, if applicable
        """ self.transformerAttention = False

        if self.transformerAttention:
            encoding_input_dim = self.valueDim
        else:
            encoding_input_dim = num_feature """
        encoding_input_dim = num_feature
        # check if layer config is specified
        if not layer_config:
            self.layer_config = [[128, hidden_dim, latent_dim], [latent_dim, hidden_dim, 128]]
            self.layer_config = [[128, 128, 32, 32, hidden_dim, latent_dim], [latent_dim, hidden_dim, 32, 32, 128, 128]]
        else:
            self.layer_config = layer_config
        e_config, d_config = [encoding_input_dim] + self.layer_config[0], self.layer_config[1] + [self.output_feature]
        e_layers = [nn.Linear(e_config[i], e_config[i+1]) for i, _ in enumerate(e_config[:-1])]
        d_layers = [nn.Linear(d_config[i], d_config[i+1]) for i, _ in enumerate(d_config[:-1])]
        e_activation, d_activation = [nn.LeakyReLU(0.1) for _ in range(len(e_layers) - 1)], [nn.LeakyReLU(0.1) for _ in range(len(d_layers) - 1)]
        if activation == 'leaky_relu':
            e_activation, d_activation = [nn.LeakyReLU(0.1) for _ in range(len(e_layers) - 1)], [nn.LeakyReLU(0.1) for _ in range(len(d_layers) - 1)]
        elif activation == 'tanh':
            e_activation, d_activation = [nn.Tanh() for _ in range(len(e_layers) - 1)], [nn.Tanh() for _ in range(len(d_layers) - 1)]
        elif activation == 'sigmoid':
            e_activation, d_activation = [nn.Sigmoid() for _ in range(len(e_layers) - 1)], [nn.Sigmoid() for _ in range(len(d_layers) - 1)]
            
        # initialize encoder and decoder
        self.norm_layer = False
        if self.norm_layer:
            # normalization layer
            e_norm, d_norm = [nn.LayerNorm(n) for n in self.layer_config[0][:-1]], [nn.LayerNorm(n) for n in self.layer_config[1][1:]]
            e_structure, d_structure = (e_layers, e_activation, e_norm), (d_layers, d_activation, d_norm)
        else:
            e_structure, d_structure = (e_layers, e_activation), (d_layers, d_activation)
        
        self.encoding_layer = nn.Sequential(*list(itertools.chain(*itertools.zip_longest(*e_structure)))[:-len(e_structure)+1])
        self.decoding_layer = nn.Sequential(*list(itertools.chain(*itertools.zip_longest(*d_structure)))[:-len(d_structure)+1])
        # initialize all weights
        for layer in self.children():
            self._init_weights(layer)
            
    def fit(self, X_train, y_train, epochs:int=5000, lr=1e-3, wd=1e-6):
        
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
            if (epoch + 1)%1000 == 0:
                print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        
        """ temp = torch.tensor(X_train).to(torch.float32)
        train_loss = torch.sum((self(temp)-temp)**2, dim=1).detach().numpy()
        self.decision_scores_ = train_loss.ravel()
        self._process_decision_scores() """
        return self
    
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        prediction = torch.sum((self.forward(X) - X)**2, dim=-1)   #  reconstruction error
        prediction = prediction.detach().numpy()
        prediction = prediction.reshape(-1, 1)/prediction.max() #  linearly convert to probabilities of being positive (anomaly)
        return np.concatenate((1-prediction, prediction), axis=-1)
        
    def decision_function(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        reconstruction_error = torch.sum((self.forward(X) - X)**2, dim=-1)
        return reconstruction_error.detach().numpy().reshape(-1, 1)   #  reconstruction error
    
    def forward(self, X):
        return self.decoding_layer(self.encoding_layer(X))
        