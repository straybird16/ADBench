import torch
import torch.nn as nn
import itertools
import math
import numpy as np
from adbench.baseline.SimpleAE.model import ae
from adbench.baseline.AADOCAE.utils import visualize_codes
from torchsummary import summary
from sklearn.metrics import roc_auc_score, average_precision_score

class aadocae(ae):
    def init_projectors(self, projectors:nn.ModuleList, num_feature:int, num_latent_dim:int)->nn.ModuleList:
        for sequence in projectors:
            sequence.append(nn.Linear(num_feature, num_latent_dim))
            sequence.append(nn.LeakyReLU(0.1))
            sequence.append(nn.Linear(num_latent_dim, num_latent_dim))
            sequence.append(nn.LeakyReLU(0.1))
            sequence.append(nn.Linear(num_latent_dim, num_latent_dim))
        return projectors
    
    def __init__(self, num_feature, output_feature=0, contamination:float=0.05, center:float=0, R:float=1, alpha:float=1e-0, latent_dim=4, hidden_dim=8, activation='leaky_relu', initialization='xavier_normal', layer_config=None, preprocess:bool=True) -> None:
        super().__init__(num_feature=num_feature, output_feature=output_feature, contamination=contamination, latent_dim=latent_dim, hidden_dim=hidden_dim, activation=activation, initialization=initialization,layer_config=layer_config,preprocess=preprocess)
        self.name = 'AADOCAE' # name
        # AADOCAE specific components
        self.v = contamination
        self.center = torch.tensor(center)
        self.alpha = alpha
        self.R = nn.Parameter(torch.tensor(R, dtype=float, requires_grad=True))
        self.error = 0
        # TESTING
        self.check_corr = False
        self.robust_scaling = True
        
        # transformer sub-structure
        self.use_attention = True
        self.num_attention_heads = 8
        self.attention_head_weights = nn.Parameter(torch.rand(self.num_attention_heads), requires_grad=True)
        self.query_projectors = nn.ModuleList([nn.Sequential() for _ in range(self.num_attention_heads)])
        self.key_projectors = nn.ModuleList([nn.Sequential() for _ in range(self.num_attention_heads)])
        self.value_projectors = nn.ModuleList([nn.Sequential() for _ in range(self.num_attention_heads)])
        
        
    def fit(self, X_train, y_train, epochs:int=6000, lr=1e-4, wd=0e-6):
        # initialize transformer components
        num_feature, d_model = X_train.shape[-1], 512
        d_k, d_v = 64, 64
        self.init_projectors(self.query_projectors, num_feature, d_model)
        self.init_projectors(self.key_projectors, num_feature, d_model)
        self.init_projectors(self.value_projectors, num_feature, d_model)
        self.W_Q = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(self.num_attention_heads)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(self.num_attention_heads)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, d_v, bias=False) for _ in range(self.num_attention_heads)])
        encoding_dim = self.num_attention_heads*d_v # + num_feature + self.latent_dim
        self.transformer_FCL = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.LeakyReLU(0.1),
            #nn.Linear(encoding_dim, 1),
            nn.Linear(encoding_dim, 1),
        )
        l = X_train.shape[0]
        batch_size_list = [l//2, l]
        grad_limit = 1e4
        
        #params_to_optimize = list(self.parameters())+list(self.query_projectors.parameters())+list(self.key_projectors.parameters())+list(self.value_projectors.parameters())+list(self.transformer_FCL.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        #optimizer = torch.optim.NAdam(self.parameters(), lr=lr, weight_decay=wd)
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd, nesterov=True, momentum=0.9)
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)
        
        criterion = nn.MSELoss(reduction='none')
        #summary(self, (X_train.shape[0], X_train.shape[1]), batch_size)
        
        for epoch in range(epochs):
            self.train()
            loss= 0   
            permutation = np.random.permutation(l)
            # mini-batch size
            #batch_size = math.ceil(l/((np.random.rand()+1)))
            batch_size = math.ceil(l/2)
            for i in range(0, l, batch_size):
                train_loss = 0
                batch_idc = permutation[i:i+batch_size]
                batch_X = X_train[batch_idc,]
                batch_Y = batch_X
                #batch_X = nn.Dropout(p=0.25)(batch_X)  # masking input
                
                # compute reconstructions
                outputs = self.forward(batch_X) 
                # compute training reconstruction loss
                rec_loss = criterion(outputs, batch_Y).sum(dim=-1).view(-1, 1)
                train_loss += rec_loss
                # randomly sample a subset of the data instances and calculate the minimal determinant of their covariance matrices
                num_samples = 3
                
                selector_loss, selector_idc, selector_weights = None, None, None
                #selector_loss, selector_idc = self.determinant_selector(batch_X, num_samples=num_samples)
                if self.use_attention:
                    selector_loss, selector_idc, selector_weights = self._transformer_selector(batch_X)
                if self.error:
                    #train_loss += self.error
                    #train_loss += torch.mean(self.instance_wise_error[selector_idc])
                    train_loss += (self.instance_wise_error*self.alpha).view(-1, 1)
                if selector_weights is not None:
                    train_loss = (train_loss*selector_weights)
                if selector_loss is not None:
                    train_loss += selector_loss
                # record loss to print
                loss += train_loss.sum()
                train_loss = train_loss.mean()
                train_loss.backward(retain_graph=True)
                # compute accumulated gradients
                nn.utils.clip_grad_norm_(self.parameters(), grad_limit)
                # perform parameter update based on current gradients
                optimizer.step()    
                optimizer.zero_grad()
                     
            # print training process for each 100 epochs
            loss /= l
            if (epoch + 1)%1000 == 0 or epoch == 0:
                print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
                #print(selector_weights[:10])
                """ with torch.no_grad():
                    rec_loss = criterion(self.forward(X_train), X_train)/l
                    _, _, w = self._transformer_selector_mp(X_train) """
                #print("Predicted anomaly ratio: {:.4f}; embeddings mean: {:.4f}, dev: {:.4f}; dev loss: {:.4f}, rec loss: {:.4f}".format((w<0).sum()/l, (self.codes**2).mean(), (self.codes**2).std(), self.error, rec_loss))

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
        # visualize codes
        #visualize_codes(self.codes.cpu().detach().numpy(), y_train)
        return self
    
    def determinant_selector(self, X, num_samples=3):
        for _ in range(num_samples):
            rng = np.random.default_rng()  # create rng generator
            batch_size = X.shape[0]
            perm = rng.permutation(batch_size)
            idc = perm[:min((batch_size + X.shape[-1] + 1)//2, batch_size*3//4)]  # sample
            subset = X[idc,]
            cov_matrix = torch.cov((self.encoding_layer(subset)).t()) # calculate determinant
            det = torch.linalg.det(cov_matrix)
            # if subset has smaller determinant in its covariance matrix
            minimum_determinant, minimum_idc = float('inf'), None
            if det < minimum_determinant:
                minimum_determinant, minimum_idc = det, idc  # update smallest determinant and the corresponding idc
                
                #complement_idc = perm[(X_train.shape[0] + X_train.shape[-1] + 1)//2:,]  # calculate the determinant of the complement subset of data instances
                #complement_set = X_train[complement_idc,]
                #complement_cov_matrix = torch.cov(self.encoding_layer(complement_set).t())
                #complement_det = torch.linalg.det(complement_cov_matrix)  # and we wish to minimize this determinant and maximize the determinant of the covariance matrix of the rest of the data
        return minimum_determinant, minimum_idc, None
    
    # a transformer selector function to select the least anomalous data samples based on the encodings of them
    def _transformer_selector(self, X):
        idc = None
        Q = self.query_projectors[0](X)
        K = self.key_projectors[0](X)
        V = self.value_projectors[0](X)
        dim_q = Q[0].shape[-1]
        encodings = []
        softmax = nn.Softmax(dim=0)
        #normalized_head_weights = nn.Softmax(dim=0)(self.attention_head_weights)
        for i in range(self.num_attention_heads):
            q, k, v = self.W_Q[i](Q), self.W_K[i](K), self.W_V[i](V)
            attention_weights = torch.matmul(q, k.t())/dim_q
            attention_weights = softmax(attention_weights)
            weighted_values = torch.matmul(attention_weights, v)
            encodings.append(weighted_values)
            #encodings.append(torch.linalg.multi_dot((q, k.t(), v)) * normalized_head_weights[i])
        #encodings.append(X)
        #encodings.append(self.codes) # embeddings of X
        X = self.transformer_FCL(torch.concat(encodings, dim=-1))
        
        X = softmax(X / (X.shape[0]**0.5)) * X.shape[0]
        #X = nn.Softmax(dim=-1)(X)
        #X = X * X.shape[0] / X.sum(dim=0)
        # the error lower than the set threshold 
        
        #X = nn.Sigmoid()(X)
        
        #idc = torch.where(X > X.quantile(0.2, dim=0, interpolation='midpoint'))[0]
        return None, idc, X
    
    # a transformer selector function to select the least anomalous data samples based on the encodings of them
    def _transformer_selector_mp(self, X):
        Q_heads = [projection(X) for projection in self.query_projectors]
        K_heads = [projection(X) for projection in self.key_projectors]
        V_heads = [projection(X) for projection in self.value_projectors]
        dim_q = Q_heads[0].shape[-1]
        encodings = []
        normalized_head_weights = nn.Softmax(dim=0)(self.attention_head_weights)
        for i in range(self.num_attention_heads):
            q, k, v = Q_heads[i], K_heads[i], V_heads[i]
            encodings.append(torch.linalg.multi_dot((q, k.t(), v)) * normalized_head_weights[i])
        encodings.append(X)
        encodings.append(self.codes) # embeddings of X
        X = self.transformer_FCL(torch.concat(encodings, dim=-1))
        
        #X = nn.Softmax(dim=0)(X) * X.shape[0]
        #X = nn.Softmax(dim=-1)(X)
        X = X * X.shape[0] / X.sum(dim=0)
        # the error lower than the set threshold 
        
        #X = nn.Sigmoid()(X)
        idc = None
        #idc = torch.where(X > X.quantile(0.2, dim=0, interpolation='midpoint'))[0]
        return None, idc, X
        
    """ def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        prediction = self.decision_function(X)
        prediction = prediction / prediction.max() #  linearly convert to probabilities of being positive (anomalous)
        return np.concatenate((1-prediction, prediction), axis=-1) """
        
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
        #score =  torch.maximum(score, normalized_latent_error)
        score += normalized_latent_error
        print("self.error_mu={:.4f}, self.error_std={:.4f}, self.latent_error_mu_={:.4f}, self.latent_error_sigma_={:.4f}; alpha = {:.4f}".format(self.error_mu_, self.error_std_, self.latent_error_mu_, self.latent_error_sigma_, self.alpha))
        return score.reshape(-1, 1)   #  reconstruction error
    
    def _encode(self, X):
        X = self.encoding_layer(X)
        self.codes = X
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
    
    def visualize_with_tsne(self, y, title='no_title'):
        visualize_codes(self.codes.cpu().detach().numpy(), y=y, title=title)
        
    def _fit_ae(self, X_train, epochs:int=1000, lr=1e-3, wd=1e-6):
        # mini-batch size
        batch_size = math.ceil(X_train.shape[0]/2)
        grad_limit = 1e3
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd, nesterov=True, momentum=0.9)
        criterion = nn.MSELoss(reduction='sum')
        
        for epoch in range(epochs):
            self.train()
            loss, l = 0, X_train.shape[0]    
            permutation = np.random.permutation(l)
            for i in range(0, l, batch_size):
                train_loss = 0
                batch_idc = permutation[i:i+batch_size]
                batch_X = X_train[batch_idc,]
                batch_Y = batch_X
                batch_X = nn.Dropout(p=0.25)(batch_X)  # masking input
                outputs = self.forward(batch_X) 
                # compute training reconstruction loss
                rec_loss = criterion(outputs, batch_Y)
                train_loss += rec_loss
                # update parameter

                train_loss.backward(retain_graph=True)
                loss += train_loss.item()
                # compute accumulated gradients
                nn.utils.clip_grad_norm_(self.parameters(), grad_limit)
                # perform parameter update based on current gradients
                optimizer.step()
                optimizer.zero_grad()        
            
            loss /= l
            # print training process for each 100 epochs
            if (epoch + 1)%1000 == 0 or epoch == 0:
                print("AE training: epoch : {}/{},s loss = {:.6f}".format(epoch + 1, epochs, loss))
                
    
    def evaluate_model(self, X, y):
        res = {}
        self.eval()
        rec_score = ((self.forward(X) - X)**2).sum(dim=-1).cpu().detach().numpy()
        dev_score = self.instance_wise_error.cpu().detach().numpy()
        if self.use_attention:
            _, _, transformer_score = self._transformer_selector(X)
            transformer_score = transformer_score.sum(dim=-1).cpu().detach().numpy()
        
        rec_aucroc, rec_aucpr = roc_auc_score(y_true=y, y_score=rec_score), average_precision_score(y_true=y, y_score=rec_score, pos_label=1)
        dev_aucroc, dev_aucpr = roc_auc_score(y_true=y, y_score=dev_score), average_precision_score(y_true=y, y_score=dev_score, pos_label=1)
        max_aucroc, max_aucpr = roc_auc_score(y_true=y, y_score=np.maximum(rec_score, dev_score)), average_precision_score(y_true=y, y_score=np.maximum(rec_score, dev_score), pos_label=1)
        ss_aucroc, ss_aucpr = roc_auc_score(y_true=y, y_score=rec_score**2+dev_score**2), average_precision_score(y_true=y, y_score=rec_score**2+dev_score**2, pos_label=1)
        print("\nModel performance:\nReconstruction AUCROC={:.4f}; AUCPR={:.4f}".format(rec_aucroc, rec_aucpr))
        print("Deviation AUCROC={:.4f}; AUCPR={:.4f}".format(dev_aucroc, dev_aucpr))
        print("Max(rec, dev) AUCROC={:.4f}; AUCPR={:.4f}\n".format(max_aucroc, max_aucpr))
        print("Sum of squared error AUCROC={:.4f}; AUCPR={:.4f}\n".format(ss_aucroc, ss_aucpr))
        if self.use_attention:
            transformer_aucroc, transformer_aucpr = roc_auc_score(y_true=y, y_score=transformer_score), average_precision_score(y_true=y, y_score=transformer_score, pos_label=1)
            print("Transformer selector AUCROC={:.4f}; AUCPR={:.4f}\n".format(transformer_aucroc, transformer_aucpr))
            
        res['rec_aucroc'], res['rec_aucpr'] = rec_aucroc, rec_aucpr
        res['dev_aucroc'], res['dev_aucpr'] = dev_aucroc, dev_aucpr
        res['max_aucroc'], res['max_aucpr'] = max_aucroc, max_aucpr
        res['ss_aucroc'], res['ss_aucpr'] = ss_aucroc, ss_aucpr
        return res
        
                