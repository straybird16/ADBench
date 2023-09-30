import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler, StandardScaler, MinMaxScaler

class preprocessor():
    def __init__(self, trim_threshold:float=None, linear_dependency_threshold:float=0.98, normalization_scheme:bool='standard') -> None:
        """Data preprocessor with APIs to fit on training data and transform incoming data. Have multiple processing schemes.

        Args:
            trim (bool, optional): Threshold of trimming features with extensively repeated values. Defaults to None.
            linear_dependency_threshold (bool, optional): Threshold of filtering features in a group of highly linearly correlated features; keep only one per such group. Defaults to 0.98.
            normalization_scheme (bool, optional): Normalization scheme per feature. Defaults to standard scaler.
        """
        self.trim_threshold, self.linear_dependency_threshold, self.normalization_scheme= trim_threshold, linear_dependency_threshold, normalization_scheme
    
    def fit(self, X):
        # X must be a 2-d array
        
        self.saved_columns = [i for i in range(X.shape[-1])]
        if self.trim_threshold is not None:
            self.saved_columns = []
            # Trim data by filtering out useless features, which are the ones who appear (almost) the same throughout observations
            for i in range(X.shape[-1]):
                if np.unique(X[:,i], return_counts=True)[1].max() < self.trim_threshold * X.shape[0]: # max frequency of occurrences
                    self.saved_columns.append(i)
            #print("Untrimmed columns: ", self.saved_columns)
            X=X[:,self.saved_columns]
            print('Train data shape after trim: ', X.shape)
        # filter linear dependency
        if self.linear_dependency_threshold:
            i, seq = 0, np.arange(X.shape[-1])
            self.saved_columns = np.array(self.saved_columns)
            while True:
                coef_matrix = abs(np.corrcoef(X, rowvar=False)) - np.eye(X.shape[-1]) # corr-coefficient matrix with self coefficient zeroed out
                # Convert NaN to 0, as 0 variance means same value across samples, which can be considered as 0 correlation
                coef_matrix = np.nan_to_num(coef_matrix)
                if i >= coef_matrix.shape[0]:
                    break # no more element to filter
                idc = coef_matrix[i] < self.linear_dependency_threshold # indices that are NOT strongly linearly dependent with i-th element
                X=X[:,idc]
                # save idc
                seq = seq[idc]
                i += 1 # go to next unfiltered feature
            self.saved_columns = self.saved_columns[seq]
            print('Train data shape after filter corrcoef: ', X.shape)
            
        if self.normalization_scheme:
            #print('Normalization scheme: ', self.normalization_scheme)
            if self.normalization_scheme == 'min_max':
                self.transformer = MinMaxScaler()
            elif self.normalization_scheme == 'robust':
                self.transformer = RobustScaler(quantile_range=(20,80))
            elif self.normalization_scheme == 'quantile_transform_uniform':
                self.transformer = QuantileTransformer(output_distribution="uniform")
            elif self.normalization_scheme == 'quantile_transform_normal':
                self.transformer = QuantileTransformer(output_distribution="normal")
            elif self.normalization_scheme == 'power_transformation_yj':
                self.transformer = PowerTransformer(method="yeo-johnson")
            elif self.normalization_scheme == 'power_transformation_bc':
                self.transformer = PowerTransformer(method="box-cox")
            elif self.normalization_scheme == 'max_abs':
                self.transformer = MaxAbsScaler()
            else:
                self.transformer = StandardScaler()   
            self.transformer.fit(X)
            #X = self.transformer.transform(X)
            #return self.transformer.transform(X)
            
    def transform(self, X):
        # feature filtering
        X = X[:,self.saved_columns]
        # scale data
        X = self.transformer.transform(X)
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    