import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import StandardScaler
from icecream import ic



class mygp(object):

    def __init__(
        self,  
        gp=None,  
        **kwargs    
    ):
    
        if gp:
            self.gp = gp    
        else:
            self.gp = GaussianProcessRegressor()    

        self.ndims=1
    

        self.sample_X = np.array([])
        self.sample_y = np.array([])

    def prn(self, **kwargs):

        print(f"ndims = {self.ndims}. sample shape = {self.sample_X.shape}")
        ic(self.sample_X, self.sample_y)
    
        return

    def sample_init(self,X,y):

        X_shape = X.shape
        X_len =  len(X_shape)

        
        if X_len == 1:
            self.ndims = 1
        elif X_len == 2:
            self.ndims = X_shape[1]
        else:         
            raise ValueError(f'sample_init() sample_X has incompatiple shape"{X_shape}" for the gaussian fit method.')
        
        self.sample_X = X.copy()
        self.sample_y = y.copy()


        self.gp.fit(self.sample_X, self.sample_y)

        sample_pred , sample_std = self.gp.predict(self.sample_X, return_std=True)

        return sample_pred

    def predict(self, grid_X):
                           
        X_len = len(grid_X.shape)

        if X_len == 1 and self.ndims == 1:
            pass      
        elif X_len ==2 and grid_X.shape[1] == self.ndims:       
            pass        
        else:
            raise ValueError(f'predict() grid_X has shape "{grid_X.shape}" dimensions. ndims = {self.ndims}')   

        grid_pred, grid_std = self.gp.predict(grid_X, return_std=True)

        return grid_pred, grid_std

    def log_marginal_likelihood(self, theta=None,eval_gradient=False, clone_kernel=True):

        return self.gp.log_marginal_likelihood( theta=theta, eval_gradient=eval_gradient,clone_kernel=clone_kernel  )
    
    def get_params(self):
        ic(self.gp.get_params(deep=True))

    def sample_add(self, X1, y1):

        if len(X1) != self.ndims: 
            raise ValueError(f'sample_add(). Error sample_X1 must have exactly "{self.ndims}"')      

        # add point to array
        X = np.concatenate((self.sample_X, [X1]),axis=0 )
        y = np.concatenate((self.sample_y, [y1]),axis=0)     

        ic(X.shape,y.shape)
        ic(X,y)
        
        # remember if sucessfull this will update self.sample_X, self.sample_y
        self.sample_init(X,y)
