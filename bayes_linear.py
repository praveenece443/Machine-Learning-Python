import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y, check_array, as_float_array
from sklearn.utils.validation import check_is_fitted
from scipy.linalg import svd
import warnings

class BayesianLinearRegression(RegressorMixin,LinearModel):

    def __init__(self, n_iter, tol, fit_intercept,copy_X, verbose):
        self.n_iter        = n_iter
        self.fit_intercept = fit_intercept
        self.copy_X        = copy_X
        self.verbose       = verbose
        self.tol           = tol
        
        
    def _check_convergence(self, mu, mu_old):

        return np.sum(abs(mu-mu_old)>self.tol) == 0
        
        
    def _center_data(self,X,y):
        X     = as_float_array(X,self.copy_X)
        X_std = np.ones(X.shape[1], dtype = X.dtype)
        if self.fit_intercept:
            X_mean = np.average(X,axis = 0)
            y_mean = np.average(y,axis = 0)
            X     -= X_mean
            y      = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1],dtype = X.dtype)
            y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
        return X,y, X_mean, y_mean, X_std
        
        
    def predict_dist(self,X):
        
        mu_pred     = self._decision_function(X)
        data_noise  = 1./self.beta_
        model_noise = np.sum(np.dot(X,self.eigvecs_)**2 * self.eigvals_,1)
        var_pred    =  data_noise + model_noise
        return [mu_pred,var_pred]