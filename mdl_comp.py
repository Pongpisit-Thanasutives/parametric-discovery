import scipy
import numpy as np 
from sklearn.base import BaseEstimator
# Code ref: https://github.com/csinva/mdl-complexity (Dwivedi et al. Revisiting minimum description length complexity in overparameterized models)

def prac_mdl_comp(X_train, y_train, variance=1):
    '''Calculate prac-mdl-comp for this dataset
    '''
    eigenvals, eigenvecs = np.linalg.eig(X_train.T @ X_train)

    def calc_thetahat(l):
        inv = np.linalg.pinv(X_train.T @ X_train + l * np.eye(X_train.shape[1]))
        return inv @ X_train.T @ y_train

    def prac_mdl_comp_objective(l):
        thetahat = calc_thetahat(l)
        mse_norm = np.linalg.norm(y_train - X_train @ thetahat)**2 / (2 * variance)
        theta_norm = l*np.linalg.norm(thetahat)**2 / (2 * variance)
        eigensum = 0.5 * np.sum(np.log((eigenvals + l) / l))
        return (mse_norm + theta_norm + eigensum) / y_train.size

    opt_solved = scipy.optimize.minimize(prac_mdl_comp_objective, bounds=((0.0, np.inf),), x0=1e-10)
    prac_mdl = opt_solved.fun
    lambda_opt = opt_solved.x
    thetahat = calc_thetahat(lambda_opt)
    
    return {
        'prac_mdl': prac_mdl,
        'lambda_opt': lambda_opt,
        'thetahat': thetahat
    }

def redundancy(X_train, l):
    eigenvals, eigenvecs = np.linalg.eig(X_train.T @ X_train)
    eigensum = 0.5 * np.sum(np.log((eigenvals + l) / l))
    return eigensum

class RidgeMDLCOMP(BaseEstimator):
    def __init__(self, variance=None):
        super().__init__()
        self.variance = variance
        self.coef_ = None
        self.lambda_opt = None
        self.prac_mdl = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.variance is None:
            self.variance = np.linalg.lstsq(X, y, rcond=None)[1]/(len(y)-X.shape[-1])
        stats = prac_mdl_comp(X, y, self.variance)
        self.coef_ = stats['thetahat']
        self.lambda_opt = stats['lambda_opt']
        self.prac_mdl = stats['prac_mdl']
        return self

    def predict(self, X):
        return X@self.theta

