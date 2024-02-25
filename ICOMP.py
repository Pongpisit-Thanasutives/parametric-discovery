import numpy as np
import statsmodels.api as sm
from scipy.stats import gmean
from scipy.linalg import solve
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.metrics.pairwise import pairwise_kernels
# Following https://arxiv.org/pdf/1408.5810.pdf

def icomp_penalty(cov, method='approx'):
    # Notes: used with -2*pll+2*icomp, cov is 2d, approx is more like an upper bound
    assert len(cov.shape) == 2
    var = np.diag(cov)
    if len(var) <= 1:
        return 0
    if method == 'exact':
        icomp = (np.log(var).sum()-np.log(np.linalg.det(cov)))/2
    else:
        ev = np.linalg.eigvals(cov)
        gm = gmean(ev).real
        am = np.mean(ev).real
        icomp = len(var)*np.log(np.divide(am, gm))/2
    return icomp

def OLS_icomp(X, y, X_cov=None):
    if X_cov is None: 
        X_cov = np.cov(X)
    ols_res = sm.OLS(y, X).fit()
    pll = ols_res.llf
    icomp = icomp_penalty(X_cov, method='approx')
    return 2*(icomp-pll)

class KRR(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.nobs = None
        self.A = None
        self.dual_coef_ = None
        self.train_residual = None
        self.pll = self.kic_1 = self.kic_2 = None
        
    # K = pairwise_kernels(X, metric='rbf')
    def fit(self, K, y):
        self.nobs = len(y)
        self.A = K + self.alpha * np.eye(len(K))
        self.dual_coef_ = solve(self.A, y, assume_a='sym')
        self.train_residual = y - self.predict(K)
        self.kic(K, self.dual_coef_, self.train_residual, self.A)
        return self
        
    def predict(self, K):
        return np.dot(K, self.dual_coef_)
    
    def kic(self, K, theta, y_res, A):
        nobs2 = self.nobs/2
        var = np.dot(y_res, y_res)
        tKt = theta.dot(K).dot(theta)
        ss = (var + self.alpha*tKt)/self.nobs
        # A = np.linalg.inv(A)
        A = np.linalg.pinv(A, hermitian=True) # pinv as an option
        sigma_theta = K.dot(A.dot(A))
        self.pll = -nobs2*np.log(2*np.pi*ss)-(var+self.alpha*tKt)/(2*ss)
        self.kic_1 = -2*self.pll+ss*np.trace(sigma_theta)
        self.kic_2 = -2*self.pll+ss*np.trace(sigma_theta.dot(sigma_theta))

