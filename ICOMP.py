import numpy as np
import statsmodels.api as sm
from scipy.stats import gmean
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels

def icomp_penalty(cov, method='approx'):
    # used with -2*pll+2*icomp
    var = np.diag(cov)
    if method == 'exact':
        icomp = (np.log(var).sum()-np.log(np.linalg.det(cov)))/2
    else:
        ev = np.linalg.eigvals(cov)
        # gm = ((ev.prod())**(1/len(ev))).real
        gm = gmean(ev).real
        am = np.mean(ev).real
        icomp = len(var)*np.log(am/gm)/2
    return icomp

def OLS_icomp(X, y, X_cov=None):
    if X_cov is None: 
        X_cov = np.cov(X)
    ols_res = sm.OLS(y, X).fit()
    pll = ols_res.llf
    icomp = icomp_penalty(X_cov, method='approx')
    return 2*(icomp-pll)

class KRR(object):
    def __init__(self, alpha=1.0):
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
        self.dual_coef_ = scipy.linalg.solve(self.A, y, assume_a='sym')
        self.train_residual = y - self.predict(K)
        self.kic(K, self.dual_coef_, self.train_residual, self.A)
        return self
        
    def predict(self, K):
        return np.dot(K, self.dual_coef_)
    
    def kic(self, K, theta, y_res, A):
        nobs2 = self.nobs/2
        var = np.dot(y_res, y_res)
        tKt = theta.dot(K).dot(theta)
        ss = (var + krr.alpha*tKt)/self.nobs
        A = np.linalg.inv(A)
        sigma_theta = K.dot(A.dot(A))
        self.pll = -nobs2*np.log(2*np.pi*ss)-(var+krr.alpha*tKt)/(2*ss)
        self.kic_1 = -2*self.pll+ss*np.trace(sigma_theta)
        self.kic_2 = -2*self.pll+ss*np.trace(sigma_theta.dot(sigma_theta))

