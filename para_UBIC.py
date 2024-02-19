from functools import partial
import numpy as np
from numpy.linalg import norm as Norm
from scipy.signal import periodogram, welch
from sklearn.linear_model import BayesianRidge
from derivative import dxdt as ddd

def ssr2llf(ssr, nobs, epsilon=1e-5):
    nobs2 = nobs / 2.0
    # llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
    llf = -nobs2*np.log(2*np.pi*ssr/nobs+epsilon)
    return llf

def rss2bic(rss, nparams, nobs, epsilon=1e-5):
    # llf = ssr2llf(rss, nobs, epsilon)
    # return -2*llf + np.log(nobs)*nparams
    return nobs*np.log(2*np.pi*rss/nobs+epsilon) + np.log(nobs)*nparams

# whenever u compute Ut_grouped - Ut_grouped_est then u can also compute transform_func(Ut_grouped) - transform_func(Ut_grouped_est)
def rss_group(Ut_grouped_est, Ut_grouped, transform_func=lambda _: _):
    Ut_grouped_diff = transform_func(Ut_grouped) - transform_func(Ut_grouped_est)
    rss = [np.linalg.norm(Ut_grouped_diff[j])**2 for j in range(len(Ut_grouped_diff))]
    return np.sum(rss)

def BIC_Loss(As,bs,x,epsilon=1e-5):
    # D: Number of candidates | m: either len(t) or len(x) (temporal or spatial group)
    D,m = x.shape
    # n: Number of horizon
    n,_ = As[0].shape
    N = n*m
    # Complexity
    k = np.count_nonzero(x)/m
    # BIC
    res = np.vstack([bs[j] - As[j]@x[:, j:j+1] for j in range(m)])
    assert len(res) == n*m
    rss = np.linalg.norm(res, ord='fro')**2
    llf = ssr2llf(rss, N, epsilon)
    # llf = -(N/2)*np.log(2*np.pi*rss/N+epsilon)
    # -2*llf + np.log(N)*k # AIC: -2*llf + 2*k
    # return -2*llf + np.log(N)*k
    return N*np.log(2*np.pi*rss/N+epsilon) + np.log(N)*k

def smooth_data(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ,dtype=int), 'valid')/WSZ    
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start , out0, stop))

def remove_f(uu, percent):
    if percent <= 0: return uu
    PSD = (uu*np.conj(uu))/np.prod(uu.shape)
    PSD = PSD.real
    mask = (PSD>np.percentile(PSD, percent)).astype(np.float32)
    return uu*mask

def power_spectral_density(*args, **kwargs):
    method = kwargs["method"]
    del kwargs["method"]
    if method.lower() == 'welch':
        return welch(*args, **kwargs)[1]
    return periodogram(*args, **kwargs)[1]

def sample_uncertainty(X_std, y_std=None):
    sample_weight = X_std ** 2
    sample_weight /= sample_weight.max(0)
    sample_weight = sample_weight.sum(1)
    if y_std is not None:
        sample_weight += y_std / y_std.max()
    sample_weight = 1 / sample_weight
    return sample_weight

def BayeRidge(X_pre, y_pre, fit_intercept=False, sample_weight=True):
    if sample_weight: sample_weight = sample_uncertainty(X_pre, y_pre)
    else: sample_weight = None
    return BayesianRidge(fit_intercept=fit_intercept).fit(X_pre, y_pre.flatten(), sample_weight=sample_weight)

def construct_group_linear_system(u, x, t, diff_kwargs, D=4, P=3, include_bias=True, dependent='spatial'):
    x_axis = 0
    t_axis = 1
    if dependent == 'temporal':
        x_axis, t_axis = t_axis, x_axis
        u = u.T

    D = max(D, 1)
    P = max(P, 1)
    differentiator_x = partial(ddd, axis=x_axis, **diff_kwargs)
    differentiator_t = partial(ddd, axis=t_axis, **diff_kwargs)

    u_t = differentiator_t(u, t)
    phi1 = [u**i for i in range(0, P+1)]
    phi1_names = ["", "u"]
    if P > 1:
        phi1_names = phi1_names + [f"u^{p}" for p in range(2, P+1)]
    assert len(phi1) == len(phi1_names)
    phi2 = [differentiator_x(u, x)]
    for _ in range(D-1):
        phi2.append(differentiator_x(phi2[-1], x))
    phi2_names = [f"u_{'x'*d}" for d in range(1, D+1)]
    assert len(phi2) == len(phi2_names)

    phi = phi1.copy(); phi_names = phi1_names.copy()
    phi.extend([(p1*p2) for p1 in phi1 for p2 in phi2])
    phi = np.array(phi)
    phi_names.extend([p1+p2 for p1 in phi1_names for p2 in phi2_names])
    phi_names = np.array(phi_names)

    Theta_grouped = np.moveaxis(phi, 0, -1)
    if not include_bias:
        Theta_grouped = Theta_grouped[:, :, 1:]
        phi_names = phi_names[1:]
    Theta = Theta_grouped.reshape(-1, len(phi_names))

    Ut = u_t.reshape(-1, 1)
    Ut_grouped = np.expand_dims(u_t, -1)

    return Ut, Theta, Ut_grouped, Theta_grouped, phi_names

def construct_fft_group(Theta_grouped, Ut_grouped, fft_percent=90):
    fft_Theta_grouped = np.array([remove_f(fft(Theta_grouped[:, :, k]), fft_percent)
                                  for k in range(Theta_grouped.shape[-1])])
    fft_Theta_grouped = np.moveaxis(fft_Theta_grouped, 0, -1)
    fft_Ut_grouped = remove_f(fft(Ut_grouped[:, :, 0]), fft_percent)
    fft_Ut_grouped = np.expand_dims(fft_Ut_grouped, -1)
    return fft_Theta_grouped, fft_Ut_grouped

