# Author: Pongpisit Thanasutives (Osaka university)
# ipython nbconvert notebook.ipynb --to script

import os; os.environ['KMP_DUPLICATE_LIB_OK']='True'
import platform; PY_VERSION = int(platform.python_version()[2])
print("Running Python", platform.python_version())

from datetime import date

import pickle
from glob import glob as flist
from collections import Counter, OrderedDict

# This is not a good import.
# from sympy import *
from sympy import Symbol, Integer, Float, Add, Mul, Lambda, simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.core import evaluate
from sympytorch import SymPyModule
import sympytorch

### Model-related imports ###
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
# from better_lstm import LSTM as BLSTM

import math
if PY_VERSION >= 8: from statistics import multimode
else: from statistics import mode
import numpy as np
from numpy import array as npar
from numpy import linalg
print("You can use npar for np.array")
import pandas as pd
from sklearn.metrics import *
from sklearn.preprocessing import PolynomialFeatures
# from pyGRNN import feature_selection as FS

# import pcgrad
# from pytorch_stats_loss import torch_wasserstein_loss, torch_energy_loss 

# Finite difference method
from findiff import FinDiff, coefficients, Coefficient

# STRidge
# from pde_diff import print_pde

# statsmodels
import statsmodels.api as sm

from pyDOE import lhs

from scipy.signal import wiener

def today():
    return date.today().strftime("%Y%m%d")

def common_element(a_list):
    tv = Counter(a_list)
    return max(tv, key=tv.get)

def mymode(a_list): 
    global PY_VERSION
    if PY_VERSION >= 8: return multimode([f[0] for f in a_list if len(f)>0])[0]
    else: return common_element([f[0] for f in a_list if len(f)>0])

def tuple_append(tup, ele):
    return tuple(list(tup).append(ele))

def search_files(directory='.', extension=''):
    extension = extension.lower()
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if extension and name.lower().endswith(extension):
                print(os.path.join(dirpath, name))
            elif not extension:
                print(os.path.join(dirpath, name))

## Saving ###
def pickle_save(obj, path):
    for i in range(2):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    print('Saved to', str(path))
    data = pickle_load(path)
    print('Test loading passed')
    del data

### Loading ###
def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print('Loaded from', str(path))
    return obj

def build_exp(program, trainable_one=True):
    x = Symbol("x"); y = Symbol("y")
    
    local_dict = {
        "add": Add,
        "mul": Mul,
        "sub": Lambda((x, y), x - y),
        "div": Lambda((x, y), x/y),
    }
    
    exp = simplify(parse_expr(str(program), local_dict=local_dict))
    if trainable_one:
        exp = exp.subs(Integer(-1), Float(-1.0, precision=53))
        exp = exp.subs(Integer(+1), Float(1.0, precision=53))
    variables = exp.atoms(Symbol)
    
    return exp, variables

# My version of sympytorch.SymPyModule
class SympyTorch(nn.Module):
    def __init__(self, expressions):
        super(SympyTorch, self).__init__()
        self.mod = sympytorch.SymPyModule(expressions=expressions)                                                                      
    def forward(self, gd):
        return torch.squeeze(self.mod(**gd), dim=-1)
    
def string2sympytorch(a_string):
    expr, variables = build_exp(a_string)
    return SympyTorch(expressions=[expr]), variables

def manipulate_expr(expr):
    for coeff in expr.atoms(Number):
        if coeff < 0.005 and coeff > 0:
            new_coeff = log(coeff)
            with evaluate(False):
                new_coeff = exp(new_coeff)
                expr = expr.subs(coeff, new_coeff)
    with evaluate(False):
        return expr

def inverse_dict(dic):
    return {v: k for k, v in dic.items()}

def string2int(s):
    out = 0
    for i in range(len(s)):
        out += ord(s[i])
    return out

def scientific2string(x):
    return format(x, '.1e')

def convert_listoftuples_dict(tup, di={}):
    for a, b in tup: di[a]=b
    return di

def dimension_slicing(a_tensor):
    c = a_tensor.shape[-1]
    out = []
    for i in range(1, c+1): out.append(a_tensor[:, i-1:i])
    return out

def cat(*args):
    return torch.cat(args, dim=-1)

def cat_numpy(*args):
    return np.hstack(args)

def get_feature(a_tensor, dim):
    return a_tensor[:, dim:dim+1]

def see_params(a_mod):
    return [ele.detach() for ele in a_mod.parameters()]

def cpu_load(a_path):
    return torch.load(a_path, map_location="cpu")

def gpu_load(a_path):
    return torch.load(a_path, map_location="cuda")

def load_weights(a_model, a_path, mode="cpu"):
    if mode=="cpu": sd = cpu_load(a_path)
    elif mode=="gpu": sd = gpu_load(a_path)
    try:
        a_model.load_state_dict(sd, strict=True)
        print("Loaded the model's weights properly")
    except: 
        try: 
            a_model.load_state_dict(sd, strict=False)
            print("Loaded the model's weights with strict=False")
        except:
            print("Cannot load the model' weights properly.")
    return a_model

def save(a_model, path):
    return torch.save(a_model.state_dict(), path)

def is_nan(a_tensor):
    return torch.isnan(a_tensor).any().item()

def to_column_vector(arr):
    return arr.flatten()[:, None]

def to_tensor(arr, g=True):
    return torch.tensor(arr).float().requires_grad_(g)

def to_complex_tensor(arr, g=True):
    return torch.tensor(arr, dtype=torch.cfloat).requires_grad_(g)

def cabs(e): 
    return torch.sqrt(e.real**2 + e.imag**2)

def to_numpy(a_tensor):
    return a_tensor.detach().numpy()

def perturb(a_array, intensity=0.01, noise_type="normal", overwrite=True):
    if intensity <= 0.0: return a_array
    if noise_type == "normal": 
        noise = intensity*np.std(a_array)*np.random.randn(a_array.shape[0], a_array.shape[1])
    elif noise_type == "uniform": 
        # This is hard...
        noise = intensity*np.std(a_array)*np.random.uniform(a_array.shape[0], a_array.shape[1])
    elif noise_type == "sparse": 
        noise = np.random.randn(a_array.shape[0], a_array.shape[1])
        mask = np.random.uniform(0, 1, (a_array.shape[0], a_array.shape[1]))
        # Makes the noisy problem easier?
        sparsemask = np.where(mask>0.9, 1, 0)
        noise = intensity*np.std(u)*noise*sparsemask
    else: 
        print("Not recognized noise_type")
        noise = 0.0
    if overwrite: return a_array + noise
    else: return noise

# This function assumes that each dimension (variable) is independent from each other.
def perturb2d(a_array, intensity, overwrite=True):
    out = np.copy(a_array)
    for i in range(a_array.shape[1]):
        out[:, i:i+1] = perturb(a_array[:, i:i+1], intensity=intensity, overwrite=overwrite)
    return out

def sampling_unit_circle(N):
    points = []
    for _ in range(N):
        length = np.sqrt(np.random.uniform(0, 1)); angle = np.pi * np.random.uniform(0, 2)
        points.append([length * np.cos(angle), length * np.sin(angle)])
    return np.array(points)

def sampling_from_rows(a_tensor, N, return_idxs=False):
    r = a_tensor.shape[0]
    idxs = np.random.choice(r, N, replace=False)
    if return_idxs: return idxs
    else: return a_tensor[idxs, :]

def lhs_sampling_from_matrix(inp, nsamples):
    if type(inp)==torch.Tensor:
        upper_bound = inp.max(axis=0).values.detach().numpy()
        lower_bound = inp.min(axis=0).values.detach().numpy()
    else:
        upper_bound = inp.max(axis=0)
        lower_bound = inp.min(axis=0)
    return lhs_sampling((nsamples, inp.shape[1]), lower_bound, upper_bound)

def lhs_sampling(dimensions, lower_bound, upper_bound):
    return lower_bound+(upper_bound-lower_bound)*lhs(dimensions[1], dimensions[0])

# is the mini and maxi is going to be dynamics during the training but not trainable because of there is no grad.
def minmax_normalize(features):
    mini = torch.min(features, axis=0)[0]
    maxi = torch.max(features, axis=0)[0]
    features_std = (features-mini) / (maxi-mini) 
    return features_std

def scale_to_range(features, lb, ub):
    scaled_features = minmax_normalize(features)
    scaled_features = (ub-lb)*scaled_features + lb  
    return scaled_features

# is the mini and maxi is going to be dynamics during the training but not trainable because of there is no grad.
def numpy_minmax_normalize(arr):
    mini = np.min(arr, axis=0)
    maxi = np.max(arr, axis=0)
    return (arr-mini)/(maxi-mini)

def numpy_scale_to_range(arr, lb, ub):
    scaled_arr = numpy_minmax_normalize(arr)
    scaled_arr = (ub-lb)*scaled_arr + lb
    return scaled_arr

def cap_values(a_tensor, lb, ub):
    return (a_tensor-lb)/(ub-lb)

def binarize(a_tensor, th):
    indices = a_tensor > th
    out = a_tensor * indices
    return (out + (1-out)) * indices

def diff_order(dterm):
    return dterm.split("_")[-1][::-1]

def diff_flag(index2feature):
    dd = {0:[], 1:[]}
    for t in index2feature:
        if '_' not in t: dd[0].append(t)
        else: dd[1].append(diff_order(t))
    return dd

def diff(func, inp):
    return grad(func, inp, create_graph=True, retain_graph=True, grad_outputs=torch.ones(func.shape, dtype=func.dtype))[0]

def gradients_dict(u, x, t, feature_names):
    grads_dict = {}
    df = diff_flag(feature_names)
    if feature_names[0].split('_')[0] == 'h': h = u
    
    for e in df[0]:
        grads_dict[e] = eval(e)
        
    for e in df[1]:
        out = u
        for c in e: out = diff(out, eval(c))
        grads_dict['u_'+e[::-1]] = out
        
    return grads_dict

# Fast group derivatives implementation
def group_diff(dependent_var, independent_vars, feature_names, function_notation="u", gd_init={}):
    xxx, ttt = independent_vars

    char = ""; fn = feature_names
    cal_terms = ['' for i in range(len(feature_names))]
    set_cal_terms = set()
    grads_dict = gd_init

    MAX_ITER = sum([len(e) for e in feature_names])
    for _ in range(MAX_ITER):
        if sum([len(f) for f in fn]) == 0: break
        
        char = mymode(fn)
        
        new_fn = []; new_set_cal_terms = set()
        for i, f in enumerate(fn):
            if len(f) > 0 and f[0] == char:
                new_fn.append(f[1:])
                cal_terms[i] += f[0]
                if cal_terms[i] not in set_cal_terms:
                    new_set_cal_terms.add(cal_terms[i])
            else: new_fn.append(f)
                
        fn = new_fn
        
        # Computing the actual derivatives here
        for e in new_set_cal_terms:
            if e not in grads_dict:
                prev = function_notation+"_"+e[:-1]
                now = function_notation+"_"+e

                if len(e) == 1:
                    if e == 'x': grads_dict[now] = diff(dependent_var, xxx)
                    elif e == 't': grads_dict[now] = diff(dependent_var, ttt)
                        
                elif prev in grads_dict:
                    if e[-1] == 'x': grads_dict[now] = diff(grads_dict[prev], xxx)
                    elif e[-1] == 't': grads_dict[now] = diff(grads_dict[prev], ttt)
                        
                else: raise Exception("The program is not working properly.")
        
        set_cal_terms = set_cal_terms.union(new_set_cal_terms)

    return grads_dict

# Careful that there is no delta[i] = 0
# Use this function to approximate a higher-order derivative
def fd_diff(func, xx, dim=0):
    return torch.diff(func, dim=dim, n=1)/(torch.diff(xx, dim=dim))

def complex_diff(func, inp, return_complex=True):
    if return_complex: return diff(func.real, inp)+1j*diff(func.imag, inp)
    else: return cat(diff(func.real, inp), diff(func.imag, inp))

def finite_diff(func, axis, delta, diff_order=1, acc_order=2):
    assert axis in range(len(func.shape))
    return FinDiff(axis, delta, diff_order, acc=acc_order)(func)

class FinDiffCalculator:
    def __init__(self, X, T, Exact, dx=None, dt=None, acc_order=2):
        print("Do not use this class with complex-valued input arrays.")
        print("This class applies 1 transpose to the Exact before doing the job.")
        self.X = X; self.T = T; self.Exact = Exact.T

        if dx is not None: self.dx = dx
        else: self.dx = self.X[0, :][1]-self.X[0, :][0]
        if self.dx == 0.0: self.dx = self.X[:, 0][1]-self.X[:, 0][0]
        print('dx =', self.dx)

        if dt is not None: self.dt = dt
        else: self.dt = self.T[:, 0][1]-self.T[:, 0][0]
        if self.dt == 0.0: self.dt = self.T[0, :][1]-self.T[0, :][0]
        print('dt =', self.dt)

        self.deltas = [self.dx, self.dt]
        self.acc_order = acc_order
    # Cal d_dt using this function
    def finite_diff(self, axis, diff_order=1):
        return to_column_vector(FinDiff(axis, self.deltas[axis], diff_order, acc=self.acc_order)(self.Exact).T)
    def finite_diff_from_feature_names(self, index2feature):
        out = {}
        for f in index2feature:
            if '_' not in f:
                if f == 'uf' or f == 'hf': out[f] = to_column_vector(self.Exact.T)
                elif f == 'x': out[f] = to_column_vector(self.X)
                elif f == '|uf|' or f == '|hf|': out[f] = to_column_vector((self.Exact.real**2+self.Exact.imag**2).T)
                else: raise NotImplementedError
            else:
                counter = Counter(f.split('_')[1])
                if len(counter.keys())==1 and 'x' in counter.keys():
                    out[f] = (self.finite_diff(axis=0, diff_order=counter['x']))
                else: raise NotImplementedError
        return out

def train_val_split(a_tensor, train_ratio=0.8):
    train_len = int(0.8*a_tensor.shape[0])
    val_len = a_tensor.shape[0]-train_len
    train_idx, val_idx = torch.utils.data.random_split(np.arange(a_tensor.shape[0]), lengths=[train_len, val_len])
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    return a_tensor[train_idx], a_tensor[val_idx]

# this function is supposed to be used with the TrainingDataset class
def get_dataloader(X_train, y_train, bs, N_sup=2000):
    return DataLoader(TrainingDataset(X_train, y_train, N_sup=N_sup), batch_size=bs)

# simple dataset class containing pair (x, y)
class XYDataset(Dataset):
    def __init__(self, X_data, y_data):
        super(XYDataset, self).__init__()
        assert X_data.shape[0] == y_data.shape[0]
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.X_data[idx, :], self.y_data[idx, :]

    def __len__(self,):
        return self.X_data.shape[0]

class TrainingDataset(Dataset):
    def __init__(self, X_train, y_train, N_sup=2000):
        super(TrainingDataset, self).__init__()
        self.X = X_train
        self.y = y_train
        self.N_sup=N_sup
        
    def __getitem__(self, index):
        if index > self.N_sup-1:
            return self.X[index], self.y[index]
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]
    
class LadderLoss(nn.Module):
    def __init__(self, return_list=False):
        super().__init__()
        self.return_list = return_list
        
    def forward(self, outputs, labels):
        valid_index = torch.where(~(labels<-999))[0]
        tmp_out = outputs[0][valid_index]
        tmp_labels = labels[valid_index]
        unsup_loss = outputs[1]
        mse_loss = 0.0
        if tmp_out.shape[0] > 0 and tmp_labels.shape[0] > 0:
            mse_loss = F.mse_loss(tmp_out, tmp_labels)
        if not self.return_list: return mse_loss+unsup 
        return [mse_loss, unsup_loss] 

class LadderUncertLoss(nn.Module):
    def __init__(self, n_task):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros((n_task)))
 
    def forward(self, outputs, labels):
        weights = torch.exp(self.log_vars)
        mse_loss = F.mse_loss(outputs[0], labels).unsqueeze(0)
        unsup_loss = outputs[1].unsqueeze(0)
        losses = torch.cat([mse_loss, unsup_loss])
        return weights.dot(losses)

def kl_normal(p, q):
    input_distribution = torch.distributions.Normal(p.mean(), p.std())
    target_distribution = torch.distributions.Normal(q.mean(), q.std())
    return torch.distributions.kl_divergence(input_distribution, target_distribution).mean().item()

def percent_error(sig, ground):
    ct = type(1j)
    sig = npar(sig).flatten()
    ground = npar(ground).flatten()
    if type(sig) == (type(ground) == ct): return 100*linalg.norm(np.abs(sig-ground), 1)/linalg.norm(np.abs(ground), 1)
    else: return 100*linalg.norm(np.abs(sig-ground), 1)/linalg.norm(ground, 1)
    
def percent_coeffcient_error(sig, ground):
    # This function is for testing a single effective coefficient.
    return np.abs(100*(sig-ground)/np.abs(ground))

def complex_l1_norm(cc):
    return np.abs(cc.real) + np.abs(cc.imag)

def relative_l2_error(sig, ground):
    return linalg.norm((sig-ground), 2)/linalg.norm(ground, 2)

def log_like_value(prediction, ground):
    nobs = float(ground.shape[0])
    nobs2 = nobs / 2.0
    ssr = np.sum(np.abs(ground - prediction)**2)
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
    return llf

def BIC_AIC(prediction, ground, nparams, reg_func = lambda x: x):
    nparams = reg_func(nparams)
    llf = log_like_value(prediction, ground)
    return -2*llf + np.log(ground.shape[0])*nparams, -2*llf + 2*nparams

# calculate bic for regression (from machine learning mastery)
def calculate_bic(n, mse, num_params):
	bic = n * np.log(mse) + num_params * np.log(n)
	return bic

# calculate aic for regression (from machine learning mastery) 
def calculate_aic(n, mse, num_params):
	aic = n * np.log(mse) + 2 * num_params
	return aic

def OLS(model, dataset, feature_names=None, print_params=True):
    X, y = dataset
    consider = np.where(np.linalg.norm(model, axis=1)>0)[0]
    X = X[:, consider]
    if feature_names is not None:
        feature_names = npar(feature_names)[consider].tolist()
        X = pd.DataFrame(X, columns=feature_names)
    res = sm.OLS(y, X).fit()
    if print_params: 
        print("Parameters = ")
        print(res.params)
        print("BIC = ", res.bic)
        print("AIC = ", res.aic)
    return res

def compare_model(current_stats, base_stats, k=1):
    '''
    This function returns the improvement to the base models when you're trying to uplift the current model complexity.
    Loop through the list with the increasing complexity, then choose the model with largest improvment.
    '''
    curMSE, curComplexity, _ = current_stats
    lastMSE, lastComplexity, _ = base_stats
    if lastComplexity >= curComplexity: 
        print("The complexity should be increasing not decreasing!")
    cur_score = -np.log(curMSE/lastMSE) / (k*curComplexity-lastComplexity) 
    return cur_score

def pysr_scores(stats):
    # stats.sort(key=lambda e:e[1])
    base_stats = stats[0]
    scores = [0.0]
    for st in stats[1:]:
        cur_score = compare_model(st, base_stats)
        if np.isnan(cur_score):
            scores.append(scores[-1])
        else:
            scores.append(cur_score)
        base_stats = st
    return scores

def rank_models(models, dataset, feature_names):
    # print("---------- Ranking models ----------")
    # rearrange the models
    compls = [len(np.nonzero(models[i])[0]) for i in range(len(models))]
    re_indices = sorted(range(len(compls)), key=lambda k: compls[k])
    models = np.squeeze(npar(models)).T[:, re_indices]
    models = [models[:, i:i+1] for i in range(models.shape[1])]
    compls.sort()

    # old code
    X, y = dataset
    preds = X@np.hstack(tuple(models))
    stats = [((mse_function(np.ones(y.shape)*y.mean(), y), 0, -1))]
    ols_models = []

    for i in range(len(models)):
        mse = mse_function(y, preds[:, i:i+1])
        complexity = len(np.nonzero(models[i])[0])
        stats.append((mse, complexity, i))
        ols_models.append(OLS(models[i], dataset, print_params=False))

    # base_bic = sm.OLS(y, np.ones(y.shape)*y.mean()).fit().bic
    base_bic = BIC_AIC(np.ones(y.shape)*y.mean(), y, 1)[0]
    print("Base BIC = ", base_bic)
    stats.sort(key=lambda e:e[1])
    scores = pysr_scores(stats)
    best_idx = np.argmax(npar(scores))
    # bic_aic = [list(BIC_AIC(res.predict(X[:, np.nonzero(models[i])[0]]).reshape(-1, 1), y, compls[i])) for i, res in enumerate(ols_models)]
    bic_aic = [list(BIC_AIC(X@models[i], y, compls[i])) for i, res in enumerate(ols_models)]

    for i in range(len(bic_aic)):
        if i == 0:
            # bic_aic[i][1] = (bic_aic[i][0]-base_bic)/(compls[i])
            bic_aic[i][1] = np.inf
        else:
            bic_aic[i][1] = (bic_aic[i][0]-bic_aic[i-1][0])/(compls[i]-compls[i-1])
    best_bic_idx = np.argmin(npar([res.bic for res in ols_models]))

    for i, st in enumerate(stats[1:]):
        w = models[st[2]]
        print_pde(w, feature_names)
        print("PySR score:", scores[i+1])
        if i+1 == best_idx: print("Based on the PySR score, this is likely to be the actual hidden PDE.")
        print("(BIC, RC_BIC) =", bic_aic[i])
        if i == best_bic_idx: print("The BIC best model!")
        print("="*50)

    return stats, scores

# def distance_loss(inputs, targets, distance_function=torch_wasserstein_loss):
#      total_loss = 0.0
#      assert inputs.shape == targets.shape
#      for i in range(inputs.shape[1]):
#          total_loss += distance_function(inputs[:, i], targets[:, i])
#      return total_loss

### Model-related code base ###
class CrossStich(nn.Module):
    def __init__(self,):
        super(CrossStich, self).__init__()
        self.transform = nn.Parameter(data=torch.eye(2), requires_grad=True)
    def forward(self, input_1, input_2):
        return self.transform[0][0]*input_1 + self.transform[0][1]*input_2, self.transform[1][0]*input_1 + self.transform[1][1]*input_2
    
def sparse_layer(in_dim, out_dim, sparsity):
    return SparseWeights(nn.Linear(in_dim, out_dim), sparsity=sparsity)

# This should be the activation function for the selector network. The module outputs a feature masking tensor.
class ThresholdSoftmax(nn.Module):
    def __init__(self, th=0.1):
        super(ThresholdSoftmax, self).__init__()
        self.sm = nn.Softmax(dim=-1)
        self.th = th
        self.prob = None
    def forward(self, inn):
        self.prob = self.sm(inn).mean(dim=0)
        thres = self.prob[torch.argsort(self.prob, descending=True)[3]]
        self.prob = torch.where(self.prob > thres, self.prob, torch.tensor([self.th]).float()) 
        samples = torch.sort(torch.multinomial(self.prob, 3))[0]
        return samples

class Swish(nn.Module):
    def __init__(self,):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x*torch.sigmoid(x)

class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

def simple_solver_model(hidden_nodes):
    model = nn.Sequential(nn.Linear(2, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, hidden_nodes),
                        nn.Tanh(),
                        nn.Linear(hidden_nodes, 1))
    return model 

def evaluate_network_mse(network, X_star, u_star):
    return ((network(X_star[:, 0:1], X_star[:, 1:2]).detach() - u_star)**2).mean().item()

def evaluate_ladder_network_mse(network, X_star, u_star):
    return ((network(X_star[:, 0:1], X_star[:, 1:2])[0].detach() - u_star)**2).mean().item()

class TorchMLP(nn.Module):
    def __init__(self, dimensions, bias=True,activation_function=nn.Tanh, bn=None, dropout=None, inp_drop=False, final_activation=None, init_weights="uniform"):
        super(TorchMLP, self).__init__()
        print("Using old implementation of TorchMLP. See models.py for more new model-related source code.")
        self.model  = nn.ModuleList()
        # Can I also use the LayerNorm with elementwise_affine=True
        # This should be a callable module.
        self.activation_function = activation_function()
        self.bn = bn
        if dropout is not None and inp_drop: self.inp_dropout = dropout
        else: self.inp_dropout = None
        self.init_weights = init_weights
        for i in range(len(dimensions)-1):
            self.model.append(nn.Linear(dimensions[i], dimensions[i+1], bias=bias))
            if self.bn is not None and i!=len(dimensions)-2:
                self.model.append(self.bn(dimensions[i+1]))
                if dropout is not None:
                    self.model.append(dropout)
            if i==len(dimensions)-2: break
            self.model.append(activation_function())
        if final_activation is not None:
            self.model.append(final_activation())
        self.model.apply(self.xavier_init)

    def xavier_init(self, m):
        if type(m) == nn.Linear:
            if self.init_weights == "normal": torch.nn.init.xavier_normal_(m.weight)
            else: torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        if hasattr(self, 'inp_dropout'):
            if self.inp_dropout is not None:
                x = self.inp_dropout(x)
        for i, l in enumerate(self.model): 
            x = l(x)
        return x

class SklearnModel:
    def __init__(self, model, X_train=None, y_train=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.is_train = False
        self.X_train_shape = None 
        if X_train is not None and y_train is not None:
            self.train(X_train, y_train)
        self.feature_importances = None
    def train(self, X_np, y_np):
        if self.is_train:
            print("This model was already trained")
        else:
            self.model.fit(X_np, y_np)
            self.X_train_shape = X_np.shape
            print("Done training")
            print("Training MSE:", mean_squared_error(self.model.predict(X_np), y_np))
    def feature_importance(self, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
        if self.feature_names is None:
            self.feature_names = [str(i) for i in range((self.X_train_shape[1]))]
        ranking = np.argsort(self.model.feature_importances_)[::-1]
        total = sum(self.model.feature_importances_)
        out = []
        for i in ranking: 
            print((self.feature_names[i], self.model.feature_importances_[i]/total))
            out.append((self.feature_names[i], self.model.feature_importances_[i]/total))
        self.feature_importances = convert_listoftuples_dict(out)
        return self.feature_importances
    def test(self, X_test, y_test, metric=None):
        y_pred = self.model.predict(X_test)
        if not metric: return mean_squared_error(y_test, y_pred) 
        else: return metric(y_test, y_pred)

class ComplexPolynomialFeatures:
    def __init__(self, feature_names, dictionary):
        self.feature_names = feature_names
        self.dictionary = dictionary
        self.poly_feature_names = PolynomialFeatures(include_bias=True).fit(np.array(list(dictionary.values())).squeeze(-1).T.real).get_feature_names(self.feature_names)
        self.output = np.ones(self.dictionary[self.feature_names[0]].shape, dtype=np.complex64)

    def fit(self,):
        for f in self.poly_feature_names[1:]:
            print("Computing", f)
            self.output = np.hstack((self.output, compute_from_description(f, self.dictionary)))
        return self.output

def compute_from_description(description, dictionary, split_keys=(" ", "^")):
    terms = description.split(split_keys[0]); out = 1
    for t in terms:
        if t in dictionary:
            out = out*dictionary[t]
        else:
            t, deg = t.split(split_keys[1])
            out = out*dictionary[t]**int(deg)
    return out

def mse_function(v1, v2):
    return ((np.abs(v1-v2))**2).mean()

# calculate aic for regression
def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic

def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

def occam_razor(scores):
    mse_performances = [e[0] for e in scores]
    complexities = [e[1] for e in scores]
    max_mse = max(mse_performances)
    max_complexity = 10*max(complexities)
    return [-math.log((e[0]/max_mse)/(max_complexity-e[1])) for e in scores]

def pyGRNN_feature_selection(X, y, feature_names):
    IsotropicSelector = FS.Isotropic_selector(bandwidth='rule-of-thumb')
    return IsotropicSelector.feat_selection((X), (y).ravel(), feature_names=feature_names, strategy ='es')

def change_learning_rate(a_optimizer, lr):
    for g in a_optimizer.param_groups:
        g['lr'] = lr
    return a_optimizer

def pcgrad_update(model, model_optimizer, losses):
    updated_grads = []
    
    for i in range(2):
        model_optimizer.zero_grad()
        losses[i].backward(retain_graph=True)

        g_task = []
        for param in model.parameters():
            if param.grad is not None:
                g_task.append(Variable(param.grad.clone(), requires_grad=False))
            else:
                g_task.append(Variable(torch.zeros(param.shape), requires_grad=False))
        # appending the gradients from each task
        updated_grads.append(g_task)

    updated_grads = list(pcgrad.pc_grad_update(updated_grads))[0]
    for idx, param in enumerate(model.parameters()): 
        param.grad = (updated_grads[0][idx]+updated_grads[1][idx])

    return model, model_optimizer, sum(losses)

def create_data_for_feynman(G, target, filename):
    if len(target.shape)>1:
        target = np.squeeze(target) 
    with open(filename, "w") as file:
        for row in range(G.shape[0]):
            string_out = ''
            for col in range(G.shape[1]):
                string_out += str(G[row][col]) + ' '
            string_out += str(target[row])
            if row == G.shape[0]-1:
                file.write(string_out) 
            else:
                file.write(string_out+'\n') 
    print("Done writing into the file")
    file.close()

# pytorch version of fft denoisg algorithm.
def fft1d_denoise(signal, thres=None, c=0, return_real=True):
    signal = signal.flatten()
    n = len(signal)
    fhat = torch.fft.fft(signal, n)
    PSD = (fhat.real**2 + fhat.imag**2) / n
    if thres is None: thres = (PSD.mean() + c*PSD.std()).item()
    indices = PSD > thres
    fhat = indices * fhat
    out = torch.fft.ifft(fhat)
    if return_real: out = out.real
    return out.reshape(-1, 1), fhat, PSD

# numpy version of fft denoising algorithm.
def fft1d_denoise_numpy(signal, thres=None, c=0, return_real=True):
    signal = signal.flatten()
    n = len(signal)
    fhat = np.fft.fft(signal, n)
    PSD = (fhat.real**2 + fhat.imag**2) / n
    if thres is None: thres = (PSD.mean() + c*PSD.std())
    indices = PSD > thres
    fhat = indices * fhat
    out = np.fft.ifft(fhat)
    if return_real: out = out.real
    return out.reshape(-1, 1), fhat, PSD

def wiener_noise(sig, noise_intensity=None):
    sig_recon = wiener(sig, noise=noise_intensity)
    return sig - sig_recon

class FFTNN(nn.Module):
    def __init__(self, c=0.0, minmax=(-5.0, 5.0)):
        super(FFTNN, self).__init__()
        self.c = nn.Parameter(data=torch.FloatTensor([float(c)]))
        self.mini = minmax[0]
        self.maxi = minmax[1]

    def forward(self, PSD):
        # Old implementation: th = PSD.mean()+torch.clamp(self.c, min=self.mini, max=self.maxi)*PSD.std()
        th = F.relu(PSD.mean()+torch.clamp(self.c, min=self.mini, max=self.maxi)*PSD.std())
        indices = F.relu(PSD-th)
        d = torch.ones_like(indices)
        d[indices>0] = indices[indices>0]
        indices = indices / d
        return indices

class FFTLSTM(nn.Module):
    def __init__(self, n_hiddens=1, num_layers=1, bidirectional=False, dropoutw=0.2, minmax=(-5.0, 5.0)):
        super(FFTLSTM, self).__init__()
        self.lstm = BLSTM(1, n_hiddens, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropoutw=dropoutw)
        self.fc = nn.Linear(n_hiddens, 1)
        self.mini = minmax[0]
        self.maxi = minmax[1]

    def forward(self, PSD):
        m, s = PSD.mean(), PSD.std()
        normalized_PSD = (PSD-m)/s
        out_seq, (h_state, _) = self.lstm(normalized_PSD.view(1, len(PSD), 1))
        h_state = self.fc(h_state[-1])
        th = F.relu(m+torch.clamp(h_state, min=self.mini, max=self.maxi)*s)
        indices = F.relu(PSD-th)
        d = torch.ones_like(indices)
        d[indices>0] = indices[indices>0]
        indices = indices / d
        return indices

class FFTConv(nn.Module):
    def __init__(self, seq_len, n_hiddens=None, n_channels=1, kernel_size=1, minmax=(-5.0, 5.0)):
        super(FFTConv, self).__init__()
        if n_hiddens is None: n_hiddens = seq_len
        self.conv = nn.Sequential(nn.Conv1d(1, n_channels, kernel_size, padding='same'), nn.AdaptiveMaxPool1d(n_hiddens))
        self.fc = nn.Linear(n_hiddens, 1)
        self.mini = minmax[0]
        self.maxi = minmax[1]

    def forward(self, PSD):
        m, s = PSD.mean(), PSD.std()
        normalized_PSD = (PSD-m)/s
        # also a good choice: h_state = F.silu(self.conv(normalized_PSD.view(1, 1, len(PSD)))).flatten()
        h_state = F.relu(self.conv(normalized_PSD.view(1, 1, len(PSD)))).flatten()
        h_state = self.fc(h_state.view(1, len(h_state)))
        th = F.relu(m+torch.clamp(h_state, min=self.mini, max=self.maxi)*s)
        indices = F.relu(PSD-th)
        d = torch.ones_like(indices)
        d[indices>0] = indices[indices>0]
        indices = indices / d
        return indices

class FFTTh(nn.Module):
    def __init__(self, c=0.9, minmax=(-5.0, 5.0), func=lambda x:x):
        super(FFTTh, self).__init__()
        self.c = nn.Parameter(data=torch.FloatTensor([float(c)]))
        self.mini = minmax[0]
        self.maxi = minmax[1]
        # self.func = lambda x:(torch.exp(-F.relu(x)))
        # self.func can return a negative value
        self.func = func

    def forward(self, PSD):
        m, s = PSD.mean(), PSD.std()
        normalized_PSD = (PSD-m)/s
        th = F.relu(m+torch.clamp(self.func(self.c)*normalized_PSD.max(), min=self.mini, max=self.maxi)*s)
        indices = F.relu(PSD-th)
        d = torch.ones_like(indices)
        d[indices>0] = indices[indices>0]
        indices = indices / d
        return indices

class ApproxL0(nn.Module):
    def __init__(self, sig=1.0, minmax=(0.0, 10.0), trainable=True):
        super(ApproxL0, self).__init__()
        self.sig = nn.Parameter(data=torch.FloatTensor([float(sig)])).requires_grad_(trainable)
        self.mini = minmax[0]
        self.maxi = minmax[1]

    def forward(self, w):
        sig = torch.clamp(self.sig, min=self.mini, max=self.maxi)
        return approx_l0(w, sig=sig)

# publication
def approx_l0(w, sig=1.0):
    sig = sig*torch.var(w)
    return len(w)-torch.exp(torch.div(-torch.square(w), 2*(torch.square(sig)))).sum()

def approx_l0_fixed(w, sig=1.0):
    sig = sig*torch.std(w)
    return len(w)-torch.exp(torch.div(-torch.square(w), 2*(torch.square(sig)))).sum()

def conditional_number(mat):
    cond = np.linalg.cond(mat)
    for i in range(1, 21):
        cond = cond / (10**i)
        if cond < 1: return cond*10
    return cond*10
