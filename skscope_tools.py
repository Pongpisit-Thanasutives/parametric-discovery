# By Pongpisit Thanasutives
import jax.numpy as jnp
import numpy as np
from skscope import ScopeSolver
from sklearn.preprocessing import normalize
from abess import LinearRegression as abess_linear
from solvel0 import refine_solvel0

def get_support(coef_vec):
    return np.where(np.abs(coef_vec)>0)[0]

def best_subset_solution(X_pre, y_pre, sparsity, p=1, normalize_order=2, normalize_axis=None, center_y=False, lstsq=False):
    XX = X_pre
    yy = y_pre.flatten()
    if center_y:
        yy = yy - yy.mean()
    if normalize_axis is not None:
        X_norm = np.linalg.norm(X_pre, ord=normalize_order, axis=normalize_axis)
        XX = np.divide(XX, X_norm)
    objective_function = lambda coefs: jnp.linalg.norm(yy-XX@coefs)**p
    scope_solver = ScopeSolver(dimensionality=XX.shape[-1], sparsity=sparsity)
    scope_solver.solve(objective_function)
    coefs = scope_solver.params
    if normalize_axis is not None:
        coefs = np.divide(coefs, X_norm)
    if lstsq:
        supports = scope_solver.get_support()
        coefs[supports] = np.linalg.lstsq(X_pre[:, supports], y_pre, rcond=None)[0][:, 0]
    return coefs

def best_subset_all_solutions(X_pre, y_pre, sparsity, p=1, normalize_order=2, normalize_axis=None, center_y=False, refine=False):
    XX = X_pre
    yy = y_pre.flatten()

    if center_y:
        yy = yy - yy.mean()

    if normalize_axis is not None:
        X_norm = np.linalg.norm(X_pre, ord=normalize_order, axis=normalize_axis)
        XX = np.divide(XX, X_norm)

    objective_function = lambda coefs: jnp.linalg.norm(yy-XX@coefs)**p
    all_coefs = []
    all_supports = []
    for sp in range(1, sparsity+1):
        scope_solver = ScopeSolver(dimensionality=XX.shape[-1], sparsity=sp)
        scope_solver.solve(objective_function)
        coefs = scope_solver.params
        if normalize_axis is not None:
            coefs = np.divide(coefs, X_norm)
        all_coefs.append(coefs)
        all_supports.append(scope_solver.get_support())
    all_coefs = np.array(all_coefs)

    if refine:
        if refine == 'original': all_supports = refine_solvel0(all_supports, (XX, yy), 'bic', False)
        else: all_supports = refine_solvel0(all_supports, (X_pre, y_pre), 'bic', False)
        all_supports = sorted([list(all_supports.track[e][0]) for e in all_supports.track], key=len)
        for i, coefs in enumerate(all_coefs):
            tmp_coef = np.zeros_like(coefs)
            tmp_coef[all_supports[i]] = np.linalg.lstsq(X_pre[:, all_supports[i]], y_pre, rcond=None)[0][:, 0]
            all_coefs[i] = tmp_coef

    return all_coefs, all_supports

def abess_solution(X_pre, y_pre, sparsity, p=1, normalize_order=2, normalize_axis=None, center_y=False, lstsq=False):
    XX = X_pre
    yy = y_pre.flatten()
    if center_y:
        yy = yy - yy.mean()
    if normalize_axis is not None:
        X_norm = np.linalg.norm(X_pre, ord=normalize_order, axis=normalize_axis)
        XX = np.divide(XX, X_norm)
    coefs = abess_linear(support_size=sparsity).fit(XX, yy).coef_
    if normalize_axis is not None:
        coefs = np.divide(coefs, X_norm)
    if lstsq:
        supports = get_support(coefs)
        coefs[supports] = np.linalg.lstsq(X_pre[:, supports], y_pre, rcond=None)[0][:, 0]
    return coefs

def abess_all_solutions(X_pre, y_pre, sparsity, p=1, normalize_order=2, normalize_axis=None, center_y=False, refine=False):
    XX = X_pre
    yy = y_pre.flatten()

    if center_y:
        yy = yy - yy.mean()

    if normalize_axis is not None:
        X_norm = np.linalg.norm(X_pre, ord=normalize_order, axis=normalize_axis)
        XX = np.divide(XX, X_norm)

    objective_function = lambda coefs: jnp.linalg.norm(yy-XX@coefs)**p
    all_coefs = []
    all_supports = []
    for sp in range(1, sparsity+1):
        coefs = abess_linear(support_size=sp).fit(XX, yy).coef_
        if normalize_axis is not None:
            coefs = np.divide(coefs, X_norm)
        all_coefs.append(coefs)
        all_supports.append(get_support(coefs))
    all_coefs = np.array(all_coefs)

    if refine:
        if refine == 'original': all_supports = refine_solvel0(all_supports, (XX, yy), 'bic', False)
        else: all_supports = refine_solvel0(all_supports, (X_pre, y_pre), 'bic', False)
        all_supports = sorted([list(all_supports.track[e][0]) for e in all_supports.track], key=len)
        for i, coefs in enumerate(all_coefs):
            tmp_coef = np.zeros_like(coefs)
            tmp_coef[all_supports[i]] = np.linalg.lstsq(X_pre[:, all_supports[i]], y_pre, rcond=None)[0][:, 0]
            all_coefs[i] = tmp_coef

    return all_coefs, all_supports

