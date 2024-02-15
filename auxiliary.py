import scikit_tt as scikit
from typing import List
from scikit_tt.data_driven.transform import Function
import numpy as np


def dudx(u, h):
    # u.shape = (Nt, Nx)
    Nx = u.shape[1]
    if Nx<=5:
        raise ValueError('Must have at least 6 spatial points.')
    else:
        ux = np.zeros_like(u)
        ux[:,2:Nx-2] = (u[:,0:Nx-4] - 8*u[:,1:Nx-3] + 8*u[:,3:Nx-1] - u[:,4:Nx])/12/h
        for i in [0, 1]:
            ux[:,i] = (-25*u[:,i] + 48*u[:,i+1] - 36*u[:,i+2] + 16*u[:,i+3] - 3*u[:,i+4])/12/h
        for i in [Nx-2, Nx-1]:
            ux[:,i] = (-25*u[:,i] + 48*u[:,i-1] - 36*u[:,i-2] + 16*u[:,i-3] - 3*u[:,i-4])/12/h
        return ux


def dudt(u, k):
    Nt = u.shape[0]
    ut = np.zeros_like(u)
    ut[1:Nt-1,:] = (u[2:Nt,:]-u[0:Nt-2,:])/(2*k)
    ut[0,:] = (-3.0/2*u[0,:] + 2*u[1,:]-u[2,:]/2)/k
    ut[Nt-1,:] = (3.0/2*u[Nt-1,:] - 2*u[Nt-2,:] + u[Nt-3,:]/2)/k
    return ut


def coordinate_major(x: np.ndarray, phi: List[Function]) -> 'TT':
    m = x.shape[1] # number of snapshots
    p = len(phi) # number of modes
    d = x.shape[0] # number of dimensions

    # define cores as list of empty arrays
    cores = [np.zeros([1, p, 1, m])] + [np.zeros([m, p, 1, m]) for _ in range(1, d)]

    # insert elements of first core
    for j in range(m):
        cores[0][0, :, 0, j] = np.array([phi[k](x[0, j]) for k in range(p)])

    # insert elements of subsequent cores
    for i in range(1, d):
        for j in range(m):
            cores[i][j, :, 0, j] = np.array([phi[k](x[i, j]) for k in range(p)])

    # append core containing unit vectors
    cores.append(np.eye(m).reshape(m, m, 1, 1))

    # construct tensor train
    psi = scikit.TT(cores)
    return psi


def mandy_cm(x: np.ndarray, y: np.ndarray, phi: List[Function]=[lambda t: 1, lambda t: t], threshold: float=0.0):
    d = x.shape[0]
    m = x.shape[1]

    # construct transformed data tensor
    psi = coordinate_major(x, phi)

    # define xi as pseudoinverse of psi
    xi = psi.pinv(d, threshold=threshold, ortho_r=False)

    # multiply last core with y
    xi.cores[d] = (xi.cores[d].reshape([xi.ranks[d], m]).dot(y.transpose())).reshape(xi.ranks[d], 1, 1, 1)

    # set new row dimension
    xi.row_dims[d] = 1

    return xi


def print_time(xi_exact):
    start_time = scikit.progress('Construct exact solution in TT format', 0)
    utl.progress('Construct exact solution in TT format', 100, cpu_time=_time.time() - start_time)
    start_time = scikit.progress('Construct exact solution in matrix format', 0)
    xi_exact_mat = xi_exact.full().reshape([np.prod(xi_exact.full().shape), 1])
    utl.progress('Construct exact solution in matrix format', 100, cpu_time=_time.time() - start_time)
