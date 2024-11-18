#! /bin/env/python3

import numpy as np
from numba import jit
from scipy import optimize
import warnings

def normal_ave(mu, sigma, fun, n=128):
    """変数xが正規分布に従うときf(x)の平均値を求める
    """
    if sigma <= 1e-20:
        return fun(mu)
    x = np.linspace(mu-3*sigma, mu+3*sigma, n)
    y = fun(x)
    weight = np.exp(-0.5*((x-mu)/sigma)**2)
    return np.sum(y*weight)/np.sum(weight)

@jit(nopython=True, cache=True)
def phi(q, r):
    """球形散乱体の散乱振幅"""
    return 3 * (np.sin(q * r) - q * r * np.cos(q * r)) / (q * r)**3

def p_sph(q, r, std=0.0):
    """正規分布に従う粒径分布の球形散乱体の散乱振幅の自乗平均"""
    if std <= 0.0:
        return phi(q, r) ** 2
    i = [normal_ave(r, r*std, lambda _r: phi(_q, _r)**2) for _q in q]
    return np.array(i)

def psq_bar(q, r, std=0.0):
    """正規分布に従う粒径分布の球形散乱体の散乱振幅の平均の自乗"""
    if std <= 0.0:
        return phi(q, r) ** 2
    i = [normal_ave(r, r*std, lambda _r: phi(_q, _r)) for _q in q]
    return np.array(i) ** 2

def beta(q, r, std=0.0):
    """psq_bar / p_sph"""
    pq = p_sph(q, r, std)
    phi_sq_bar = psq_bar(q, r, std)
    return phi_sq_bar / pq

@jit(nopython=True, cache=True)
def S_PY(q, eta_hs, r_hs):
    """ structure factor by Percus-Yavick approximation
    Parameters
    ----------
    q : array of magnitude of scattering vector
    eta_hs : volume fraction of hard spheres
    r_hs : interaction radius of hard spheres
    """
    alpha = (1+2*eta_hs)**2 * (1-eta_hs)**-4
    beta = -6 * eta_hs * (1+eta_hs*0.5)**2 * (1-eta_hs)**-2
    gamma = eta_hs * alpha * 0.5
    A = r_hs * q
    sinA, cosA = np.sin(A), np.cos(A)
    g = alpha * (sinA - A*cosA) * A**-2 \
        + beta * (2*A*sinA + (2-A**2)*cosA - 2) * A**-3 \
        + gamma * (-A**4*cosA + 4*((3*A**2-6)*cosA + (A**3-6*A)*sinA + 6)) * A**-5
    return (1 + 24*eta_hs * g * A**-1) ** -1

def S_decouple(q, r, eta, c, std=0.0):
    if std <= 0.0:
        return S_PY(q, eta, r)
    r_hs = normal_ave(r, r*std, lambda _r: ((c*_r)**3)) ** (1.0/3)
    return 1 + beta(q, r, std) * (S_PY(q, eta, r_hs) - 1)

def decouple_fit(q, i, r0, d_rho0=1.0, eta0=1e-2, c0=1.5, std0=1e-1, q_range=(0.1, 8)):
    i = i[(q_range[0] <= q) & (q <= q_range[1])]
    q = q[(q_range[0] <= q) & (q <= q_range[1])]
    def _residual(params):
        d_rho, r, eta, c, std = params
        return i - d_rho**2 * p_sph(q, r, std) * S_decouple(q, r, eta, c, std)
    params0 = [d_rho0, r0, eta0, c0, std0]
    warnings.filterwarnings("ignore")
    lb = [0, 1e-10, 1e-3, 1.0, 0.0]
    ub = [1e10, 10, 1, 10, 0.5]
    res = optimize.least_squares(_residual, params0, bounds=(lb, ub))
    warnings.resetwarnings()
    if not res.success:
        warnings.warn(res.message)
    return res.x
