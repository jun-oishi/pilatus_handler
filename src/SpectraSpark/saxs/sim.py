#! /usr/bin/env python3

from pymatgen.core import Structure
from numba import jit
import numpy as np

from .qi2d import _radial_average

@jit(nopython=True, cache=True)
def _oriented_saxs(coords, scat_factor, q_mesh):
    """
    原子配置に対する小角散乱振幅を計算する

    Parameters
    ----------
    coords : np.ndarray
        原子の座標[cartesian]
    scat_factor : np.ndarray[complex]
        原子の散乱因子
    q_mesh : np.ndarray
        散乱ベクトルの3次元配列: q[i, j] = (qx, qy, qz)

    Returns
    -------
    f_mesh : np.ndarray[complex]
        散乱振幅の2次元配列: f[i, j] = f(q[i, j])
    """
    f_mesh = np.zeros(q_mesh.shape[:2], dtype=np.complex128)
    for i in range(f_mesh.shape[0]):
        for j in range(f_mesh.shape[1]):
            q = q_mesh[i, j]
            for k in range(len(coords)):
                f_mesh[i, j] += scat_factor[k] * np.exp(1j * (q @ coords[k]))
    return f_mesh

@jit(nopython=True, cache=True)
def _periodicizer(q_mesh, unitcell, na, nb, nc):
    """
    単位胞の散乱振幅から結晶の散乱振幅を計算する

    Parameters
    ----------
    q_mesh : np.ndarray
        散乱ベクトルの3次元配列: q[i, j] = (qx, qy)
    unitcell : np.ndarray
        結晶の単位胞の格子ベクトル: unitcell = [a, b, c] = [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]
    na, nb, nc : int
        繰り返し数

    Returns
    -------
    f_cr : np.ndarray[complex]
        結晶の並進対称性による散乱振幅の2次元配列: f_cr[i, j] = f_cr(q[i, j])
    """
    thresh = 1e-10
    f_cr = np.empty(q_mesh.shape[:2], dtype=np.complex128)
    a, b, c = unitcell
    for i in range(f_cr.shape[0]):
        for j in range(f_cr.shape[1]):
            q = q_mesh[i, j]
            qa, qb, qc = q @ a, q @ b, q @ c
            ca = (np.exp(1j * na * qa) - 1) / (np.exp(1j * qa) - 1) if np.abs(qa) > thresh else na
            cb = (np.exp(1j * nb * qb) - 1) / (np.exp(1j * qb) - 1) if np.abs(qb) > thresh else nb
            cc = (np.exp(1j * nc * qc) - 1) / (np.exp(1j * qc) - 1) if np.abs(qc) > thresh else nc
            f_cr[i, j] = ca * cb * cc
    return f_cr

@jit(nopython=True, cache=True)
def _rotate_q(q_mesh, theta, phi):
    """
    散乱ベクトルを回転する
    z軸でphi回した後にy軸でtheta回す

    Parameters
    ----------
    q_mesh : np.ndarray
        散乱ベクトルの2次元配列: q[i, j] = (qx, qy)
    theta : float
        回転角[rad]
    phi : float
        回転角[rad]

    Returns
    -------
    q_rot : np.ndarray
        回転後の散乱ベクトルの3次元配列: q_rot[i, j] = (qx, qy, qz)
    """
    shape = (q_mesh.shape[0], q_mesh.shape[1], 3)
    q_rot = np.empty(shape, dtype=np.float64)
    for i in range(q_mesh.shape[0]):
        for j in range(q_mesh.shape[1]):
            qx, qy = q_mesh[i, j]
            q_rot[i, j, 0] = np.cos(theta) * (qx * np.cos(phi) - qy * np.sin(phi))
            q_rot[i, j, 1] = (qx * np.sin(phi) + qy * np.cos(phi))
            q_rot[i, j, 2] = -np.sin(theta) * (qx * np.cos(phi) - qy * np.sin(phi))
    return q_rot


def sim_saxs(structure:Structure, scat_factor:dict[str,complex], q_max, q_step,
              na=10, nb=-1, nc=-1, n_theta=60, n_phi=120):
    """
    方位平均をとった散乱強度を計算する

    Parameters
    ----------
    structure : pymatgen.Structure
        結晶構造
    scat_factor : dict[str,complex]
        原子種に対する散乱因子: scat_factor[元素記号] = 散乱因子
    q_max : float
        散乱ベクトルの最大値
    q_step : float
        散乱ベクトルの刻み幅
    na, nb, nc : int
        繰り返し数, nb, ncが指定されない場合はnaと同じ値
    n_theta, n_phi : int
        方位角の分割数

    Returns
    -------
    q : np.ndarray
        散乱ベクトルの大きさの配列
    i : np.ndarray
        1次元化された散乱強度の配列
    """
    coords = np.array([site.coords for site in structure.sites]) * 0.1 # to nm
    arr_f = np.array([scat_factor[str(site.specie)] for site in structure.sites])
    unitcell = structure.lattice.matrix * 0.1 # to nm
    if nb < 0:
        nb = na
    if nc < 0:
        nc = na

    n_q = int(q_max / q_step)
    qxx, qyy = np.meshgrid(np.linspace(-q_max, q_max, n_q),
                           np.linspace(0, q_max, n_q//2))
    q_mesh = np.stack([qxx, qyy], axis=-1)
    i_mesh = np.zeros(q_mesh.shape[:2])
    for theta in np.linspace(0, np.pi, n_theta, endpoint=False):
        for phi in np.linspace(0, 2*np.pi, n_phi, endpoint=False):
            q_rot = _rotate_q(q_mesh, theta, phi)
            _f = _oriented_saxs(coords, arr_f, q_rot) * _periodicizer(q_rot, unitcell, na, nb, nc)
            i_mesh += np.abs(_f)**2
    i_mesh /= n_theta * n_phi

    r, i = _radial_average(i_mesh, n_q/2-0.5, n_q/2-0.5)
    q = r * (2*q_max / n_q)
    return q, i
