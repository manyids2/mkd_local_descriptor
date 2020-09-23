# pylint: disable=C0302
import os
import numpy as np
from itertools import repeat
from scipy.special import i0, iv
from sklearn.preprocessing import normalize

pi = np.pi
ex = np.expand_dims
sqz = np.squeeze
zero_ = np.array([0], dtype=np.float32)


def ret_float32(func):
    def wrapper(*args, **kwargs):  # pylint: disable=R1710
        rets = func(*args, **kwargs)
        if isinstance(rets, np.ndarray):  # pylint: disable=R1705
            return rets.astype(np.float32)
        elif isinstance(rets, list):
            return map(np.array, rets, repeat(np.float32))

    return wrapper


def cart2pol(x, y):
    phi = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return phi, rho


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def get_grid(patch_size):
    x, y = [np.arange(-1 * (patch_size - 1), patch_size, 2, dtype=np.float32)] * 2
    xx, yy = np.meshgrid(x, y)
    phi, rho = cart2pol(xx, yy)
    rho = rho / np.sqrt(2 * np.power((patch_size - 1), 2))
    xx, yy = [item / (patch_size - 1) for item in [xx, yy]]
    grid = {'x':xx, 'y':yy, 'rho':rho, 'phi':phi}
    return grid


def embcoef(kappa, n):
    C = 0.5 * (iv(0, kappa) - np.exp(-1 * kappa)) / np.sinh(kappa)
    an = iv(range(1, n + 1), kappa) / np.sinh(kappa)
    coef = np.insert(an, 0, C, axis=0)
    return coef


class Kernel:
    def __init__(self, name, *, kappa=8, n=2):
        self.name = name
        self.k = kappa
        self.n = n
        self.d = 2 * n + 1
        self.coeffs = self.get_vonmises_coeffs()
        self.frange = self.get_frange()
        self.dimensionality = self.d
        self.weights = self.get_weights()
        self.embedding = None

    @ret_float32
    def get_vonmises_coeffs(self):
        coeffs = embcoef(self.k, self.n)
        return coeffs

    @ret_float32
    def get_frange(self):
        frange = np.arange(self.n) + 1
        return frange

    @ret_float32
    def get_weights(self):
        n = self.n
        weights = np.zeros([2 * n + 1], dtype=np.float32)
        weights[:n + 1] = np.sqrt(self.coeffs)
        weights[n + 1:] = np.sqrt(self.coeffs[1:])
        return weights

    @ret_float32
    def get_embedding(self, patch):
        s = patch.shape
        patch = ex(patch, len(s))
        frange = self.frange * patch
        emb0 = np.ones((*s, 1), dtype=np.float32)
        emb1 = np.cos(frange)
        emb2 = np.sin(frange)
        embedding = np.concatenate([emb0, emb1, emb2], axis=-1)
        embedding = self.weights * embedding
        return embedding

    def set_embedding(self, patch_):
        self.embedding = self.get_embedding(patch_)

    def __repr__(self):
        outstr = 'name  : %s\n' % self.name
        outstr += 'kappa : %d\n' % self.k
        outstr += 'n     : %d\n' % self.n
        outstr += 'dim   : %d' % self.d
        return outstr


def get_kron_order(d1, d2):
    kron_order = np.zeros([d1 * d2, 2], dtype=np.int32)
    for i in range(d1):
        for j in range(d2):
            kron_order[i * d2 + j, :] = [i, j]
    return kron_order


@ret_float32
def get_kron_kernel(emb1, emb2, axis=-1):
    assert (axis < 0), 'Broadcasting gets fucked if axis is spec from left'
    s1 = emb1.shape
    s2 = emb2.shape
    kron_order = get_kron_order(s1[axis], s2[axis])
    kron_kernel = np.take(emb1, kron_order[:, 0], axis=axis) * np.take(
        emb2, kron_order[:, 1], axis=axis)
    return kron_kernel


@ret_float32
def gaussian_mask(rho, sigma=1, correct_rho=True):
    # make max(rho) = sqrt(2)
    if correct_rho:
        rho = rho * np.sqrt(2)
    gmask = np.exp(-1 * rho**2 / sigma**2)
    return gmask


@ret_float32
def load_fspecial_gaussian_filter(sigma):
    rx = np.arange(-2, 3, dtype=np.float32)
    fx = np.exp(-1 * np.square(rx / (sigma * np.sqrt(2.0))))
    fx = np.expand_dims(fx, 1)
    gx = np.dot(fx, fx.T)
    gx = gx / np.sum(gx)
    return gx

