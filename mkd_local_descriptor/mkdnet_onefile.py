# pylint: disable=C0302
import os
import numpy as np
from itertools import repeat
from scipy.special import i0, iv
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


pi = np.pi
ex = np.expand_dims
sqz = np.squeeze
zero_ = np.array([0], dtype=np.float32)

# ------------- Kernel stuff ------------------


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

# -------------- whitening numpy stuff -------------------------


def replace_nan(data, a=0):
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = a
    return data


def do_powerlaw(data, powerlaw=1.0):
    data = np.sign(data) * np.power(np.abs(data), powerlaw)
    return data


def vecpostproc(data, powerlaw=1.0):
    data = do_powerlaw(data, powerlaw)
    data = normalize(data, norm='l2')
    data = replace_nan(data)
    return data


def xform_data(data,
               model,
               *,
               xform='pca',
               reduce_dims=128,
               l2normalize=True,
               keval=40,
               t=0.7):
    mn = model["mean"]
    evecs = model["eigvecs"]
    evals = model["eigvals"]

    data = data - mn

    evecs = evecs[:, :min(reduce_dims, data.shape[1])]
    evals = evals[:min(reduce_dims, data.shape[1])]

    pval = 1.0
    if xform == 'pca':
        data = data @ evecs
        pval = 0.5
    elif xform == 'whiten':
        evecs = evecs @ np.diag(np.power(evals, -0.5))
        data = data @ evecs
    elif xform == 'lw':
        data = data @ evecs
    elif xform == 'pcaws':
        alpha = evals[keval]
        evals = ((1 - alpha) * evals) + alpha
        evecs = evecs @ np.diag(np.power(evals, -0.5))
        data = data @ evecs
    elif xform == 'pcawt':
        m = -0.5 * t
        evecs = evecs @ np.diag(np.power(evals, m))
        data = data @ evecs
    else:
        raise KeyError('Unknown transform - %s' % xform)

    if l2normalize:
        data = vecpostproc(data, pval)

    return data

# ------------- Layers stuff ------------------


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=0.6)


def input_norm(x, eps=1e-8):
    flat = x.contiguous().view(x.size(0), -1)
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + eps
    x = (x - mp.view(-1, 1, 1, 1)) / sp.view(-1, 1, 1, 1)
    return x


class L2Norm(nn.Module):
    def __init__(self, *, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):  # pylint: disable=W0221
        norm = torch.sqrt(torch.sum(x * x, dim=-1) + self.eps)
        x = x / norm.unsqueeze(-1)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):  # pylint: disable=W0221
        return x.view(self.shape)

    def extra_repr(self):
        return '{}'.format(self.shape)


class ReluOpt(nn.Module):
    def __init__(self, *, do_relu=True):
        super().__init__()
        self.do_relu = do_relu
        self.relu = nn.ReLU()

    def forward(self, x):  # pylint: disable=W0221
        return self.relu(x) if self.do_relu else x

    def extra_repr(self):
        return 'do_relu={}'.format(self.do_relu)


class Conv_BN_Relu(nn.Module):
    def __init__(self,
                 idims,
                 odims,
                 *,
                 kernel_size=3,
                 padding=0,
                 stride=1,
                 bias=False,
                 affine=False,
                 do_relu=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(idims,
                      odims,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=bias),
            nn.BatchNorm2d(odims, affine=affine),
            ReluOpt(do_relu=do_relu),
        )

    def forward(self, x):  # pylint: disable=W0221
        return self.features(x)


class WXform_base(nn.Module):
    def __init__(self, idims, odims, *, with_bias=False, device='cuda'):
        super().__init__()
        self.idims = idims
        self.odims = odims
        self.with_bias = with_bias

        self.weight = Parameter(
            torch.Tensor(np.ones([idims, odims], dtype=np.float32)).to(device),
            requires_grad=True)
        nn.init.orthogonal_(self.weight.data, gain=0.6)

        if self.with_bias:
            self.bias = Parameter(
                torch.Tensor(np.ones([idims], dtype=np.float32)).to(device), requires_grad=True)
            nn.init.constant_(self.bias.data, 0.00)

    def forward(self, x):  # pylint: disable=W0221
        if self.with_bias:
            output = x - self.bias
        else:
            output = x
        output = output @ self.weight

        return output

    def extra_repr(self):
        return '{idims},{odims}'.format(**self.__dict__)


class Gradients(nn.Module):
    def __init__(self, *, patch_size=32, do_smoothing=True, sigma=1.4, device='cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.do_smoothing = do_smoothing
        self.sigma = sigma
        self.eps = 1e-8

        delta = np.array([1, 0, -1])
        xf = delta.reshape([1, 3])
        yf = delta.reshape([3, 1])
        xf, yf = [
            f_[np.newaxis, np.newaxis, :, :].astype(np.float32) for f_ in [xf, yf]
        ]
        self.xf = torch.Tensor(xf).to(device)
        self.yf = torch.Tensor(yf).to(device)
        self.gp = nn.ReplicationPad2d((2, 2, 2, 2))
        self.xp = nn.ReplicationPad2d((1, 1, 0, 0))
        self.yp = nn.ReplicationPad2d((0, 0, 1, 1))
        self.bias = torch.zeros([1], dtype=torch.float32).to(device)

        if do_smoothing:
            gaussian = load_fspecial_gaussian_filter(sigma)
            gaussian = gaussian[np.newaxis, np.newaxis, :, :].astype(np.float32)
            self.gaussian = torch.Tensor(gaussian).to(device)

    def forward(self, x):  # pylint: disable=W0221

        if self.do_smoothing:
            x = self.gp(x)
            x = F.conv2d(x, self.gaussian, self.bias, 1, 0)

        inx = self.xp(x)
        gx = F.conv2d(inx, self.xf, self.bias, 1, 0)

        iny = self.yp(x)
        gy = F.conv2d(iny, self.yf, self.bias, 1, 0)

        mags = torch.sqrt(torch.pow(gx, 2) + torch.pow(gy, 2) + self.eps)
        oris = torch.atan2(gy, gx)

        y = torch.cat([mags, oris], dim=1)

        return y

    def extra_repr(self):
        return 'patch_size:{}, pad=ReplicationPad2d'.format(self.patch_size)


class GaussianSmoothing(nn.Module):
    def __init__(self, *, sigma=1.4, device='cpu'):
        super().__init__()

        self.sigma = sigma

        gaussian = load_fspecial_gaussian_filter(sigma)
        gaussian = gaussian[np.newaxis, np.newaxis, :, :].astype(np.float32)
        self.gaussian = torch.Tensor(gaussian).to(device)

        self.gp = nn.ReplicationPad2d((2, 2, 2, 2))

    def forward(self, x):  # pylint: disable=W0221

        x = self.gp(x)
        output = F.conv2d(x, self.gaussian, None, 1, 0)

        return output


class EmbedGradients(nn.Module):
    def __init__(self,
                 patch_size=32,
                 kappa=8,
                 n=3,
                 device='cuda',
                 relative=False,
                 weigh_mags=True):
        super().__init__()
        self.patch_size = patch_size
        self.relative = relative
        self.weigh_mags = weigh_mags
        self.eps = 1e-8
        self.in_shape = [-1, 2, patch_size, patch_size]

        kernel = Kernel(name='theta', kappa=kappa, n=n)
        weights = torch.Tensor(kernel.weights.reshape(-1, 1, 1).astype(np.float32)).to(device)  # pylint: disable=E1121
        frange = torch.Tensor(kernel.frange.reshape(-1, 1, 1).astype(np.float32)).to(device)

        self.emb0 = torch.ones([1, 1, patch_size, patch_size]).to(device)
        self.frange = frange
        self.weights = weights

        if relative:
            grid = get_grid(patch_size)
            self.phi = torch.Tensor(grid['phi']).to(device)

        self.odims = kernel.d

    def emb_ori(self, oris):
        if self.relative:
            oris = oris - self.phi
        emb0 = self.emb0.repeat(oris.size(0), 1, 1, 1)
        frange = self.frange * oris
        emb1 = torch.cos(frange)
        emb2 = torch.sin(frange)
        embedding = torch.cat([emb0, emb1, emb2], dim=1)
        embedding = self.weights * embedding
        return embedding

    def emb_mags(self, mags):
        mags = torch.sqrt(mags + self.eps)
        return mags

    def forward(self, grads):  # pylint: disable=W0221
        mags = grads[:, :1, :, :]
        oris = grads[:, 1:, :, :]

        if self.weigh_mags:
            y = self.emb_ori(oris) * self.emb_mags(mags)
        else:
            y = self.emb_ori(oris)

        return y


# TODO: for now, just copied the original grid.
def get_rho_phi(coords, fmap_size):
    # Only for log and linear cases.
    if coords == 'linear':
        rho = np.arange(2 * fmap_size) / (2 * fmap_size - 1)
        rho = rho[1::2]
    if coords == 'log':
        rho_range = np.arange(fmap_size) / (fmap_size / 2) - ((fmap_size - 1) / fmap_size)
        normGrid = (rho_range + 1) / 2
        rho = np.exp(normGrid * np.log(24.0)) - 1

    phi = (((np.arange(fmap_size) / fmap_size) * np.pi * 2) + np.pi / 8) % (2 * np.pi)

    return rho, phi


class ExplicitSpacialEncoding(nn.Module):
    def __init__(self,
                 dtype='cart',
                 fmap_size=8,
                 in_dims=128,
                 do_gmask=True,
                 do_l2=True,
                 kappa=1,
                 n=1,
                 coords='naive'):
        super().__init__()

        self.dtype = dtype
        self.fmap_size = fmap_size
        self.in_dims = in_dims
        self.do_gmask = do_gmask
        self.do_l2 = do_l2
        self.kappa = kappa
        self.n = n
        self.coords = coords
        self.grid = None
        self.gmask = None
        self.factors = {"phi": 1.0, "rho": np.pi, "x": np.pi / 2, "y": np.pi / 2}

        if self.dtype == 'cart':
            self.parametrization = ['x', 'y']
            emb, gmask = self.precompute_cart()
        elif self.dtype == 'polar':
            self.parametrization = ['rho', 'phi']
            emb, gmask = self.precompute_polar(coords)
        else:
            raise NotImplementedError(f'{self.dtype} is not implemented.')

        if self.do_gmask:
            self.gmask = gmask
            emb = emb * gmask

        self.emb = torch.from_numpy(emb)
        self.d_emb = self.emb.shape[1]
        self.out_dims = self.in_dims * self.d_emb
        self.odims = self.out_dims

        self.emb2, self.idx1 = self.init_kron()
        self.register_buffer('embedding', self.emb2)
        self.register_buffer('kron_ordering', self.idx1)

        self.norm = L2Norm()

    def extra_repr(self):
        return f'dtype:{self.dtype}, in_dims:{self.in_dims}, do_l2:{self.do_l2}, kappa:{self.kappa}, n:{self.n}'

    def precompute_cart(self):
        # x,y grids, adjust for VM kernel.
        grids = get_grid(self.fmap_size)
        grids = {k:v * self.factors[k] for k,v in grids.items()}
        grid = np.array([grids['x'], grids['y']])
        self.grid = grid
        rho = np.sqrt((grid**2).sum(axis=0))
        gmask = gaussian_mask(rho / rho.max(), sigma=1, correct_rho=False)

        # x,y kernels.
        kernels = {k:Kernel(k, kappa=self.kappa, n=self.n) for k in self.parametrization}
        # x,y embeddings.
        embeddings = {k: kernels[k].get_embedding(grid[i]) for i, k in enumerate(self.parametrization)}
        # joint embeddings.
        cart_emb = get_kron_kernel(embeddings['x'], embeddings['y'], axis=-1)
        cart_emb = np.expand_dims(cart_emb, 0)
        cart_emb = cart_emb.transpose(0,3,1,2)
        return cart_emb, gmask

    def precompute_polar(self, coords):
        if coords in ['log', 'linear']:
            # rho,phi grids, adjust for VM kernel.
            rho, phi = get_rho_phi(coords, self.fmap_size)
            # TODO: Correct properly for rho
            # rho = self.factors['rho'] * rho
            grid = np.array(np.meshgrid(rho, phi))
            sigma = 1.0
            if coords == 'log':
                sigma = 10.0
            gmask = gaussian_mask(grid[0], sigma=sigma, correct_rho=True)
        elif coords == 'naive':
            grids = get_grid(self.fmap_size)
            grids = {k:v * self.factors[k] for k,v in grids.items()}
            grid_xy = np.array([grids['x'], grids['y']])
            rho = np.sqrt((grid_xy**2).sum(axis=0))
            gmask = gaussian_mask(rho / rho.max(), sigma=1, correct_rho=False)
            grid = np.array([grids['rho'], grids['phi']])

        self.grid = grid

        # rho,phi kernels.
        self.parametrization = ['rho', 'phi']
        kernels = {k:Kernel(k, kappa=self.kappa, n=self.n) for k in self.parametrization}
        # rho,phi embeddings.
        embeddings = {k: kernels[k].get_embedding(grid[i]) for i, k in enumerate(self.parametrization)}
        # joint embeddings.
        polar_emb = get_kron_kernel(embeddings['phi'], embeddings['rho'], axis=-1)
        polar_emb = np.expand_dims(polar_emb, 0)
        polar_emb = polar_emb.transpose(0,3,1,2)
        return polar_emb, gmask

    def init_kron(self):
        kron = get_kron_order(self.in_dims, self.d_emb)
        idx1 = torch.Tensor(kron[:, 0]).type(torch.int64)
        idx2 = torch.Tensor(kron[:, 1]).type(torch.int64)
        emb2 = torch.index_select(self.emb, 1, idx2)
        return emb2, idx1

    # function to forward-propagate inputs through the network
    def forward(self, fcn_activations):  # pylint: disable=W0221
        emb1 = torch.index_select(fcn_activations, 1, self.idx1.to(fcn_activations.device))
        output = emb1 * (self.emb2).to(emb1.device)
        output = output.sum(dim=(2, 3))
        if self.do_l2:
            output = self.norm(output)
        return output


class BMVC(nn.Module):
    def __init__(self, dtype='concat',
                 patch_size=32,
                 whitening=None,
                 model_file=None,
                 training_set=None,
                 reduce_dims=128,
                 do_l2=True,
                 do_final_l2=True,
                 do_gmask=True,
                 device='cuda'):
        super().__init__()

        self.patch_size = patch_size
        self.whitening = whitening
        self.reduce_dims = reduce_dims
        self.training_set = training_set
        self.do_l2 = do_l2
        self.do_final_l2 = do_final_l2
        self.do_gmask = do_gmask
        self.device = device
        self.in_shape = [-1, 1, patch_size, patch_size]
        self.dtype = dtype

        self.model_file = model_file
        if self.model_file is None:
            this_dir, _ = os.path.split(__file__)
            data_path = os.path.join(this_dir, 'mkd-concat-pt.pkl')
            self.model_file = data_path

        self.grads = Gradients(patch_size=patch_size,
                               do_smoothing=True,
                               sigma=1.4 * (patch_size / 64.0),
                               device=device)

        if self.whitening is not None:
            # TODO: save also as torch tensors.
            whitening_models = np.load(self.model_file, allow_pickle=True)
            algo = 'lw' if self.whitening == 'lw' else 'pca'
            whitening_model = whitening_models[self.training_set][algo]
            self.whitening_model = {k:torch.from_numpy(v.astype(np.float32)) for k,v in whitening_model.items()}
            self.whitening_model = {k:v.to(device) for k,v in self.whitening_model.items()}

        if dtype in ['cart', 'concat']:
            ori_abs = EmbedGradients(patch_size=patch_size,
                                     weigh_mags=True,
                                     relative=False,
                                     device=device)
            cart_emb = ExplicitSpacialEncoding(dtype='cart',
                                               fmap_size=self.patch_size,
                                               in_dims=7,
                                               kappa=1,
                                               n=1,
                                               do_gmask=self.do_gmask,
                                               do_l2=self.do_l2)
            self.cart_feats = nn.Sequential(ori_abs, cart_emb)

        if dtype in ['polar', 'concat']:
            ori_rel = EmbedGradients(patch_size=patch_size,
                                     weigh_mags=True,
                                     relative=True,
                                     device=device)
            polar_emb = ExplicitSpacialEncoding(dtype='polar',
                                               fmap_size=self.patch_size,
                                               in_dims=7,
                                               kappa=8,
                                               n=2,
                                               do_gmask=self.do_gmask,
                                               do_l2=self.do_l2)
            self.polar_feats = nn.Sequential(ori_rel, polar_emb)

        self.norm = L2Norm()
        if dtype == 'concat':
            self.odims = polar_emb.odims + cart_emb.odims
        elif dtype == 'cart':
            self.odims = cart_emb.odims
        elif dtype == 'polar':
            self.odims = polar_emb.odims

        self.out_dim = self.odims
        if self.whitening is not None:
            self.out_dim = self.reduce_dims
            self.odims = self.reduce_dims

    def forward(self, patches):  # pylint: disable=W0221
        g = self.grads(patches)

        if self.dtype in ['polar', 'concat']:
            pe = self.polar_feats(g)
        if self.dtype in ['cart', 'concat']:
            ce = self.cart_feats(g)

        if self.dtype == 'concat':
            y = torch.cat([pe, ce], dim=1)
        elif self.dtype == 'cart':
            y = ce
        elif self.dtype == 'polar':
            y = pe
        if self.do_final_l2:
            y = self.norm(y)

        if self.whitening is not None:
            y = torch_xform_data(y, self.whitening_model, xform=self.whitening, reduce_dims=self.reduce_dims)
        return y

    def extra_repr(self):
        return 'idims:{}, do_l2:{}\npolar:{}, cart:{}\nuse_st:{}, do_log:{},\n polar_gmask:{}, cart_gmask:{}'.format(
            self.idims, self.do_l2, self.polar, self.cart, self.use_st, self.do_log,
            self.polar_gmask, self.cart_gmask)


def torch_xform_data(data, wh_model, xform, keval=40, t=0.7, reduce_dims=128):

    data_dim = data.shape[1]
    mn = wh_model["mean"]
    evecs = wh_model["eigvecs"]
    evals = wh_model["eigvals"]

    data = data - mn
    evecs = evecs[:, :min(reduce_dims, data_dim)]
    evals = evals[:min(reduce_dims, data_dim)]

    pval = 1.0
    if xform == 'pca':
        data = data @ evecs
        pval = 0.5
    elif xform == 'whiten':
        evecs = evecs @ torch.diag(torch.pow(evals, -0.5))
        data = data @ evecs
    elif xform == 'lw':
        data = data @ evecs
    elif xform == 'pcaws':
        alpha = evals[keval]
        evals = ((1 - alpha) * evals) + alpha
        evecs = evecs @ torch.diag(torch.pow(evals, -0.5))
        data = data @ evecs
    elif xform == 'pcawt':
        m = -0.5 * t
        evecs = evecs @ torch.diag(torch.pow(evals, m))
        data = data @ evecs
    else:
        raise KeyError('Unknown transform - %s' % xform)

    # powerlaw.
    data = torch.sign(data) * torch.pow(torch.abs(data), pval)

    # l2norm
    data = L2Norm()(data)

    return data


class WhitenBMVC(nn.Module):
    def __init__(self, patch_size=32, device='cuda', do_l2=True):
        super().__init__()
        self.patch_size = patch_size
        self.do_l2 = do_l2
        self.in_shape = [-1, 1, patch_size, patch_size]
        self.odims = 128
        bmvc = BMVC(patch_size=patch_size, device=device, do_l2=do_l2)
        self.features = nn.Sequential(
            bmvc, WXform_base(idims=bmvc.odims, odims=128, with_bias=True, device=device),
            Reshape(-1, 128, 1, 1), nn.BatchNorm2d(128, affine=False),
            Reshape(-1, 128), L2Norm())
        self.features.apply(weights_init)

    def forward(self, x):  # pylint: disable=W0221
        y = self.features(x)
        return y


class WhitenDescs(nn.Module):
    def __init__(self, idims, odims, device='cuda'):
        super().__init__()
        self.in_shape = [-1, idims]
        self.device = device
        self.odims = odims
        self.idims = idims
        self.out_dim = odims
        self.features = nn.Sequential(
            WXform_base(idims=idims, odims=odims, with_bias=True, device=device),
            Reshape(-1, odims, 1, 1), nn.BatchNorm2d(odims, affine=False),
            Reshape(-1, odims), L2Norm())
        self.features.apply(weights_init)

    def forward(self, x):  # pylint: disable=W0221
        y = self.features(x)
        return y


def orig_FCN():
    # model processing patches of size [32 x 32] and giving description vectors of length 2**7
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32, affine=False),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32, affine=False),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64, affine=False),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64, affine=False),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128, affine=False),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128, affine=False),
        nn.ReLU(),
        nn.Dropout(0.1),
    )


def FCN():
    return nn.Sequential(
        Conv_BN_Relu(1, 32, kernel_size=3, padding=1),
        Conv_BN_Relu(32, 32, kernel_size=3, padding=1),
        Conv_BN_Relu(32, 64, kernel_size=3, padding=1, stride=2),
        Conv_BN_Relu(64, 64, kernel_size=3, padding=1),
        Conv_BN_Relu(64, 128, kernel_size=3, padding=1, stride=2),
        Conv_BN_Relu(128, 128, kernel_size=3, padding=1),
        nn.Dropout(p=0.1),
    )


def FC():
    return nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
    )


def Whitening(mid_dims, out_dims):
    return nn.Sequential(
        WXform_base(idims=mid_dims, odims=out_dims, with_bias=True),
        Reshape(-1, out_dims, 1, 1), nn.BatchNorm2d(out_dims, affine=False),
        Reshape(-1, out_dims))


class HardNet(nn.Module):
    def __init__(self,
                 arch):
        super().__init__()

        self.arch = arch
        self.l2norm = L2Norm()

        # FCN + FC.
        if self.arch == 'orig':
            self.fcn = orig_FCN()
            self.fc = FC()
            self.features = nn.Sequential(self.fcn, self.fc)
        elif self.arch == 'hardnet':
            self.fcn = FCN()
            self.fc = FC()
            self.features = nn.Sequential(self.fcn, self.fc)
        elif self.arch == 'cart':
            self.fcn = FCN()
            self.encoding = ExplicitSpacialEncoding(dtype='cart', fmap_size=8)
            self.fc = nn.Sequential(self.encoding, Whitening(self.encoding.out_dims, self.encoding.out_dims))
        elif self.arch == 'polar':
            self.fcn = FCN()
            self.encoding = ExplicitSpacialEncoding(dtype='polar', fmap_size=8)
            self.fc = nn.Sequential(self.encoding, Whitening(self.encoding.out_dims, self.encoding.out_dims))
        else:
            raise NotImplementedError(f'{self.arch} not implemented.')

        # Common architecture.
        self.features = nn.Sequential(self.fcn, self.fc)

        # initialize weights
        self.features.apply(weights_init)

    def forward(self, patches):  # pylint: disable=W0221
        x_features = self.features(input_norm(patches))
        x = x_features.view(x_features.size(0), -1)
        x = self.l2norm(x)
        return x, patches
