import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .kernel import load_fspecial_gaussian_filter, \
    Kernel, \
    get_grid, \
    gaussian_mask, \
    get_kron_kernel, \
    get_kron_order


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

