import os
import numpy as np
import torch
import torch.nn as nn

from .layers import Gradients, EmbedGradients, ExplicitSpacialEncoding, L2Norm


class MKD(nn.Module):
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
            data_path = os.path.join(this_dir, f'mkd-{dtype}-{patch_size}.pkl')
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

