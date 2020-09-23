import numpy as np
import torch
import torch.nn as nn

from .layers import weights_init, input_norm, \
    Conv_BN_Relu, Reshape, WXform_base, L2Norm, ExplicitSpacialEncoding


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

