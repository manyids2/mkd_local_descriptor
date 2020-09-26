import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mkd_local_descriptor as mm
import mkd_pytorch as mp


COEFFS_N1_K1 = [0.38214156, 0.48090413]
COEFFS_N2_K8 = [0.14343168, 0.268285, 0.21979234]
COEFFS_N3_K8 = [0.14343168, 0.268285, 0.21979234, 0.15838885]
COEFFS = {'xy':COEFFS_N1_K1, 'rhophi':COEFFS_N2_K8, 'theta':COEFFS_N3_K8}


def print_coeffs():
    Kernel = mm.kernel.Kernel
    params = [[2, 8], [3, 8], [1, 1]]
    for n, kappa in params:
        kernel = Kernel('x', kappa=kappa, n=n)
        print(f'COEFFS_N{n}_K{kappa} = {kernel.coeffs}')


if __name__ == "__main__":
    patch_size = 32
    whitening = 'pcawt'

    mkd = mm.MKD(dtype='concat',
                 patch_size=patch_size,
                 whitening=whitening,
                 do_gmask=False,
                 device='cpu')

    patches = torch.rand(12, 1, patch_size, patch_size)
    descs = mkd(patches)
    print(f'descs: {descs.shape}')

    mp_mkd = mp.MKD(dtype='concat',
                    patch_size=patch_size,
                    whitening=whitening,
                    do_gmask=False,
                    device='cpu')

    mp_descs = mp_mkd(patches)
    print(f'mp_descs: {mp_descs.shape}')

    desc_diff = ((descs.numpy() - mp_descs.numpy())**2).sum()
    print(f'desc_diff: {desc_diff}')

    print(mp_mkd)

