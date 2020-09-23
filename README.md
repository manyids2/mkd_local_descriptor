# Multiple Kernel Local Descriptor

Implementation of [Multiple Kernel Local Patch Descriptor](https://arxiv.org/abs/1707.07825) using PyTorch.
Includes whitening models learned on PhotoTourism dataset, supervised and unsupervised versions.

## Installation
 - [release](https://github.com/manyids2/mkd_local_descriptor/archive/1.0.tar.gz)
 - `pip install https://github.com/manyids2/mkd_local_descriptor/archive/1.0.tar.gz`

## Usage

```python
import torch
import mkd_local_descriptor as mm

# will automatically take model_file trained on liberty, for proper patch_size, else closest.

mkd = mm.MKD(dtype='concat',             # 'concat', 'polar', 'cart'.
             patch_size=64,              # 64, 32 have learned models.
             whitening=None,             # None, 'lw', 'pca', 'pcaws', 'pcawt'.
             training_set='liberty',     # 'liberty', 'notredame', 'yosemite'
             reduce_dims=128,
             do_l2=True,
             do_final_l2=True,
             do_gmask=True,
             device='cpu')

patches = torch.rand(12, 1, 64, 64)
descs = mkd(patches)
print(f'descs: {descs.shape}')
```

## Bibliography
  Please cite :
```
  @article{mukundan2017multiple,
    title={Multiple-kernel local-patch descriptor},
    author={Mukundan, Arun and Tolias, Giorgos and Chum, Ond{\v{r}}ej},
    journal={arXiv preprint arXiv:1707.07825},
    pages={6},
    year={2017}
  }

  @article{mukundan2018understanding,
    title={Understanding and improving kernel local descriptors},
    author={Mukundan, Arun and Tolias, Giorgos and Bursuc, Andrei and J{\'e}gou, Herv{\'e} and Chum, Ond{\v{r}}ej},
    journal={International Journal of Computer Vision},
    pages={1},
    year={2018},
  }
```
