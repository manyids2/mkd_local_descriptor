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

## Performance

- Trained on 32x32

```
  ------------------------------------------------------------------------------
  Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
  ------------------------------------------------------------------------------
  trained on       liberty notredame  liberty yosemite  notredame yosemite
  tested  on           yosemite           notredame            liberty
  ------------------------------------------------------------------------------
  Kornia-RootSIFT-32     58.24              49.07               49.65
  MKD-concat-None-32 57.86  57.86        49.93  49.93        48.78  48.78
  MKD-concat-lw-32   72.68  72.44        61.34  59.52        61.28  59.87
  MKD-concat-pca-32  64.92  64.55        55.15  54.21        54.31  53.81
  MKD-concat-pcaws-32 66.96  66.61        56.59  55.28        55.62  54.92
  MKD-concat-pcawt-32 68.66  68.24        57.64  56.05        56.78  55.86
  ------------------------------------------------------------------------------
```

- Trained on 64x64

```
  ------------------------------------------------------------------------------
  Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
  ------------------------------------------------------------------------------
  trained on       liberty notredame  liberty yosemite  notredame yosemite
  tested  on           yosemite           notredame            liberty
  ------------------------------------------------------------------------------
  Kornia-RootSIFT-64     56.58              47.68               48.20
  MKD-concat-None-64 57.48  57.48        49.49  49.49        48.29  48.29
  MKD-concat-lw-64   73.36  73.15        61.90  59.95        61.94  60.35
  MKD-concat-pca-64  65.58  65.36        55.53  54.60        55.03  54.46
  MKD-concat-pcaws-64 67.74  67.16        57.14  55.80        56.13  55.63
  MKD-concat-pcawt-64 69.44  68.96        58.19  56.59        57.41  56.55
  ------------------------------------------------------------------------------
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
