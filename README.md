# Multiple Kernel Local Descriptor

Implementation of [Understanding and Improving Kernel Local Descriptors](https://arxiv.org/abs/1811.11147) using PyTorch.
Includes whitening models learned on PhotoTourism dataset, supervised and unsupervised versions.

## Installation
`pip install https://github.com/manyids2/mkd_local_descriptor/archive/1.0.1.tar.gz`

## Usage

*NOTE* Uses the 'pcawt' version (attenuation) of unsupervised whitening as default as it generalizes better. However, the 'lw' performs
better on PhotoTourism benchmark.

```python
import torch
import mkd_local_descriptor as mm

# Use model_file trained on PhotoTourism for 64x64 patch_size.

mkd = mm.MKD(dtype='concat',             # 'concat', 'polar', 'cart'.
             patch_size=64,
             whitening='pcawt',             # None, 'lw', 'pca', 'pcaws', 'pcawt'.
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

Evaluated on the [brown_phototour_revisited benchmark](https://github.com/ducha-aiki/brown_phototour_revisited).

```
@misc{BrownRevisited2020,
  title={UBC PhotoTour Revisied},
  author={Mishkin, Dmytro},
  year={2020},
  url = {https://github.com/ducha-aiki/brown_phototour_revisited}
}
```


- patch_size = 64
```
  ------------------------------------------------------------------------------
  Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
  ------------------------------------------------------------------------------
  trained on       liberty notredame  liberty yosemite  notredame yosemite
  tested  on           yosemite           notredame            liberty
  ------------------------------------------------------------------------------
  Kornia-RootSIFT-64     56.58              47.68               48.20
  MKD-concat-None-64  57.48  57.48        49.49  49.49        48.29  48.29
  MKD-concat-lw-64    73.36  73.15        61.90  59.95        61.94  60.35
  MKD-concat-pca-64   65.58  65.36        55.53  54.60        55.03  54.46
  MKD-concat-pcaws-64 67.74  67.16        57.14  55.80        56.13  55.63
  MKD-concat-pcawt-64 69.44  68.96        58.19  56.59        57.41  56.55
  ------------------------------------------------------------------------------
```

- patch_size = 32

```
  ------------------------------------------------------------------------------
  Mean Average Precision wrt Lowe SNN ratio criterion on UBC Phototour Revisited
  ------------------------------------------------------------------------------
  trained on       liberty notredame  liberty yosemite  notredame yosemite
  tested  on           yosemite           notredame            liberty
  ------------------------------------------------------------------------------
  Kornia-RootSIFT-32        58.24              49.07               49.65
  MKD-concat-None-32    57.86  57.86        49.93  49.93        48.78  48.78
  MKD-concat-lw-32      72.29  71.98        60.90  58.81        60.71  59.13
  MKD-concat-pca-32     64.46  64.32        54.82  53.87        54.13  53.62
  MKD-concat-pcaws-32   66.59  66.05        56.43  55.21        55.22  54.78
  MKD-concat-pcawt-32   68.05  67.61        57.22  55.76        56.27  55.50
  ------------------------------------------------------------------------------
```

## Extraction times

Times for extraction of 1024 batches on Tesla P100 (16GB).

patch_size | batch_size | time(seconds)
:--------: | :--------: | :--:
    64     | 64         |  8.96
    64     | 128        | 10.50
    64     | 256        | 19.15
    64     | 512        | 36.98
    64     | 1024       | 72.81

patch_size | batch_size | time(seconds)
:--------: | :--------: | :--:
    32     | 64         |  5.93
    32     | 128        |  4.21
    32     | 256        |  5.67
    32     | 512        |  9.12
    32     | 1024       | 17.26

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
