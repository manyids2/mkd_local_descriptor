# Multiple Kernel Local Descriptor

Implementation of [Multiple Kernel Local Patch Descriptor](https://arxiv.org/abs/1707.07825) using PyTorch.
Includes whitening models learned on PhotoTourism dataset, supervised and unsupervised versions.

## Usage

```python
import mkd_local_descriptor as mm

patch_size = 64

# will automatically take model_file trained on liberty, for proper patch_size, else closest.
# TODO: for now only for 'concat'

mkd = mm.MKD(dtype='concat',
             patch_size=patch_size,
             whitening=None,
             training_set='liberty',
             reduce_dims=128,
             do_l2=True,
             do_final_l2=True,
             do_gmask=True,
             device='cpu')

patches = torch.rand(12, 1, patch_size, patch_size)
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
