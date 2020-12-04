# Dataset

The ALFRED dataset contains 8k+ expert demostrations with 3 or more language annotations each.
A trajectory consists of a sequence of expert actions, the corresponding image observations, and language annotations describing segments of the trajectory.
For the details of the ALFRED dataset such as file structure, see the repository of <a href="https://github.com/askforalfred/alfred/tree/master/data">ALFRED</a>.

## Extracting Resnet Features
### Resnet Features without Color Swap
To extract Resnet features without color swap augmentation from raw image sequences:
```bash
$ python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 32 --gpu --visual_model resnet18 --filename feat_conv.pt
```
This will save `feat_conv.pt` files inside each trajectory root folder.

### Resnet Features with Color Swap
To extract Resnet features with color swap augmentation from raw image sequences:
```bash
$ python models/utils/extract_resnet_colorSwap.py --data data/full_2.1.0 --batch 32 --gpu --visual_model resnet18
```
This will save `feat_conv.pt`, `feat_conv_colorSwap1.pt`, and `feat_conv_colorSwap2.pt` files insides each trajectory root folder.
