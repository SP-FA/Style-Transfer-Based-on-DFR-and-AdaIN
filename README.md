# Style Transfer Based on DFR and AdaIN

![https://img.shields.io/badge/python-3.8-green](https://img.shields.io/badge/Python-3.8-green)
![https://img.shields.io/badge/pytorch-1.11.0-yellow](https://img.shields.io/badge/Pytorch-1.11.0-yellow)
![https://img.shields.io/badge/CUDA-11.3-brightgreen](https://img.shields.io/badge/CUDA-11.3-brightgreen)

[zh-ch](./doc/zh-ch.md)

![./sources/cover.png](./sources/cover.png)

## Overview

This is an image style transformation model. This model is based on `AdaIN` [1], and a `DFR` layer [2] is added between `VGG Encoder` and `AdaIN` to achieve better style conversion.

The network structure is shown as follows:

![./sources/ourNet.png](./sources/ourNet.png)

`DFR` is a method to extract multiple different features from a style image by rotating features from multiple angles [2]. In this experiment, we average the different features obtained by rotation, so as to obtain a more comprehensive feature information of the style image.

The experiment process is shown as follows:

![./sources/DFR.png](./sources/DFR.png)

## Results

The following is a comparison of our results with other algorithms, where:
1. Ours(1) use 0°, 90°, 180°, 270°
2. Ours(2) use 45°, 135°, 225°, 315°
3. Ours(3) use 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

![./sources/compare.png](./sources/compare.png)

## Getting started

### Installation

Check the `requirements.txt`:

`pip install -r requirements.txt`

### Data

Prepare the style images and content images, the default data path is `./input/content` and `./input/style`

### Train

### Test

Run the following command to test:

```commandline
python test.py [--content content_path --style style_path]
```

Optional parameters are shown in the following table:

| Parameter    | Introduction                                                     |
|--------------|------------------------------------------------------------------|
| content      | Path to the content image                                        |
| style        | Style image path                                                 |
| encoder      | Path to the encoder. The default is `models/encoder.pth`         |
| decoder      | Path to the decoder. The default is `models/decoder.pth`         |
| output       | Path to output the resulting image                               |
| content_size | Specifies a size for the content image, default is 512           |
| style_size   | Specifies a size for the style image, default is 512             |
| save_ext     | The extension to use when saving the image                       |
| alpha        | The degree of control style conversion should be between 0 and 1 |

## Reference

1. [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)
2. [Deep Feature Rotation for Multimodal Image Style Transfer](https://arxiv.org/pdf/2202.04426.pdf)