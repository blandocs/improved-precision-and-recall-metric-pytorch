## Improved Precision and Recall Metric for Assessing Generative Models &mdash; Pytorch Implementation 
![Python 3.7.3](https://img.shields.io/badge/python-3.7.3-green.svg?style=plastic)
![Pytorch 1.0.0](https://img.shields.io/badge/tensorflow-1.0.0-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

This repository is for personal practice.

> **Improved Precision and Recall Metric for Assessing Generative Models**<br>
> Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila<br>
> [Paper (arXiv)](https://arxiv.org/abs/1904.06991)
>
> **Abstract:** *The ability to evaluate the performance of a computational model is a vital requirement for driving algorithm research. This is often particularly difficult for generative models such as generative adversarial networks (GAN) that model a data manifold only specified indirectly by a finite set of training examples. In the common case of image data, the samples live in a high-dimensional embedding space with little structure to help assessing either the overall quality of samples or the coverage of the underlying manifold. We present an evaluation metric with the ability to separately and reliably measure both of these aspects in image generation tasks by forming explicit non-parametric representations of the manifolds of real and generated data. We demonstrate the effectiveness of our metric in StyleGAN and BigGAN by providing several illustrative examples where existing metrics yield uninformative or contradictory results. Furthermore, we analyze multiple design variants of StyleGAN to better understand the relationships between the model architecture, training methods, and the properties of the resulting sample distribution. In the process, we identify new variants that improve the state-of-the-art. We also perform the first principled analysis of truncation methods and identify an improved method. Finally, we extend our metric to estimate the perceptual quality of individual samples, and use this to study latent space interpolations.*

## Usage

This repository provides code for reproducing StyleGAN truncation sweep and realism score experiments. This code was tested with Python 3.7.3, Pytorch 1.0.0 and NVIDIA 1660 GPU.

To run the below code examples, you need to obtain the FFHQ dataset ([images1024x1024](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)). You can download it from [Flickr-Faces-HQ repository](http://stylegan.xyz/ffhq). To generate truncated images, you can use [pretrained_example.py](https://github.com/blandocs/improved-precision-and-recall-metric-pytorch/blob/master/pretrained_example.py) slightly modified version of [styleGAN pretrained_example.py](https://github.com/NVlabs/stylegan/blob/master/pretrained_example.py). Note that pretrained_example.py should executed in TF environment. Please refer [styleGAN respository](https://github.com/NVlabs/stylegan).

### Precision and Recall with truncated data

Precision and Recall of StyleGAN truncation sweep can be evaluated with:

```
python main.py --cal_type precision_and_recall --generated_dir truncation_0_7
```
Reference output is:

```
Precision: 0.671875
Recall: 0.4375
```

### Realism score

Evaluation of realism score using StyleGAN and FFHQ dataset can be run with:

```
python main.py --cal_type realism --generated_dir realism_test
```

Reference output is:

```
realism_data/high_realism_1.237395_53.png realism score: 1.0880612134933472
realism_data/low_realism_0.378355_19.png realism score: 0.7332894206047058
```
