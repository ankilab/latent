# A single latent channel is sufficient for biomedical image segmentation

This repository contains related code to the paper below.
All relevant data, such as the pre-trained deep neural network, is available on [zenodo](https://zenodo.org/record/5772799).

## How to use the code

To use the code, you need a Python installation together with relevant libraries (imageio, numpy, scipy, tensorflow).
In general, no GPU is needed, but especially for training and large scale data inference recommended.

### Training deep neural network

We provide code in `training` to train a U-Net-like architecture with a reduced latent space.
The default configuration has a single latent space channel, i.e. the latent space image ![eq](https://latex.codecogs.com/gif.latex?\Psi_1).

For proper usage, you need the [BAGLS dataset](http://www.bagls.org).

### Generate latent space images

We provide a pre-trained model for retrieving the latent space images at zenodo.
In `latent_generation`, you will find the respective Jupyter notebook.

### Visualize the latent space

In `visualize`, you find a Jupyter notebook that uses a pre-trained model and its decoder,
as well as endoscopic images to show the respective latent space 
and options to investigate the latent space.


## How to cite this code

Kist et al. "A single latent channel is sufficient for biomedical image segmentation", biorxiv 2021

