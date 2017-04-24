MNIST Generative Adversarial Networks (PyTorch)
=======
Sam Greydanus. April 2017. MIT License.

Samples
--------
Vanilla D and G networks (**100 lines of code**)

![vanilla-gan](static/vanilla-gan.png?raw=true)

CNN D and vanilla G networks (**130 lines of code**)

![cnn-gan](static/cnn-gan.png?raw=true)

Jupyter Notebooks
--------
* [vanilla-gan](https://nbviewer.jupyter.org/github/greydanus/mnist-gan/blob/master/vanilla-gan.ipynb)
* [cnn-gan](https://nbviewer.jupyter.org/github/greydanus/mnist-gan/blob/master/cnn-gan.ipynb)

About
--------
I use the classic MNIST dataset to achieve ultra-simple GAN results. Think of this repo as a lab where you can get comfortable with GANs before trying them on something more complex (e.g. CIFAR, ImageNet).

Dependencies
--------
* All code is written in python 3.6. You will need:
 * Numpy
 * matplotlib
 * [PyTorch](http://pytorch.org/): much easier to write and debug than TensorFlow!
 * [Jupyter](https://jupyter.org/)
