MNIST Generative Adversarial Networks (PyTorch)
=======
Sam Greydanus. April 2017. MIT License.

About
--------
I use the classic MNIST dataset to achieve ultra-simple GAN results. Think of this repo as a lab where you can get comfortable with GANs before trying them on something more complex (e.g. CIFAR, ImageNet).

* [vanilla-gan](https://nbviewer.jupyter.org/github/greydanus/mnist-gan/blob/master/vanilla-gan.ipynb): **100 lines of code**
* [cnn-gan](https://nbviewer.jupyter.org/github/greydanus/mnist-gan/blob/master/cnn-gan.ipynb): **130 lines of code**

Samples
--------
Vanilla discriminator (D) and generator (G) networks

![vanilla-gan](static/vanilla-gan.png?raw=true)

CNN discriminator (D) and vanilla generator (G) network

![cnn-gan](static/cnn-gan.png?raw=true)


Dependencies
--------
* All code is written in python 3.6. You will need:
 * Numpy
 * matplotlib
 * [PyTorch](http://pytorch.org/): much easier to write and debug than TensorFlow!
 * [Jupyter](https://jupyter.org/)
