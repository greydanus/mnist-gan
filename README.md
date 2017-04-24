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
For an easy intro to the code (along with equations and explanations) check out these Jupyter notebooks:
* [vanilla-gan](https://nbviewer.jupyter.org/github/greydanus/mnist-gan/blob/master/vanilla-gan.ipynb)
* [cnn-gan](https://nbviewer.jupyter.org/github/greydanus/mnist-gan/blob/master/cnn-gan.ipynb)

Getting started
--------
* check dependencies (see below)
* run the jupyter notebooks


About
--------
I use the classic MNIST dataset to achieve ultra-simple GAN results. GANs are extremely promising architectures but they present some unique training challenges. Think of this repo as a laboratory where you can get comfortable with them before trying them on something more complex (e.g. CIFAR, ImageNet).

Dependencies
--------
* All code is written in python 3.6. You will need:
 * Numpy
 * matplotlib
 * [PyTorch](http://pytorch.org/): much easier to write and debug than TensorFlow. My new favorite framework!
 * [Jupyter](https://jupyter.org/)
