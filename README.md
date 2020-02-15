# pytorch-vgg

Some scripts to convert the [VGG-16](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
and [VGG-19](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) models [1] from Caffe to PyTorch.

The converted models can be used with the PyTorch model zoo and are available here:

VGG-16: https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth

VGG-19: https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth

These models expect different preprocessing than the other models in the PyTorch model zoo.
Images should be in BGR format in the range [0, 255], and the following BGR values should then be
subtracted from each pixel: [103.939, 116.779, 123.68]

[1] Karen Simonyan and Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR 2015
