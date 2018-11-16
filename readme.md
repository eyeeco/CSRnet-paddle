# CSRnet

## Introduction
The crowd density estimation method, which is applied to crowded scenes, has been widely used in airports, stations, operating vehicles, art galleries, etc. because of its accuracy and speed far above the naked eye count. It can effectively prevent hidden dangers such as crowded trampling and overloading, and on the other hand, it can help retailers to collect passenger traffic.
The training and test pictures used in this competition are all from the general monitoring scene, but include a variety of perspectives (such as low altitude, high altitude, fisheye, etc.), the relative size of the pedestrians in the map will also have a large difference. Some training data refer to public data sets (such as ShanghaiTech [1], UCF-CC-50 [2], WorldExpo'10 [3], Mall [4], etc.).

![1542369222453](img\1542369222453.png)

## Our Model
Population density estimates can be divided into three main categories: detection-based methods, regression-based methods, and density estimation-based methods.
The fundamental idea of the proposed design is to deploy a deeper CNN for capturing high-level features with larger receptive fields and generating high-quality densitymaps without brutally expanding network complexity.
Coupled with the analysis of the above problems, we propose such a model framework
* Data Partition
* Density Map
* Deep CNN network

### Data Partition

In order to improve the accuracy of prediction, we intend to train different parameters of the model for different types of pictures, so we need to divide the training pictures.

We selected k images for each of types of images, then performed image retrieval for each image set, and used the image retrieval results to divide the dataset into three categories, and then trained the three types of images separately.

![1542369041510](img\1542369041510.png)

### Density Map

The process of generating the density map is actually a Gaussian filtering process, mapping a very sparse graph (pedestrian marker) onto a density map.

G_σ=  1/(2 πσ) e^(-(x^2+y^2)/2σ^2 )

Although this step seems unremarkable, it provides the possibility of training for convolutional neural networks, and because of the size of the density map and the original image, we can focus on designing a convolutional network without the need for dense layers. Not affected by the size of the image.

![1542369133474](img\1542369133474.png)

### Deep CNN Network

Our network is divided into two parts. The front layer is the VGG16 layer. After the vgg16 convolution layer and the four pooling layers, the size of the picture becomes 1/8 of the original. The latter layer is the
dilated  convolution part.  we modify the dense layer of vgg to a number of dilated convolutions.

![1542369196522](img\1542369196522.png)

## references

[1] Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). Single-image crowd counting via multi-column convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 589-597).

[2] Idrees, H., Saleemi, I., Seibert, C., & Shah, M. (2013, June). Multi-source multi-scale counting in extremely dense crowd images. In Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on (pp. 2547-2554). IEEE.

[3] Zhang, C., Li, H., Wang, X., & Yang, X. (2015, June). Cross-scene crowd counting via deep convolutional neural networks. In Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on (pp. 833-841). IEEE.

[4] Chen, K., Loy, C. C., Gong, S., & Xiang, T. (2012). Feature mining for localised crowd counting. In BMVC (Vol. 1, No. 2, p. 3).
