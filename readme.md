# CSRnet

## Introduction
The crowd density estimation method, which is applied to crowded scenes, has been widely used in airports, stations, operating vehicles, art galleries, etc. because of its accuracy and speed far above the naked eye count. It can effectively prevent hidden dangers such as crowded trampling and overloading, and on the other hand, it can help retailers to collect passenger traffic.
The training and test pictures used in this competition are all from the general monitoring scene, but include a variety of perspectives (such as low altitude, high altitude, fisheye, etc.), the relative size of the pedestrians in the map will also have a large difference. Some training data refer to public data sets (such as ShanghaiTech [1], UCF-CC-50 [2], WorldExpo'10 [3], Mall [4], etc.).

## Our Model
Population density estimates can be divided into three main categories: detection-based methods, regression-based methods, and density estimation-based methods.
The fundamental idea of the proposed design is to deploy a deeper CNN for capturing high-level features with larger receptive fields and generating high-quality densitymaps without brutally expanding network complexity.
Coupled with the analysis of the above problems, we propose such a model framework
> Data Partition
> Density Map
> Deep CNN network

## references
[1] Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). Single-image crowd counting via multi-column convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 589-597).

[2] Idrees, H., Saleemi, I., Seibert, C., & Shah, M. (2013, June). Multi-source multi-scale counting in extremely dense crowd images. In Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on (pp. 2547-2554). IEEE.

[3] Zhang, C., Li, H., Wang, X., & Yang, X. (2015, June). Cross-scene crowd counting via deep convolutional neural networks. In Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on (pp. 833-841). IEEE.

[4] Chen, K., Loy, C. C., Gong, S., & Xiang, T. (2012). Feature mining for localised crowd counting. In BMVC (Vol. 1, No. 2, p. 3).
