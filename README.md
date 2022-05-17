# Deep Residual Learning for Image Recognition Pytorch Implementation

This repository is `Deep Residual Learning for Image Recognition` known as `ResNet` Pytorch third-party implementation.

PDF: https://arxiv.org/pdf/1512.03385.pdf

Arxiv: https://arxiv.org/abs/1512.03385

Original implementation of Kaiming He uses `caffe`: https://github.com/KaimingHe/deep-residual-networks, but many other implementations can be found, e.g., Pytorch official implementation: https://pytorch.org/hub/pytorch_vision_resnet/.

Notes:

- The network architecture detailed in the paper is know as `ResNet V1`. According to different implementation and their experiments, there are `ResNet V1.5` and `ResNet V2`. In this repository, I simply re-implement `ResNet V1`, so, there is disparity between my implementation and Pytorch official's.

- I trained AlexNet on `Cifar10`, `Cifar100`, and `PascalVOC2012`. Due to the computation limitation, I didn't train on larger datasets like `ImageNet`.

- Since images in `Cifar10` and `Cifa100` are pretty tiny (32 * 32 * 3), using 7*7 kernel in the first convolutional layer (input_transform) will simply crash the network. There are two ways to address the problem:

  1. Modify the first layer kernel size
  2. Resize the image from 32 to 224 before put into the network

  I have tested both methods, finding that: **Resizing the image size and following training settings (especially lr starts from 0.1 and decreases) will results in a much much more better results**

- `The standard training epochs for ImageNet is 100`, more specifically, as you can find the following training setups in NVIDIA training logs (https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch)

  - 50 Epochs -> configuration that reaches 75.9% top1 accuracy
  - 90 Epochs -> 90 epochs is a standard for `ImageNet` networks
  - 250 Epochs -> best possible accuracy.

- Current version of the repository is V2 that I basically rewrite all the code, so, you can only pull the v2 branch and run the codes.



### Findings

1. **Using larger input image and using original setting is better**. As aforementioned, resizing `Cifar` image from 32 \* 32 to 256 \* 256 and apply center crop to get 224 \* 224 results a much better performance (32 \* 32 image with first transformation later of 3 \* 3 ksize only achieve ~63% top1 accuracy, while 224 \* 224 image can easily achieve ~90% top1 accuracy). 
2. **Using large learning rate for large input is very important**. Learning with 0.1 learning rate, the model can reach the standard results.
3. **Downsampling of the first block in the first layer is important for tiny inputs**. My implementation shows great degradation compared with torch implementation for tiny image (32 \* 32) of Cifar, but after resize the image from 32 \* 32, the degradation vanished.
   1. Further implementation of do not using downsampling in my model is needed
