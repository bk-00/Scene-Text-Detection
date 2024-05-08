# Scene-Text-Detection
A simulation project for detecting text region in a scene text image.

It applies machine learning technology which the network is trained with the Vgg-16 model in PyTorch framework.
Training iteration is about 5000 iterations and with 1500 training images.
The network is trained based on the pre-trained Faster Region Convolutional Neural Network and with CPU mode only.
As training is heavy and insufficient training images, the accuracy is not high, which is around 27%. 
Below are some output images: 

![Fig3](https://github.com/bk-00/Scene-Text-Detection/assets/30174905/c7a7495d-8bbf-4aaa-b93a-025ec0cace9c)

![Fig2](https://github.com/bk-00/Scene-Text-Detection/assets/30174905/68c17ba5-4dce-4080-9388-331472494831)

![Fig1](https://github.com/bk-00/Scene-Text-Detection/assets/30174905/e7ed241f-83b5-4195-b715-06a023536af8)

Notice that not all text in the scene text images is properly captured by the bounding box.
