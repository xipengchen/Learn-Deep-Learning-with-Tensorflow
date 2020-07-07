# Keras Convolutional Neural Networks (CNNs)
> *Created on: June 28, 2020*<br/>
> *Written by: Xp Chen*<br/>

Let's explore CNNs with Keras API for TF 2.0

[A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
## :question: What is Fliter :question:
  In CNNs, **filters are not defined**. The value of each filter is learned during the training process.

  By being able to learn the values of different filters, CNNs can find more meaning from images that humans and human designed filters might not be able to find.

  More often than not, we see the filters in a convolutional layer learn to detect abstract concepts, like the boundary of a face or the shoulders of a person. By stacking layers of convolutions on top of each other, we can get more abstract and in-depth information from a CNN. [source from](https://www.saama.com/different-kinds-convolutional-filters/)
  
## :question: Which are hyperparameters in CNNs :question:
![](https://github.com/xipengchen/Learn-Deep-Learning-with-Tensorflow/blob/master/indicated_image/hyperparameter.png)
![](https://github.com/xipengchen/Learn-Deep-Learning-with-Tensorflow/blob/master/indicated_image/hyperparameter%202.png)

## :question: What is ImageDataGenerator() ? :question:
   `ImageDataGenerator()` is [Image Augmentation](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/) for Deep Learning With Keras 


<img src="" alt="" width="400" height="200">
