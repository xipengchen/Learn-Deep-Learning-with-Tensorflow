# Keras Convolutional Neural Networks (CNNs) Coding Along
> *Created on: July 05, 2020*<br/>
> *Written by: Xp Chen*<br/>

Real Example Notebook to Use [CNNs for Deep Learning on Custom Images-Malaria](https://github.com/xipengchen/Learn-Deep-Learning-with-Tensorflow/blob/master/03-Deep-Learning-Custom-Images-Malaria.ipynb)

## 1. The Data (MNIST Digit Numbers) & (CIFAR-10 Multiple Classes)

```python
import pandas as pd
import numpy as np

# Loading the data from keras datasets

# MNIST
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# CIFAR-10 is a dataset of 50,000 32x32 color training images, 
# labeled over 10 categories, and 10,000 test images.
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape
x_train[0].shape

import matplotlib.pyplot as plt
# Show the images
plt.imshow(x_train[0])
```
### 2. PreProcessing Data
