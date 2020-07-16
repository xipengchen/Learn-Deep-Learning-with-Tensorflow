# Keras Convolutional Neural Networks (CNNs) Coding Along
> *Created on: July 05, 2020*<br/>
> *Written by: Xp Chen*<br/>

:red_circle: **Real Example Notebook to Use [CNNs for Deep Learning on Custom Images-Malaria](https://github.com/xipengchen/Learn-Deep-Learning-with-Tensorflow/blob/master/03-Deep-Learning-Custom-Images-Malaria.ipynb)**

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
## 2. PreProcessing Data
### 2.1 PreProcessing Labels (y)
We first need to make sure the labels will be understandable by our CNN.
```python
# Check labels
y_train[0]

# If labels are literally categories of numbers, need to  translate this to be "one hot encoded" so that CNN can understand, otherwise it will think this is some sort of regression problem on a continuous axis.
# Luckily , Keras has an easy to use function for this:
from tensorflow.keras.utils import to_categorical
y_train.shape
y_example = to_categorical(y_train)
y_example.shape


y_cat_train = to_categorical(y_train,10) # 10 means 10 categories for labels
y_cat_test = to_categorical(y_test,10)


```

### 2.2 Reshaping the Data
 CNN need to add one more dimension to show we're dealing with 1 RGB channel (since technically the images are in black and white, only showing values from 0-255 on a single channel), an color image would have 3 dimensions.
```python
x_train.shape # output = (60000,28,28)

# Reshape to include channel dimension (in this case, 1 channel)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)

```

## 3. Building the Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


model = Sequential()

## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])  # we can add in additional metrics https://keras.io/metrics/
# Summary
model.summary()
```

## 4. Training the model
```python
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=23

model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

```


## 5. Model Evaluation

```python
model.metrics_names

losses = pd.DataFrame(model.history.history)
losses.head()

losses[['accuracy','val_accuracy']].plot()
losses[['loss','val_loss']].plot()

print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))

from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(x_test)
confusion_matrix(y_test,predictions)

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
# https://github.com/matplotlib/matplotlib/issues/14751

```
## 6. Predicting a given image

```python
my_image = x_test[16]
plt.imshow(my_image)

# SHAPE --> (num_images,width,height,color_channels)
model.predict_classes(my_image.reshape(1,32,32,3))


```
