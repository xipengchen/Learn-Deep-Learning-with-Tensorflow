# Keras Syntax Basics With Regression Model
> *Created on: June 22, 2020*<br/>
> *Written by: Xp Chen*<br/>
* This note contains Keras Syntax to processing data and conduct EDA wit raw data
* Keras Regression Model to Housing price (data sourse from Kaggle Database -- [King County, WA](https://www.kaggle.com/harlfoxem/housesalesprediction))

* With TensorFlow 2.0 , Keras is now the main API choice. Let's work through a simple regression project to understand the basics of the Keras syntax and adding layers.

## 1. The Data

To learn the basic syntax of Keras, we will use a very simple fake data set, in the subsequent lectures we will focus on real datasets, along with feature engineering! For now, let's focus on the syntax of TensorFlow 2.0.

Let's pretend this data are measurements of some rare gem stones, with 2 measurement features and a sale price. Our final goal would be to try to predict the sale price of a new gem stone we just mined from the ground, in order to try to set a fair price in the market.

### 1.1 Load the Data
```python
import pandas as pd
import numpy as np

df = pd.read_csv('../DATA/fake_reg.csv')
df.head()
```

### 1.2 Exploratory Data Analysis
#### 1.2.1 Exploratory Data Analysis (for housing price related to each house feature)
```python
import seaborn as sns
import matplotlib.pyplot as plt

# To plot all features and to see the relation for each two features
sns.pairplot(df)

# check the null values
df.isnull().sum()

df.describe().transpose()

# Plot more columns to understand
plt.figure(figsize=(12,8))

# Housing data King country from Kaggle
sns.distplot(df['price'])

sns.countplot(df['bedrooms'])

plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='sqft_living',data=df)

sns.boxplot(x='bedrooms',y='price',data=df)
```

#### 1.2.1 Geographical Properties (for housing location)
```python
plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='long',data=df)

plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='lat',data=df)

plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')

# Processing Anamoly data
df.sort_values('price',ascending=False).head(20)
len(df)*(0.01)
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]

plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=non_top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)

```

### 1.3 Other Features
```python
sns.boxplot(x='waterfront',y='price',data=df)

```
## 2. Working with Feature Data
* To get rid of useless features like ID, which will not be helpful for future model learning
* To find out other features like (date datatype) that will be helpful for further learning

### 2.1 Working with Feature Data

```python
df.info()

# e.g drop ID column
df = df.drop('id',axis=1)

# Feature Engineering from Date
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

sns.boxplot(x='year',y='price',data=df)
sns.boxplot(x='month',y='price',data=df)

df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()

df = df.drop('date',axis=1)

# May be worth considering to remove this or feature engineer categories from it
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)

# could make sense due to scaling, higher should correlate to more value
df['yr_renovated'].value_counts()

df['sqft_basement'].value_counts()
```



### 2.2 Train Test Split
```python
# split label and features
X = df.drop('price',axis=1)
y = df['price']

# Convert Pandas to Numpy for Keras
# After scaling will convert dataframe to array

# Features
X = df[['feature1','feature2']].values

# Label
y = df['price'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
```

### 2.3 Scaling the data
```python
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

# Notice to prevent data leakage from the test set, we only fit our scaler to the training set
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train.shape
X_test.shape

```

### 2.4  Creating a Model
There are two ways to create models through the TF 2 Keras API, either pass in a list of layers all at once, or add them one by one
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error,mean_squared_error
model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

# Final output node for prediction
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
```
### 2.5 Choosing an [optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) and loss

Keep in mind what kind of problem you are trying to solve:
```python
# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')
```

## 3. Training the model

##### Below are some common definitions that are necessary to know and understand to correctly utilize Keras:

* Sample: one element of a dataset.
    * Example: one image is a sample in a convolutional network
    * Example: one audio file is a sample for a speech recognition model
* Batch: a set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model.A batch generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to process and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluation/prediction).
* Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
* When using validation_data or validation_split with the fit method of Keras models, evaluation will be run at the end of every epoch.
* Within Keras, there is the ability to add callbacks specifically designed to be run at the end of an epoch. Examples of these are learning rate changes and model checkpointing (saving).

```python
model.fit(X_train,y_train,epochs=250,verbose =1)

# Don't forget to transfer df to np.array
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)
```
## 4. Evaluate the model

[Documentation about model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

### 4.1 Get the loss curve

```python
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

# To see the trend of losses with the number of epochs
# 1
losses = pd.DataFrame(model.history.history)
losses.plot()

# 2
loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)),y=loss)
plt.title("Training Loss per Epoch");

```
### 4.2 Compare final evaluation (MSE) on training set and test set.

These should hopefully be fairly close to each other.

```python
model.metrics_names

training_score = model.evaluate(X_train,y_train,verbose=0)
test_score = model.evaluate(X_test,y_test,verbose=0)

# Make Prediction
predictions = model.predict(X_test)
# get some evaluation values
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
explained_variance_score(y_test,predictions)

# Our predictions
plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r')

# Get difference between test data and prediction data
errors = y_test.values.reshape(6480, 1) - predictions
sns.distplot(errors)

# for test label data and predictions
# We can concat each column together and get the difference
# or error and then plot the error

```

## 5. Predicting on brand new data

What if we just saw a brand new gemstone from the ground? What should we price it at? This is the **exact** same procedure as predicting on a new test data!
```python
# [[Feature1, Feature2]]
new_gem = [[998,1000]]

# Don't forget to scale!
scaler.transform(new_gem)
model.predict(new_gem)

###########################################################################
# Using dataframe last row
single_house = df.drop('price',axis=1).iloc[0]
single_house
single_house = scaler.transform(single_house.values.reshape(-1, 19))
model.predict(single_house)

```
## 6. Saving and Loading a Model

```python
from tensorflow.keras.models import load_model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# Load Model
later_model = load_model('my_model.h5')
later_model.predict(new_gem)
```

