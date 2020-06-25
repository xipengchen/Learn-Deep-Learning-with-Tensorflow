# Keras Classification
> *Created on: June 25, 2020*<br/>
> *Written by: Xp Chen*<br/>

Let's explore a classification task with Keras API for TF 2.0

*This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets. https://goo.gl/U2Uwz2*

## 1. The Data
### 1.1 Load the Data
```python
import pandas as pd
import numpy as np

df = pd.read_csv('../DATA/cancer_classification.csv')
df.head()
df.info()
df.describe().transpose()
```

### 1.2 Exploratory Data Analysis
#### 1.2.1 Exploratory Data Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

# To plot all features and to see the relation for each two features
sns.countplot(x='benign_0__mal_1',data=df)

sns.heatmap(df.corr())

# check the null values
df.isnull().sum()

df.describe().transpose()

# Plot more columns to understand
plt.figure(figsize=(12,8))



plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')

# Processing Anamoly data
df.corr()['benign_0__mal_1'].sort_values()
df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
```

## 2. Working with Feature Data
* To get rid of useless features like ID, which will not be helpful for future model learning
* To find out other features like (date datatype) that will be helpful for further learning

### 2.1 Train Test Split
```python
# split label and features
X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
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
    # For a binary classification problem
       model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')
```

## 3. Training the model & How to deal with "Model Overfitting"

### 3.1 Example One: Choosing too many epochs and overfitting!

```python
# https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch

model.fit(x=X_train,
          y=y_train,
          epochs=600,
          validation_data=(X_test, y_test), verbose=1)

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
```

### 3.2 Example Two: Early Stopping

We obviously trained too much! Let's use early stopping to track the val_loss and stop training once it begins increasing too much!
```python
model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
# help(EarlyStopping) to learn more details about arguments
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train,
          y=y_train,
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
```
### 3.3 Example Three: Adding in DropOut Layers

```python
from tensorflow.keras.layers import Dropout

# Dropout means that turning off how many percentage of neurons for each layer

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train,
          y=y_train,
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

```

## 4. Model Evaluation

```python
predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report,confusion_matrix
# https://en.wikipedia.org/wiki/Precision_and_recall
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
```
