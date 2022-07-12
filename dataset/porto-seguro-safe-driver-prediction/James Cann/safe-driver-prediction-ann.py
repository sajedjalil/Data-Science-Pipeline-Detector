# Basic ANN model with 57 input layers, 57 middle layers


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset for training and validation
dataset = pd.read_csv('../input/train.csv')

# Importing the dataset for submission
dataset_submit = pd.read_csv('../input/test.csv')

#splitting the training dataset into training and testing chunks
#independant variables
X = dataset.iloc[:,1:58].values
X_submit = dataset_submit.iloc[:,1:58].values
X_submit_id = dataset_submit.iloc[:,0].values

#dependant variables
y = dataset.iloc[:,58].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_submit = sc.transform(X_submit)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 114, kernel_initializer = 'uniform', activation = 'relu', input_dim = 57))

# Adding the second hidden layer
classifier.add(Dense(units = 57, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 57, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, verbose=2, batch_size = 10000, epochs = 1000)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting the Submission set results
y_pred_submit = classifier.predict(X_submit)
#y_pred_submit = (y_pred_submit > 0.5)

#Create the Kaggle submission
y_pred_submit_int=[['id','target']]
for i in range(len(X_submit)):
    pair=[X_submit_id[i],y_pred_submit[i][0]]
    y_pred_submit_int.append(pair)

#Write submission to storage    
raw_data = y_pred_submit_int
df = pd.DataFrame(raw_data)
df.to_csv(path_or_buf = 'submission03.csv', index=None, header=False)