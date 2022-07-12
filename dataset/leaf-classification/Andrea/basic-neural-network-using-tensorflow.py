import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


#read data set
train_file = "../input/train.csv"
test_file = "../input/test.csv"
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)


x_train = train.drop(['species', 'id'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

x_test = test.drop(['id'], axis=1).values

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# Build 3 layer DNN with 1024, 512, 256 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[1024,512,256],
n_classes=99)

# Fit model.
classifier.fit(x=x_train, y=y_train, steps = 2000)

# Make prediction for test data
y = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)

# prepare csv for submission
test_ids = test.pop('id')
submission = pd.DataFrame(y_prob, index=test_ids, columns=le.classes_)
submission.to_csv('submission_log_reg.csv')