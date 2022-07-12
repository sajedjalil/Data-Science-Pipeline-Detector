import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#read data set
train_file = "../input/train.csv"
test_file = "../input/test.csv"
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)


x_train = train.drop(['species', 'id'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

# Build 3 layer DNN with 1024, 512, 256 units respectively.
classifier = svm.SVC(kernel="linear", gamma=1, C=1.)

# Fit model.
classifier.fit(x_train,y_train)
print(y_train)

# Make prediction for test data
x_test = test.drop(['id'], axis=1).values
x_test = scaler.transform(x_test)

y = classifier.predict(x_test)

y_prob = []
for p in y:
    tmp = [0 if i != p else 1 for i in range(99)]
    y_prob.append(tmp)

# prepare csv for submission
test_ids = test.pop('id')
submission = pd.DataFrame(y_prob, index=test_ids, columns=le.classes_)
submission.to_csv('submission_log_reg.csv')