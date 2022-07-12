
import os

# Random Forest Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Training Dataset

tr_dataset = pd.read_csv('../input/train/train.csv')
X_train = tr_dataset.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]].values
y_train = tr_dataset.iloc[:, [23]].values

te_dataset = pd.read_csv('../input/test/test.csv')
X_test = te_dataset.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22]].values

# print(X)
# print(y)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred

te_dataset['AdoptionSpeed'] = y_pred

te_dataset[['PetID', 'AdoptionSpeed']].to_csv('RandomForest-submission.csv', index=False)



