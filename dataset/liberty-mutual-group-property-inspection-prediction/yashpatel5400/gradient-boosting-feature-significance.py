"""
Author: Yash Patel
Name: LibertyMutual.py 
Description: Predicts the hazard score based on provided
anonymized columns
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv', index_col='Id')

columns = train.drop(['Hazard'], axis=1).columns
for column in columns:
    uniqueVals = train[column].unique()
    if not isinstance(uniqueVals[0], int):
        mapper = dict(zip(uniqueVals, range(len(uniqueVals))))
        train[column] = train[column].map(mapper).astype(int)

print("Train a Gradient Boosting model")
clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
# print("Train a Random Forest model")
# clf = RandomForestClassifier(n_estimators=25)

clf.fit(train[columns], train['Hazard'])

# sort importances
indices = np.argsort(clf.feature_importances_)
# plot as bar chart
plt.barh(np.arange(len(columns)), clf.feature_importances_[indices])
plt.yticks(np.arange(len(columns)) + 0.25, np.array(columns)[indices])
_ = plt.xlabel('Relative importance')
plt.savefig("Liberty_Feature_Importance.png")