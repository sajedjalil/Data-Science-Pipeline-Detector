"""
Author: Yash Patel
Name: PhysicsFeatureImportance.py 
Description: Determines and plots the relative importances of the
features given in the raw data for the particle collisions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

train = pd.read_csv('../input/training.csv', index_col='id')

variables = train.columns[0:-5]
baseline = GradientBoostingClassifier(n_estimators=50, learning_rate=0.005, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
#baseline = RandomForestClassifier(n_estimators=25)

baseline.fit(train[variables], train['signal'])

# sort importances
indices = np.argsort(baseline.feature_importances_)
# plot as bar chart
plt.barh(np.arange(len(variables)), baseline.feature_importances_[indices])
plt.yticks(np.arange(len(variables)) + 0.25, np.array(variables)[indices])
_ = plt.xlabel('Relative importance')
plt.savefig("Physics_Feature_Importance.png")