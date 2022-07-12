import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

train = pd.read_csv("../input/train_info.csv")

train['date'].str.extract('(\d)')

train_data = train.values

print(train_data[:, 2:])

recognizer = RandomForestClassifier(n_estimators = 10)
recognizer.fit(train_data[:, 2:],train_data[:, 1])
print("in-sample score")
print(recognizer.score(train_data[:, 1:],train_data[:, 0]))
print("feature importance")
#print(recognizer.feature_importances_)
print(recognizer.coef_)
print("cross validation score")
score = cross_val_score(recognizer, train_data[:, 1:],train_data[:, 0])
score = np.mean(score)
print(score)