# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

"""
This is my first attempt at predicting the probabilities that a claim is suitable for an accelerated approval.
The model is not very sophisticated. I will be doing more in-depth entries with better preprocessing.
"""


import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
ss = pd.read_csv("../input/sample_submission.csv")

colNames = list(train.columns)

# Separate the data and labels.
X_train_df = train.iloc[:, 2:len(colNames)]
X_test_df = test.iloc[:,1:len(colNames)]
y = train.iloc[:,1]

# Replace missing values with mean of attributes (Not very effective)
for i in range(X_train_df.shape[1]):
    if X_train_df.iloc[:,i].dtype == 'float64':
        X_train_df.iloc[:,i].fillna(np.mean(X_train_df.iloc[:,i]), inplace = True)
        
for i in range(X_test_df.shape[1]):
    if X_test_df.iloc[:,i].dtype == 'float64':
        X_test_df.iloc[:,i].fillna(np.mean(X_test_df.iloc[:,i]), inplace = True)


# Slim data down to just the numerical columns
X_train_array = X_train_df.select_dtypes(['float64']).as_matrix()
X_test_array = X_test_df.select_dtypes(['float64']).as_matrix()

# train random forest model
rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 500, random_state = 1, n_jobs = 2)
rf.fit(X_train_array, y)

# predict probabilities and export to CSV
predictions = rf.predict_proba(X_test_array)
predictions = DataFrame(predictions[:,1])
submission1 = DataFrame(test["ID"])
submission1["PredictedProb"] = predictions
submission1.to_csv("submission1.csv", index = False)