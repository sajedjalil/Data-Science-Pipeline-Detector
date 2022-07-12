import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# read in raw data .csv from Kaggle 
train = pd.read_csv('../input/train.csv', index_col='ID')
test = pd.read_csv('../input/test.csv', index_col='ID')
test_ID = test.index

y = train['target']
train = train.drop(['target'], axis=1)
train = train.drop(['v3'], axis=1)

big = train.append(test)
big = big.interpolate()

non_num = ['v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 
            'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 
            'v113', 'v125']

lbl = LabelEncoder()
for i in non_num: 
    print(i)
    big[i] = lbl.fit_transform(big[i])

# remove features with less than 80% variance  
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
big = sel.fit_transform(big)

# Prepare the inputs for the model
X = big[0:train.shape[0]]#.as_matrix()
test = big[train.shape[0]::]#.as_matrix()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X, y)
predictions = gbm.predict(test)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'ID': test_ID,
                            'PredictedProb': predictions })
submission.to_csv("submission.csv", index=False)