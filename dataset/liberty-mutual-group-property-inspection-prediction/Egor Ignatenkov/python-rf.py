# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no python loops, zips, etc.
# Significantly (>30x) faster than previous implementions

import numpy as np 

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

for i in range(train.shape[1]):
    if train.iloc[:,i].dtype=='O':
        train.iloc[:,i]=le.fit_transform(train.iloc[:,i])
        
predictors=train.columns.values.tolist()[2:10]

for i in range(test.shape[1]):
    if test.iloc[:,i].dtype=='O':
        test.iloc[:,i]=le.fit_transform(test.iloc[:,i])

from sklearn.cross_validation import KFold
kf = KFold(train.shape[0], n_folds=10)

from sklearn.ensemble import RandomForestClassifier        
alg=RandomForestClassifier()
#alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

predictions = []
for trrain, ttest in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (train[predictors].iloc[trrain,:])
    # The target we're using to train the algorithm.
    train_target = train["Hazard"].iloc[trrain]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(train[predictors].iloc[ttest,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)


#alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
#alg.fit(train[predictors],train['Hazard'])
#predictions=alg.predict(train[predictors])

print(Gini(train['Hazard'], predictions))