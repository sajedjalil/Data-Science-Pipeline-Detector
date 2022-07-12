# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Load the packages for modeling
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Seperate out predictors and target from the training data set
# Remove the ID field from the test dataset and save it.
# Drop the ID field from the training set
train_y = train['TARGET']
train.drop(['ID', 'TARGET'], axis=1, inplace=True)
train_x = train
test_id = test['ID']
del test['ID']


### Missing value imputation 
### Remove duplicate and constant column

# Fixing the outliers in column 'var3'
train_x['var3'].replace(-999999,0, inplace=True)
test['var3'].replace(-999999,0, inplace=True)


# Remove all the columns which have constant values. 
# These columns have zero std deviation.
rm_col=[] 
for col in train_x.columns:
    if train_x[col].std()==0:
        rm_col.append(col)

train_x.drop(rm_col, axis=1, inplace=True)
test.drop(rm_col, axis=1, inplace=True)

# Remove the duplicate columns. 
# Here we have columns with different name but exactly same values for each rows
# We will compare all pairs of columns
dups_col = []
for ii in range(len(train_x.columns)-1):
    for jj in range(ii+1,len(train_x.columns)):
        col1=train_x.columns[ii]
        col2=train_x.columns[jj]
        # take the columns as arrays adn then compare the values.
        if np.array_equal(train_x[col1].values, train_x[col2].values) and not col2 in dups_col:
            dups_col.append(col2)

train_x.drop(dups_col, axis=1, inplace=True)
test.drop(dups_col, axis=1, inplace=True)


### Feature selection using XGBoost classifier
#### We will leverage the feature importance attribute of the XGBoost  classifier to find top 50 features.

# Define XGBoost classifier with some standard parameter settings
xgb_clf = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, min_child_weight=1,
                           gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                           nthread=4,seed=10)

# Learn the model with training data
xgb_clf.fit(train_x,train_y)

# Plot the top 50 important features
imp_feat_xgb=pd.Series(xgb_clf.feature_importances_, index=train_x.columns).sort_values(ascending=False)
imp_feat_xgb[:50].plot(kind='bar',title='Top 50 Important features as per XGBoost', figsize=(12,8))
plt.ylabel('Feature Importance Score')
plt.subplots_adjust(bottom=0.25)
plt.savefig('FeatureImportance.png')
plt.show()


# Save indexes of the important features in descending order of their importance
indices = np.argsort(xgb_clf.feature_importances_)[::-1]

# list the names of the names of top 50 selected features adn remove the unicode
select_feat =[str(s) for s in train_x.columns[indices][:60]]

# Make the subsets with 50 features only
train_x_sub = train_x[select_feat]
test_sub = test[select_feat]


### Predict with best model

### I have done step by step parameter tuning to come up with the below estimator values.
### Please refer to my notebook https://www.kaggle.com/sanjib03/santander-customer-satisfaction/xgb-feat-selection-and-param-tuning

# Take the best model 
best_xgb_clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.7,
                               gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
                               min_child_weight=5, missing=None, n_estimators=75, nthread=-1,
                               objective='binary:logistic', reg_alpha=0.01, reg_lambda=1,
                               scale_pos_weight=1, seed=10, silent=True, subsample=0.7)
                               
                               
# Learn model 
best_xgb_clf.fit(train_x_sub,train_y)

                             
# Make prediction with test data
predicted_proba = best_xgb_clf.predict_proba(test_sub)

# Save the prediction in CSV file
# predicted_proba has probabilities for each Target class for each observation.
# We are concerned about probability of class 1 and hence taking predicted_proba[:,1]
submission = pd.DataFrame({'ID':test_id,'TARGET':predicted_proba[:,1]})
submission.to_csv('submission.csv', index=False)       
