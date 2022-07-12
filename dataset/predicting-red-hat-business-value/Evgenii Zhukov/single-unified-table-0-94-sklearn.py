# ## Preprocessing/Merging People and Activities
# 
# This script converts features in people and activities into integers, then merges everything into a single table. Makes it easy to drop into classifiers in Sklearn or XGBoost. 
# 
# Conveniently, most of the data can be easily encoded to numeric values with simple string splitting. 
# 
# Scored ~0.944 with Random Forest Classifier in Sklearn out of the box. 
# 
# 

import numpy as np
import pandas as pd

act_train = pd.read_csv('../input/act_train.csv')
act_test = pd.read_csv('../input/act_test.csv')
people = pd.read_csv('../input/people.csv')
# Save the test IDs for Kaggle submission
test_ids = act_test['activity_id']

def preprocess_acts(data, train_set=True):
    
    # Getting rid of data feature for now
    data = data.drop(['date', 'activity_id'], axis=1)
    if(train_set):
        data = data.drop(['outcome'], axis=1)
    
    ## Split off _ from people_id
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    
    columns = list(data.columns)
    
    # Convert strings to ints
    for col in columns[1:]:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)
    return data

def preprocess_people(data):
    
    # TODO refactor this duplication
    data = data.drop(['date'], axis=1)
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    
    #  Values in the people df is Booleans and Strings    
    columns = list(data.columns)
    bools = columns[11:]
    strings = columns[1:11]
    
    for col in bools:
        data[col] = pd.to_numeric(data[col]).astype(int)        
    for col in strings:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)
    return data
# Preprocess each df
peeps = preprocess_people(people)
actions_train = preprocess_acts(act_train)
actions_test = preprocess_acts(act_test, train_set=False)
# Merege into a unified table

# Training 
features = actions_train.merge(peeps, how='left', on='people_id')
labels = act_train['outcome']

# Testing
test = actions_test.merge(peeps, how='left', on='people_id')

# Check it out...
features.sample(10)
## Split Training Data
from sklearn.cross_validation import train_test_split

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=num_test, random_state=23)

## Out of box random forest
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV

clf = RandomForestClassifier()
clf = xgb.XGBClassifier(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)
## Training Predictions
proba = clf.predict_proba(X_test)
preds = proba[:,1]
score = roc_auc_score(y_test, preds)
print("Area under ROC {0}".format(score))
# Test Set Predictions
test_proba = clf.predict_proba(test)
test_preds = test_proba[:,1]

# Format for submission
output = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': test_preds })
output.head()
output.to_csv('redhat.csv', index = False)