#This is fork from Single Unified Table ~0.94 Sklearn by jeffd23

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline,FeatureUnion, make_union
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import xgboost as xgb

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

# Training 
features = actions_train.merge(peeps, how='left', on='people_id')
labels = act_train['outcome']

# Testing
test = actions_test.merge(peeps, how='left', on='people_id')
## Split Training Data

num_test = 0.10
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=num_test, random_state=23)

XGB = xgb.XGBClassifier()
ETC = ExtraTreesClassifier()
RFC = RandomForestClassifier()
DTC = DecisionTreeClassifier()

CLASS = VotingClassifier(estimators=[('ETC', ETC),('RFC',RFC),('DTC',DTC),('XGB',XGB)], voting='soft')
# Can not run PolynomialFeatures not enought memory 
pipeclf = Pipeline([#('FEATURE', PolynomialFeatures(degree=2,include_bias=False)),
                     ('SCALE', MinMaxScaler()),
                     ('CLASS', CLASS)])
                     
print('Start training')
pipeclf.fit(X_train, y_train)

## Training Predictions
proba = pipeclf.predict_proba(X_test)
preds = proba[:,1]
score = roc_auc_score(y_test, preds)
print("Area under ROC {0}".format(score))

# Test Set Predictions
print('Make predictions and submission')
test_proba = pipeclf.predict_proba(test)
test_preds = test_proba[:,1]

# Format for submission
output = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': test_preds })
output.to_csv('redhat.csv', index = False)