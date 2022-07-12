import pandas as pd
import numpy as np
import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from operator import itemgetter
import time
import copy

seed = 2016
np.random.seed(seed)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))

def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    return sorted(output)


def simple_load():

    print("Read people.csv...")
    people = pd.read_csv("../input/people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv("../input/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test = pd.read_csv("../input/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

    print("Process tables...")
    for table in [train, test]:
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True)
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    print("Merge...")
    train = train.merge(people, on="people_id", suffixes=("_act", ""))
    test = test.merge(people, on="people_id", suffixes=("_act", ""))
    
    # Set index to activity id
    train = train.set_index("activity_id")
    test = test.set_index("activity_id")

    # Correct some data types
    for field in ["date_act", "date"]:
        train[field] = pd.to_datetime(train[field])
        test[field] = pd.to_datetime(test[field])

    return train, test

def xgboost_process(train,test,features):
    print("Process tables... ")
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table.drop('date', axis=1, inplace=True)
        table.drop('date_act', axis=1, inplace=True)
        table.drop('people_id', axis=1, inplace=True)
    features.remove('date')
    features.remove('date_act')
    return train, test, features
    
def model():

    # Load in the data set simply by merging together
    train, test = simple_load()
    
    # Get features
    features = get_features(train,test)
    
    # Processing for XGBoost
    
    train, test, features = xgboost_process(train, test, features)
    
    X_train, X_valid = train_test_split(train, test_size=0.1, random_state=0)
    
    # Outcome in y
    y = X_train['outcome']
    del X_train['outcome']
    
    y_check = X_valid['outcome']
    del X_valid['outcome']
    
    params = {
    'objective': 'binary:logistic',
    'max_depth': 10,
    'learning_rate': 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8, 
    "silent": 0,
    'n_estimators': 100
}
    
    
    print('Starting... ')
    start_time = time.time()
    gbm = XGBClassifier(**params)
    gbm.fit(X_train,y)
    check = gbm.predict(X_valid)
    
    print("Validating...")
    score = roc_auc_score(y_check, check)
    print('Check error value: {:.6f}'.format(score))
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))


def main():

    # Write a benchmark file to the submissions folder
    model()

if __name__ == "__main__":
    main()