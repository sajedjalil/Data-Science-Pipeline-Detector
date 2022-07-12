# Loading libraries
import pandas as pd
import numpy as np
# Global constants and variables
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
train = pd.read_csv('../input/'+TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)
test = pd.read_csv('../input/'+TEST_FILENAME, parse_dates=['Dates'], index_col=False)
train.info()
train = train.drop(['Descript', 'Resolution', 'Address'], axis = 1)
test = test.drop(['Address'], axis = 1)
def feature_engineering(data):
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['DayOfWeek'] = data['Dates'].dt.dayofweek
    data['WeekOfYear'] = data['Dates'].dt.weekofyear
    return data
train = feature_engineering(train)
test = feature_engineering(test)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
train['PdDistrict'] = enc.fit_transform(train['PdDistrict'])
category_encoder = LabelEncoder()
category_encoder.fit(train['Category'])
train['CategoryEncoded'] = category_encoder.transform(train['Category'])
print(category_encoder.classes_)
enc = LabelEncoder()
test['PdDistrict'] = enc.fit_transform(test['PdDistrict'])
print(train.columns)
print(test.columns)
x_cols = list(train.columns[2:12].values)
x_cols.remove('Minute')
print(x_cols)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 10)
clf.fit(train[x_cols], train['CategoryEncoded'])
test['predictions'] = clf.predict(test[x_cols])
def field_to_columns(data, field, new_columns):
    for i in range(len(new_columns)):
        data[new_columns[i]] = (data[field] == new_columns[i]).astype(int)
    return data
test['Category'] = category_encoder.inverse_transform(test['predictions'])
categories = list(category_encoder.classes_)
test = field_to_columns(test, 'Category', categories)
import time
PREDICTIONS_FILENAME_PREFIX = 'predictions_'
PREDICTIONS_FILENAME = PREDICTIONS_FILENAME_PREFIX + time.strftime('%Y%m%d-%H%M%S') + '.csv'
print(test.columns)
submission_cols = [test.columns[0]]+list(test.columns[14:])
print(submission_cols)
print(PREDICTIONS_FILENAME)
test[submission_cols].to_csv(PREDICTIONS_FILENAME, index = False)