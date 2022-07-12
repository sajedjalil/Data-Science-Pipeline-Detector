import numpy as np
from numpy import sort
import pandas as pd
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import csv


def exportCSV(fileName, header, data):
    with open(fileName, 'w') as outputFile:
        writer = csv.writer(outputFile, dialect='excel')

        writer.writerow(header)

        for row in data:
            writer.writerow(np.array(row))


#Start
dataset_act = pd.read_csv('../input/act_train.csv', parse_dates=['date'])
dataset_ppl = pd.read_csv('../input/people.csv', parse_dates=['date'])
raw_data = pd.merge(dataset_act, dataset_ppl, on='people_id', how='inner')

dataset_test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])
raw_data_test = pd.merge(dataset_test, dataset_ppl, on='people_id', how='inner')

'''
raw_data.fillna('-999', inplace=True)

#Clear
y = raw_data['outcome'].values

output_act_id = raw_data_test['activity_id'].values

raw_data.drop(['activity_id', 'outcome', 'date_x', 'date_y'], axis=1, inplace=True)
raw_data_test.drop(['activity_id', 'date_x', 'date_y'], axis=1, inplace=True)

for col in raw_data:
    if(raw_data[col].dtype == np.object):
        raw_data[col] = raw_data[col].str.replace('type ', '')

raw_data['group_1'] = raw_data['group_1'].str.replace('group ', '')
raw_data['people_id'] = raw_data['people_id'].str.replace('ppl_', '')

for col in raw_data_test:
    if(raw_data_test[col].dtype == np.object):
        raw_data_test[col] = raw_data_test[col].str.replace('type ', '')

raw_data_test['group_1'] = raw_data_test['group_1'].str.replace('group ', '')
raw_data_test['people_id'] = raw_data_test['people_id'].str.replace('ppl_', '')
'''

raw_data.fillna('-999', inplace=True)
raw_data_test.fillna('-999', inplace=True)


#Clear
y = raw_data['outcome'].values

output_act_id = raw_data_test['activity_id'].values

raw_data.drop(['activity_id', 'date_x', 'date_y'], axis=1, inplace=True)
raw_data_test.drop(['activity_id', 'date_x', 'date_y'], axis=1, inplace=True)

raw_data_test['outcome'] = 0
for col in raw_data.columns:
    if(col != 'outcome' and col != 'people_id'):
        avgByGroup = raw_data.groupby(col).mean().reset_index()
        #outcomes = raw_data['outcome'].values

        temp = pd.merge(raw_data[[col, 'outcome']], avgByGroup, suffixes=('x_', ''), how='left', on=col, left_index=True)['outcome']
        temp_test = pd.merge(raw_data_test[[col, 'outcome']], avgByGroup, suffixes=('x_', ''), how='left', on=col, left_index=True)['outcome']
        
        #temp = ((temp*temp.shape[0])-outcomes)/(temp.shape[0]-1)

        temp.fillna(temp.mean())
        temp_test.fillna(temp_test.mean())

        raw_data[col] = temp.values
        raw_data_test[col] = temp_test.values

raw_data.drop(['people_id', 'outcome'], axis=1, inplace=True)
raw_data_test.drop(['people_id', 'outcome'], axis=1, inplace=True)

#Predict
#x = raw_data.values
#x_test = raw_data_test.values
x = raw_data[['group_1', 'char_38', 'char_7_y']].values
x_test = raw_data_test[['group_1', 'char_38', 'char_7_y']].values


model = xgboost.XGBClassifier()
model.fit(x, y)

#print (raw_data.columns)
#print (model.feature_importances_)


'''
selection = SelectFromModel(model, threshold=0.055, prefit=True)
x = selection.transform(x)
'''

model = xgboost.XGBClassifier()
model.fit(x, y)

kfold = cross_validation.KFold(n=len(x), n_folds=10, random_state=7)
cv_results = cross_validation.cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_results.mean(), cv_results.std() * 2))

#x_test = selection.transform(x_test)

pred = model.predict(x_test)

out = zip(output_act_id, pred)
exportCSV('output.csv', ['activity_id', 'outcome'], out)


