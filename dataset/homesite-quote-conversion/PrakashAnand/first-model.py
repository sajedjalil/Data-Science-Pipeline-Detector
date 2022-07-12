import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.shape)
print(test.shape)

y = train['QuoteConversion_Flag']
train.drop(train.columns[2], axis=1, inplace=True)

train_length = len(train)

# Now we combine test and train for preprocessing
complete_data = train.append(test)
print(complete_data.shape)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, auc, roc_curve
# Things to try as extract day and month part from the quotes date, create bins for them
# Check the unique values for all the columns

def pre_processing(data):
    # Extract month and date part
    data['Date'] = pd.to_datetime(pd.Series(data['Original_Quote_Date']))
    # Now drop this column from the data frame
    data = data.drop('Original_Quote_Date', axis=1)
    data['Month'] = data['Date'].apply(lambda x: int(str(x)[5:7]))
    data['weekday'] = data['Date'].dt.dayofweek
    data.drop('Date', axis=1, inplace=True)
    le = LabelEncoder()
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
    return data



complete_data = pre_processing(complete_data)
print('pre_processing done.....!')
print(complete_data.dtypes)
train_data = complete_data.iloc[0:train_length, ]
test_data = complete_data.iloc[train_length:, ]
# print(train_data.shape)
# print(test_data.shape)
# X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2)

#clf = xgb.XGBClassifier(n_estimators=30,
#                        nthread=-1,
#                        max_depth=10,
#                        learning_rate=0.025,
#                        silent=True,
#                        subsample=0.8,
#                        colsample_bytree=0.8)
                        
#xgb_model = clf.fit(X_train, y_train, eval_metric="auc")

param = {}
param = {'bst:max_depth':10, 'objective':'binary:logistic'}
param['nthread'] = -1
param['eval_metric'] = ['auc']
param['learning_rate'] = 0.025
param['silent'] = True
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8

# param['scale_pos_weight'] = sum_wneg/sum_wpos

num_round = 10

dtrain = xgb.DMatrix(train_data, label=y, missing=np.nan)
dtest = xgb.DMatrix(test_data, missing=np.nan)
evallist = [(dtest, 'eval'), (dtrain, 'train')]

gbm = xgb.train(param, dtrain, num_round)
prediction_1 = gbm.predict(dtest)

# prediction = [0 if x<0.5 else 1 for x in prediction_1]

# print(confusion_matrix(y_test, prediction))
# print(roc_auc_score(y_test, prediction_1))
# print(accuracy_score(y_test, prediction))

sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = prediction_1
sample.to_csv('xgb_benchmark.csv', index=False)
