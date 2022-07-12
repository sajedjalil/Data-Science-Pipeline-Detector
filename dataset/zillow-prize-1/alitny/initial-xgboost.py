import numpy as np
import pandas as pd
import xgboost as xgb
import gc

prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv('../input/train_2016_v2.csv')
test = pd.read_csv('../input/sample_submission.csv')
test['parcelid'] = test['ParcelId']

from sklearn.preprocessing import LabelEncoder

for c in prop.columns:
    prop[c] = prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))
        
merged = train.merge(prop, how='left', on='parcelid')
x_train = merged.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = merged['logerror']

test_merge = test.merge(prop, how='left', on='parcelid')
p_id = prop['parcelid']
x_test = prop.drop(['parcelid'], axis=1)

del merged, prop, train, test; gc.collect()

params = {
    'eval_metric':'mae'
}
dtrain = xgb.DMatrix(x_train, label=y_train)
print('Start CV')
cv_result = xgb.cv(params, dtrain, nfold=5, num_boost_round=305, early_stopping_rounds=50, verbose_eval=10, show_stdv=False)
print(cv_result)
print(len(cv_result))
model = xgb.train(dict(params), dtrain, num_boost_round=len(cv_result), evals=[(dtrain, 'eval')])
print('Finished fitting')
del dtrain, x_train, y_train, params ; gc.collect()

dtest = xgb.DMatrix(x_test)
pred = model.predict(dtest)
del model, dtest, x_test ; gc.collect()
print('Finished predicting')

y_pred = []
for i, predict in enumerate(pred):
    y_pred.append(str(round(predict, 4)))
y_pred = np.array(y_pred)
del pred ; gc.collect()

output = pd.DataFrame({'ParcelId': p_id, 
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
        
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv('test.csv', index=False)
