from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb


test = pd.read_csv('input/stage1_sample_submission.csv')

train = pd.read_csv('../input/stage1_labels.csv') 
trainp = train.copy()
trainp['Id'] = trainp['Id'].map(lambda x: x.split('_')[1])
piv = pd.pivot_table(trainp, columns='Id', values='Probability', aggfunc='mean', fill_value=0)
d = pd.DataFrame.to_dict(piv)

d['Zone9']['Probability'] = 0.05

y = train['Probability'].values
pid = test['Id'].values

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['len'] = df_all['Id'].map(len)

for i in range(38):
    df_all['c'+str(i)] = df_all['Id'].map(lambda x: str(x[i]))
df_all['c38'] = df_all['Id'].map(lambda x: str(x[i]) if len(x)==39 else '')
df_all = df_all.drop(['Id','Probability','len'], axis=1)

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        df_all[c] = lbl.fit_transform(df_all[c].values)
        #print(c, len(df_all[c].unique()))

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

params = {
    'eta': 0.02,
    'max_depth': 5,
    'objective': 'reg:linear',
    'eval_metric': 'logloss',
    'seed': 12,
    'silent': True
}

fold = 5
for i in range(fold):
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=50)
    if i != 0:
        pred += model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
pred /= fold
submission = pd.DataFrame(pred, columns=['Probability'])
submission['Id'] = pid
submission['Probability'] += submission['Id'].map(lambda x: d[str(x).split('_')[1]]['Probability'])*3
submission['Probability'] /= 4
submission.to_csv('submission_xgb.csv', index=False)