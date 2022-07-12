
''' 
This code is the modification of the code shared by anokas with a better LB score.
'''

# Any results you write to the current directory are saved as output.

import pandas as pd, numpy as np, datetime, gc, re

print(datetime.datetime.now())

usecols = ['ncodpers', 'fecha_dato', 
           'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
            'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

df_train = pd.read_csv('../input/train_ver2.csv', usecols=usecols, parse_dates=['fecha_dato'], infer_datetime_format=True)
# df_train = pd.read_csv("D:\\Kaggle\\Santander Products Recommendation\\Data\\train_ver2.csv", usecols=usecols
#                       , parse_dates=['fecha_dato'] , infer_datetime_format=True)

dftr_prev = df_train[df_train['fecha_dato'].isin(['28-04-2016'])]
dftr_curr = df_train[df_train['fecha_dato'].isin(['28-05-2016'])]

dftr_prev.drop(['fecha_dato'], axis=1, inplace = True)
dftr_curr.drop(['fecha_dato'], axis=1, inplace = True)

dfm = pd.merge(dftr_curr,dftr_prev, how='inner', on=['ncodpers'], suffixes=('', '_prev'))

prevcols = [col for col in dfm.columns if '_ult1_prev' in col]
currcols = [col for col in dfm.columns if '_ult1' in col and '_ult1_prev' not in col]

for cols in currcols:
    print(cols)
    dfm[cols] = dfm[cols].astype('category')

for cols in prevcols:
    print(cols)
    dfm[cols] = dfm[cols].astype('category')

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

dfm.fillna(0, inplace=True)

models = {}
model_preds = {}
id_preds = defaultdict(list)

model_preds_train = {}
id_preds_train = defaultdict(list)

ids = dfm['ncodpers'].values
testids = dftr_curr['ncodpers'].values

prevcols = [col for col in dfm.columns if '_ult1_prev' in col]
currcols = [col for col in dfm.columns if '_ult1' in col and '_ult1_prev' not in col]

for c in dfm.columns:
    if c != 'ncodpers' and '_ult1_prev' not in c:
        print(c)
        y_train = dfm[c]
        x_train = dfm.drop(currcols,1,inplace=False).drop(['ncodpers'],1,inplace=False)
        x_test = dftr_curr.drop(['ncodpers'],1,inplace=False)
        
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        p_train = clf.predict_proba(x_train)[:,1]
        p_test = clf.predict_proba(x_test)[:,1]
        
        models[c] = clf
        model_preds[c] = p_test
        for id, p in zip(testids, p_test):
            id_preds[id].append(p)
            
        model_preds_train[c] = p_train
        for id, p in zip(ids, p_train):
            id_preds_train[id].append(p)
            
        print(roc_auc_score(y_train, p_train))

already_active = {}

dftrcurr = dftr_curr

for row in dftrcurr.values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(dftrcurr.columns[1:], row) if c[1] > 0]
    already_active[id] = active

test_preds = {}
for id, p in id_preds.items():
    preds = [i[0] for i in sorted([i for i in zip(dftrcurr.columns[1:], p) if i[0] not in already_active[id]], key=lambda i:i [1], reverse=True)[:7]]
    test_preds[id] = preds

sample = pd.read_csv('../input/sample_submission.csv')
# sample = pd.read_csv('D:\\Kaggle\\Santander Products Recommendation\\Data\\sample_submission.csv')

test_preds_prods = []
for row in sample.values:
    id = row[0]
    p = test_preds[id]
    test_preds_prods.append(' '.join(p))

sample['added_products'] = test_preds_prods

sample.to_csv('modified_collab_sub_by_KB.csv', index=False)
# sample.to_csv('D:\\Kaggle\\Santander Products Recommendation\\Submission\\S_NLogitModels_ToPred_sub_check.csv', index=False)
