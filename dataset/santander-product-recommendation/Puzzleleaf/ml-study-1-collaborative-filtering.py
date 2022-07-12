import pandas as pd, numpy as np, datetime, gc, re

print(datetime.datetime.now())

usecols = ['ncodpers', 'fecha_dato', 
           'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
            'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

df_train = pd.read_csv('../input/train_ver2.csv', usecols=usecols, parse_dates=['fecha_dato'], infer_datetime_format=True)

df_train_arr = [df_train[df_train['fecha_dato'].isin(['28-01-2016'])], df_train[df_train['fecha_dato'].isin(['28-02-2016'])], df_train[df_train['fecha_dato'].isin(['28-03-2016'])], df_train[df_train['fecha_dato'].isin(['28-04-2016'])], df_train[df_train['fecha_dato'].isin(['28-05-2016'])]]

for train_item in df_train_arr:
    train_item.drop(['fecha_dato'], axis=1, inplace = True)
    train_item.fillna(0, inplace=True)

df_train_filtered_arr = [df_train_arr[0].iloc[:, 1:25], df_train_arr[1].iloc[:, 1:25], df_train_arr[2].iloc[:, 1:25], df_train_arr[3].iloc[:, 1:25], df_train_arr[4].iloc[:, 1:25]]

# 데이터 전처리 완료
models = {}

model_preds = {}
model_preds_train = {}


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

id_preds = defaultdict(list)
id_preds_train = defaultdict(list)

clf = LogisticRegression()
ids = df_train_arr[4]['ncodpers'].values

for idx, columns in enumerate(df_train_arr):
    for c in columns:
        if c != 'ncodpers':
            y_train = df_train_arr[idx][c]
            x_train = df_train_arr[idx]
            clf.fit(x_train, y_train)
            if idx == 4:
                p_train = clf.predict_proba(x_train)[:,1]
                models[c] = clf
                model_preds[c] = p_train
                for id, p in zip(ids, p_train):
                    id_preds[id].append(p)
                print(roc_auc_score(y_train, p_train))
  
# 고객들이 마지막에 보유했던 상품 체크


already_active = {}
dftrcurr = df_train.drop_duplicates(['ncodpers'], keep='last')
dftrcurr.drop(['fecha_dato'], axis=1, inplace = True)
dftrcurr.fillna(0, inplace=True)

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

test_preds_prods = []

for row in sample.values:
    id = row[0]
    p = test_preds[id]
    test_preds_prods.append(' '.join(p))

sample['added_products'] = test_preds_prods

sample.to_csv('last.csv', index=False)