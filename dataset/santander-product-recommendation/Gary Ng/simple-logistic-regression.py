'''
reference:
https://www.kaggle.com/anokas/santander-product-recommendation/collaborative-filtering-btb-lb-0-01691/code

'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

targetcols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

usecols = ['cod_prov', 'ind_actividad_cliente', 'ncodpers',
           'renta', 'indresi', 'fecha_dato', 'ult_fec_cli_1t',
           'antiguedad', 'indrel', 'nomprov', 'indrel_1mes', 
           'sexo', 'conyuemp', 'canal_entrada', 'ind_nuevo',
           'pais_residencia', 'age', 'segmento', 'tiprel_1mes',
           'indfall', 'fecha_alta', 'indext', 'ind_empleado',
           'tipodom']

limit_row = 6000000
train = pd.read_csv('../input/train_ver2.csv',usecols=targetcols)
                   # nrows = limit_row)
sample = pd.read_csv('../input/sample_submission.csv')
train = train.drop_duplicates(['ncodpers'],keep='last')
train.fillna(0,inplace=True)
model_pred = {}
ids = train.ncodpers.values
pred = defaultdict(list)
for col in train.columns:
    if col != 'ncodpers':
        y_train = train[col]
        x_train = train.drop(['ncodpers',col],axis=1)
        clf = LogisticRegression()
        clf.fit(x_train,y_train)
        y_pred = clf.predict_proba(x_train)[:,1]
        model_pred[col] = y_pred
        
        for id,y_hat in zip(ids,y_pred):
            pred[id].append(y_hat)

        print('ROC Socre : %f' %(roc_auc_score(y_train,y_pred)))
#print(pred)
active_ = {}
for val in train.values:
    val = list(val)
    id = val.pop(0) ## pop ncodpers (customer id)
    ## active column
    active  = [c[0] for c in zip(train.columns[1:],val) if c[1] > 0]
    active_[id] = active
    
train_preds = {}
for id,val in pred.items():
    
    #preds = [i[0] for i in sorted([for i in zip(train.columns[1,],val) if i[0] not in active_[id]],key = lambda i:i[1],reverse=True)]
    preds = [i[0] for i in sorted([i for i in zip(train.columns[1:],val) if i[0] not in active_[id]], key=lambda i:i [1], reverse=True)[:7]]
    train_preds[id] = preds
    
test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))

sample['added_products'] = test_preds
sample.to_csv('recommendation.csv',index=False)
