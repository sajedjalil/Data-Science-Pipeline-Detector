import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict



usecols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
 
nrows = 20       
df_train = pd.read_csv('../input/train_ver2.csv', usecols=usecols,nrows=nrows)
sample = pd.read_csv('../input/sample_submission.csv')

#print(df_train[['ncodpers', 'ind_ahor_fin_ult1'] ])
df_train = df_train.drop_duplicates(['ncodpers'], keep='last' )
print(df_train[['ncodpers', 'ind_ahor_fin_ult1'] ])

df_train.fillna(0, inplace=True)

models = {}
model_preds = {}
id_preds = defaultdict(list)
ids = df_train['ncodpers'].values

print(id_preds)
print(ids)
