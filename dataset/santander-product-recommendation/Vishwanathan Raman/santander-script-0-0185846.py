# This script has been derived from the script provided by anokas
# Here is the link to the same. https://www.kaggle.com/anokas/santander-product-recommendation/collaborative-filtering-btb-lb-0-01691/comments#142627
# There have been quite a few changes made to the script and has been documented in the code
# Additional features e.g. age, sexo,pais_residencia have been used
# Empty values of age have been replaced by analysing the data.
# This script will help you climb a few place above the script provied by anokas
# Note : Run the script in a local machine or elsewhere, It gets killed in the kaggle environment
# Happy Coding and Learning


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

df_train_orig  = pd.read_csv('../input/train_ver2.csv')
sample = pd.read_csv('../input/sample_submission.csv')

useCols2 = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1',
       'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
       'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
       'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1',
       'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
       'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1',
       'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
       'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1',
       'ind_recibo_ult1']
usecols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

# Convert the date object to date
df_train_orig['fecha_dato']=pd.to_datetime(df_train_orig['fecha_dato'])
df_train_orig['fecha_alta']=pd.to_datetime(df_train_orig['fecha_alta'])
df_train_orig['ult_fec_cli_1t']=pd.to_datetime(df_train_orig['ult_fec_cli_1t'])

# Age Manipulations
# The task is take the max Age for each of the ncodpers and use them instead
df_train_orig['age']=df_train_orig['age'].str.strip()
df_train_orig.loc[df_train_orig['age']=='NA','age'] = 0
df_train_orig['age'] = df_train_orig['age'].astype('float64', raise_on_error = False)
df_train_orig.loc[df_train_orig['age'].isnull(),'age']=0
df_train_aggOnAge = pd.DataFrame(df_train_orig[['ncodpers','age']].groupby('ncodpers').max().reset_index())

# Gross Income Manipulations
# The task is take the max Income for each of the ncodpers and use them instead
df_train_orig.loc[df_train_orig['renta'].isnull(),'renta']=0
df_train_aggOnrenta = pd.DataFrame(df_train_orig[['ncodpers','renta']].groupby('ncodpers').max().reset_index())

# Finding the intersection between sample and train. Retaining only those that are in sample
uq_ncodpers = np.intersect1d(sample['ncodpers'].unique(),df_train_orig['ncodpers'].unique())
df_train_orig_red = df_train_orig[df_train_orig['ncodpers'].isin(uq_ncodpers)]


df_train = df_train_orig_red.drop_duplicates(['ncodpers'], keep='last')

# Replacing the Age where the last record does not have it
dfo = df_train.set_index(['ncodpers'])['age']
df1 = df_train_aggOnAge.set_index(['ncodpers'])['age']
dfo.update(df1)
df_train.loc[:,'age'] = dfo.values

# There are only five missing values in sexo. Replacing the missing with V
df_train.loc[df_train['sexo'].isnull(),'sexo']='V'

# Grouping age residence and sex and finding the median to replace the missing values.
df_train_aggOnAgeByResiSexo = pd.DataFrame(df_train[['pais_residencia','sexo','age']].groupby(['pais_residencia','sexo']).median().reset_index())
df_train_aggOnAgeByResiSexo.columns = ['pais_residencia', 'sexo', 'median_age']
df_train = pd.merge(df_train,df_train_aggOnAgeByResiSexo,on=['pais_residencia','sexo'])
df_train.loc[df_train['age']==0,'age']=df_train.loc[df_train['age']==0,'median_age']
df_train.drop('median_age',1,inplace=True)

df_train = pd.concat([pd.get_dummies(df_train['sexo']),df_train],axis=1)
df_train = pd.concat([pd.get_dummies(df_train['pais_residencia']),df_train],axis=1)

df_train.drop(['fecha_dato','ind_empleado','pais_residencia','sexo','fecha_alta','ind_nuevo','antiguedad','indrel','ult_fec_cli_1t','indrel_1mes','tiprel_1mes','indresi','indext','conyuemp','canal_entrada','indfall','tipodom','cod_prov','nomprov','ind_actividad_cliente','renta','segmento'],1,inplace=True)

models = {}
model_preds = {}
id_preds = defaultdict(list)
ids = df_train['ncodpers'].values
#for c in df_train.columns:
for c in useCols2:
    if c != 'ncodpers':
        print(c)
        y_train = df_train[c]
        x_train = df_train.drop([c, 'ncodpers'], 1)
        
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        p_train = clf.predict_proba(x_train)[:,1]
        
        models[c] = clf
        model_preds[c] = p_train
        for id, p in zip(ids, p_train):
            id_preds[id].append(p)
            
        print(roc_auc_score(y_train, p_train))

df_train = df_train.loc[:,usecols]

already_active = {}
for row in df_train.values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(df_train.columns[1:], row) if c[1] > 0]
    already_active[id] = active
    
train_preds = {}
for id, p in id_preds.items():
    # Here be dragons
    preds = [i[0] for i in sorted([i for i in zip(df_train.columns[1:], p) if i[0] not in already_active[id] and i[1]>0.0], key=lambda i:i [1], reverse=True)[:7]]
    train_preds[id] = preds
    
test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))

sample['added_products'] = test_preds
sample.to_csv('collab_sub.csv', index=False)
