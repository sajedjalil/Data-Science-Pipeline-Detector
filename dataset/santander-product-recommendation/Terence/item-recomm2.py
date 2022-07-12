import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from collections import defaultdict
from sklearn import ensemble
import matplotlib.pyplot as plt

usecols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
       
#cd /Volumes/My Passport Ultra/Data/kaggle/santanderProductRecomm/input   
feaList = ['renta', 'age']
data = pd.read_csv('../input/train_ver2.csv',usecols=usecols + feaList)
print("finish reading csv")

print("data length is " + str(data.__len__()))
data = data.drop_duplicates(['ncodpers'], keep = 'last')
print("data length is " + str(data.__len__()))

data.loc[data['renta'].isnull(),'renta'] = 0.0
data['age'] = data['age'].str.strip()
data.loc[ data['age'] =='NA','age'] = 0
data['age'] = data['age'].astype('float64', raise_on_error = False)
data.loc[ data['age'].isnull(),'age']=0

print("finish feature transformations S1")

def renta_out(x):
    result = 1.0
    if x <= 200000:
        result = x/200000
    else:
        result = 1.0   
    return result

def age_p(x):
    result = 1
    if x < 20:
        result = 20
    elif x<80:
        result = x
    else:
        result = 80
    return result

data['renta'] = data.renta.apply(renta_out)
data['age'] = data.age.apply(age_p)

print("finish renta and age")
#feaList = ['renta', 'age']
#data_sub = data.ix[:, feaList + usecols]
#df_train = data.ix[:, usecols]
sample = pd.read_csv('../input/sample_submission.csv')

print("finish sample reading")

#df_train = df_train.drop_duplicates(['ncodpers'], keep='last')
#data_sub = data.drop_duplicates(['ncodpers'], keep='last')
#print(data_sub.__len__(), df_train.__len__())

low_product = ['ind_aval_fin_ult1','ind_ahor_fin_ult1']

data.fillna(0, inplace = True)
# try other fill methods
#df_train.fillna(-1, inplace = True)
#data_sub.fillna(-1, inplace = True)
models = {}
model_preds = {}
id_preds = defaultdict(list)
#ids = df_train['ncodpers'].values
#ids = data_sub['ncodpers'].values
ids = data.ncodpers.values
newList = ['ncodpers'] + feaList
print("before training and predictions")
for c in usecols:
    if c != 'ncodpers':
    #and c in ['ind_cco_fin_ult1']:
    #and c in ['ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']:
        print(c)
        #y_train = df_train[c]
        #x_train = df_train.drop([c, 'ncodpers'], 1)
        #y_train = data_sub[c]
        y_train = data[c]
        x_train = data.drop([c,'ncodpers'], 1)
    
        clf = ensemble.ExtraTreesClassifier(n_estimators=50, n_jobs=-1,max_depth=8, min_samples_split=10, verbose=1)
        clf.fit(x_train, y_train)
        p_train1 = clf.predict_proba(x_train)[:,1]
        aucScore1 = roc_auc_score(y_train, p_train1)
        print('Extra Tree value is ' + str(c) + ' \t ' + str(roc_auc_score(y_train, p_train1)))

        clf1 = LogisticRegression()
        clf1.fit(x_train, y_train)
        p_train2 = clf1.predict_proba(x_train)[:,1]
        print('Logistic value is ' + str(c) + ' \t ' + str(roc_auc_score(y_train, p_train2)))
        aucScore2 = roc_auc_score(y_train, p_train2)

        #clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=3, random_state=0, subsample = 0.7, max_features = 'auto')
        #clf2.fit(x_train, y_train)
        #p_train3 = clf2.predict_proba(x_train)[:,1]
        #print('GBDT value is ' + str(c) + ' \t ' + str(roc_auc_score(y_train, p_train3)))
        
        #p_train = p_train1*0.9+p_train2*0.1 
        p_train3 = p_train1*0.9 + p_train2*0.2
        aucScore3 = roc_auc_score(y_train, p_train3)
        if (aucScore3 > 0.7):
            p_train = p_train3
        elif aucScore2 > aucScore1:
            p_train = p_train2
        else:
            p_train = p_train1
        for id, p in zip(ids, p_train):
            id_preds[id].append(p)
        #print('value is ' + str(c) + ' \t ' + str(y_train.value_counts()))
        
        print('value is ' + str(c) + ' \t ' + str(roc_auc_score(y_train, p_train)))

print("active set module") 
#quit()

already_active = {}
for row in data.ix[:, usecols].values:
    row = list(row)
    id = row.pop(0)
    active = [c[0] for c in zip(usecols[1:], row) if c[1] > 0]
    already_active[id] = active
    
print("train preds module")     

train_preds = {}
for id, p in id_preds.items():
    # Here be dragons
    preds = [i[0] for i in sorted([i for i in zip(usecols[1:], p) if i[0] not in already_active[id] and i[0] not in low_product], key=lambda i:i [1], reverse=True)[:7]]
    train_preds[id] = preds

print("test preds module")     

test_preds = []
for row in sample.values:
    id = row[0]
    p = train_preds[id]
    test_preds.append(' '.join(p))

sample['added_products'] = test_preds
sample.to_csv('recommItems.csv', index=False)





