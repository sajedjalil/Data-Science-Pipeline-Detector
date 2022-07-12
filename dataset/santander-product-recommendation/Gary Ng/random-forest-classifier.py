import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import time
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


targetcols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
dtype_list = {'ind_cco_fin_ult1': 'float16',
              'ind_deme_fin_ult1': 'float16',
              'ind_aval_fin_ult1': 'float16',
              'ind_valo_fin_ult1': 'float16',
              'ind_reca_fin_ult1': 'float16',
              'ind_ctju_fin_ult1': 'float16',
              'ind_cder_fin_ult1': 'float16', 
              'ind_plan_fin_ult1': 'float16',
              'ind_fond_fin_ult1': 'float16', 
              'ind_hip_fin_ult1': 'float16',
              'ind_pres_fin_ult1': 'float16', 
              'ind_nomina_ult1': 'float16', 
              'ind_cno_fin_ult1': 'float16',
              'ncodpers': 'int64',
              'ind_ctpp_fin_ult1': 'float16',
              'ind_ahor_fin_ult1': 'float16',
              'ind_dela_fin_ult1': 'float16',
              'ind_ecue_fin_ult1': 'float16',
              'ind_nom_pens_ult1': 'float16',
              'ind_recibo_ult1': 'float16',
              'ind_deco_fin_ult1': 'float16',
              'ind_tjcr_fin_ult1': 'float16', 
              'ind_ctop_fin_ult1': 'float16',
              'ind_viv_fin_ult1': 'float16',
              'ind_ctma_fin_ult1': 'float16'}       


feature_cols = ["ind_empleado","pais_residencia","sexo",
                "age", "ind_nuevo", "antiguedad", "nomprov",
                "segmento"]

train_file = '../input/train_ver2.csv'
test_file = '../input/test_ver2.csv'
train_size = 13647309
nrows = 1000000
start_idx = train_size - nrows
for idx,col in enumerate(feature_cols):
    
    start_time = time.time()
    train = pd.read_csv(train_file,usecols=[col])
    test = pd.read_csv(test_file,usecols=[col])
    print(col)
    ### data preprocessing
    if col == 'age':
        train[col] = pd.to_numeric(train[col],errors='coerce')
        test[col] = pd.to_numeric(test[col],errors='coerce')
        
        train.loc[train.age < 18,"age"]  = train.loc[(train.age >= 18) & (train.age <= 30),"age"].mean(skipna=True)
        test.loc[test.age > 100,"age"] = test.loc[(test.age >= 30) & (test.age <= 100),"age"].mean(skipna=True)
        
        train['age'].fillna(train['age'].mean(),inplace=True)
        test['age'].fillna(test['age'].mean(),inplace=True)
        train['age'] = train['age'].astype(int)
        test['age']= test['age'].astype(int)
        
        
    elif col == 'ind_nuevo':
       train.loc[train[col].isnull(),col] = 1
       test.loc[test[col].isnull(),col] = 1
    elif col == 'antiguedad':
        train[col] = pd.to_numeric(train[col],errors='coerce')
        test[col] = pd.to_numeric(test[col],errors = 'coerce')
        train.loc[train[col].isnull(),col] = train[col].min()
        train.loc[train[col] < 0 , col] = 0 
        test.loc[test[col].isnull(),col] = test[col].min()
        test.loc[test[col] <0 ,col] = 0
    elif col =='nomprov':
        train[col].fillna('Unknown',inplace=True)
        test[col].fillna('Unknown',inplace=True)
    elif col =='segmento':
        train[col] = train[col].apply(lambda x:str(x).split('-')[0])
        test[col] = test[col].apply(lambda x:str(x).split('-')[0])
        train.loc[train[col].isnull(),col] = 'Unknown'
        test.loc[test[col].isnull(),col] = 'Unknown'
    else:
        train[col].fillna(-999,inplace=True)
        test[col].fillna(-999,inplace=True)
    ##### 
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].values) + list(test[col].values))
        temp_train = le.transform(list(train[col].values)).reshape(-1,1)[start_idx:,:]
        temp_test = le.transform(list(test[col].values)).reshape(-1,1)
    else:
        temp_train = np.array(train[col]).reshape(-1,1)[start_idx:,:]
        temp_test = np.array(test[col]).reshape(-1,1)
    if idx == 0:
        x_train = temp_train.copy()
        x_test = temp_test.copy()
    else:
        x_train = np.hstack([x_train,temp_train])
        x_test = np.hstack([x_test,temp_test])
    print(x_train.shape,x_test.shape)
    print('Time is %0.2f' %(time.time() - start_time))
    del train
    del test

y_train = pd.read_csv(train_file,usecols = targetcols,dtype=dtype_list)
last_instance = y_train.drop_duplicates(y_train,keep='last')
y_train = np.array(y_train.fillna(0)).astype('int')[start_idx:,1:]
print(x_train.shape,y_train.shape)



print('Running Model...')
clf = RandomForestClassifier(n_estimators=10,
                             max_depth=10,
                             n_jobs=-1,
                             random_state=42)

clf.fit(x_train,y_train)
del x_train
del y_train
print('Predicting....')
## [n_sample , n_class]
y_pred = np.array(clf.predict_proba(x_test))[:,:,1].T ## [n_class,n_sample]
del x_test

print('Getting the last instance dictionary ....')
last_instance.fillna(0,inplace=True)
recommendation_product = {}
targetcols = np.array(targetcols)
for idx,row_val in last_instance.iterrows():
    
    ids = row_val['ncodpers']
    used_product = set(targetcols[np.array(row_val[1:]) == 1])
    recommendation_product[ids] = used_product
del last_instance

print('Submission ....')
## [n_class , n_sample]
pred = np.argsort(y_pred,axis=1) ## sort probability by axis 1 and return index
#print(pred)
pred = np.fliplr(pred) 
test_ids = np.array(pd.read_csv(test_file,usecols=['ncodpers'])['ncodpers'])
final_preds = []
for idx,predicted in enumerate(pred):
    ids = test_ids[idx]
    top_product = targetcols[predicted]
    used_product = recommendation_product.get(ids,[])
    new_top_product = []
    for product in top_product:
        if product not in used_product:
            new_top_product.append(product)
        if len(new_top_product) == 7:
            break
    final_preds.append(' '.join(new_top_product))
result = pd.DataFrame({'ncodpers':test_ids,'added_products':final_preds})
result.to_csv('submission.csv',index=False)
print('Finish. Time is %0.2f' %(time.time() - start_time))



