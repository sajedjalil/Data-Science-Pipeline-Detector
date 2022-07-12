
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble

# columns to be used as features #
feature_cols = ["ind_empleado","pais_residencia","sexo","age", "ind_nuevo", "antiguedad", "nomprov", "segmento"]
dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1'] 


# In[2]:

Data_DIR = "F:/Demoonism/ML/Kaggle/Santander/"
Sample_DIR = "F:/Demoonism/ML/Kaggle/Santander/"
Test_DIR = "F:/Demoonism/ML/Kaggle/Santander/"

train_file = Data_DIR + "train_ver2.csv"
test_file = Test_DIR + "test_ver2.csv"
train_size = 13647309
nrows = 500000 # change this value to read more rows from train


# In[3]:

start_index = train_size - nrows	
for ind, col in enumerate(feature_cols):
    print(col)
    train = pd.read_csv(train_file, usecols=[col])
    test = pd.read_csv(test_file, usecols=[col])
    train.fillna(-99, inplace=True)
    test.fillna(-99, inplace=True)
    if train[col].dtype == "object":
        le = preprocessing.LabelEncoder()
        le.fit(list(train[col].values) + list(test[col].values))
        temp_train_X = le.transform(list(train[col].values)).reshape(-1,1)[start_index:,:]
        temp_test_X = le.transform(list(test[col].values)).reshape(-1,1)
    else:
        temp_train_X = np.array(train[col]).reshape(-1,1)[start_index:,:]
        temp_test_X = np.array(test[col]).reshape(-1,1)
    if ind == 0:
        train_X = temp_train_X.copy()
        test_X = temp_test_X.copy()
    else:
        train_X = np.hstack([train_X, temp_train_X])
        test_X = np.hstack([test_X, temp_test_X])
    print(train_X.shape, test_X.shape)
del train
del test


# In[4]:

train_y = pd.read_csv(train_file, usecols=['ncodpers']+target_cols, dtype=dtype_list)
last_instance_df = train_y.drop_duplicates('ncodpers', keep='last')
train_y = np.array(train_y.fillna(0)).astype('int')[start_index:,1:]
print(train_X.shape, train_y.shape)
print(test_X.shape)

print("Running Model..")
model = ensemble.RandomForestClassifier(n_estimators=70, max_depth=10, min_samples_leaf=10, n_jobs=4, random_state=2016)
model.fit(train_X, train_y)
del train_X, train_y


# In[5]:

test_id_temp = np.array(pd.read_csv(test_file, usecols=['ncodpers'])['ncodpers'])
TestbatchSize = 3 
batch = np.array_split(test_X, TestbatchSize)
test_id = np.array_split(test_id_temp, TestbatchSize)


# In[6]:

for minibatch, ids in zip(batch, test_id) :
    print("Predicting batch size ", len(test_id))
    preds = np.array(model.predict_proba(minibatch))[:,:,1].T
    
    print("Getting last instance dict..")
    last_instance_df = last_instance_df.fillna(0).astype('int')
    cust_dict = {}
    target_cols = np.array(target_cols)
    for ind, row in last_instance_df.iterrows():
        cust = row['ncodpers']
        used_products = set(target_cols[np.array(row[1:])==1])
        cust_dict[cust] = used_products
    #del last_instance_df
    
    print("Creating submission for this batch")
    preds = np.argsort(preds, axis=1)
    preds = np.fliplr(preds)

    final_preds = []
    for ind, pred in enumerate(preds):
        cust = ids[ind]
        top_products = target_cols[pred]
        used_products = cust_dict.get(cust,[])
        new_top_products = []
        for product in top_products:
            if product not in used_products:
                new_top_products.append(product)
            if len(new_top_products) == 7:
                break
        final_preds.append(" ".join(new_top_products))
    out_df = pd.DataFrame({'ncodpers':ids, 'added_products':final_preds})
    out_df.to_csv('batchPredict.csv', mode='a', index=False, header = False)

print("All batch done!")


