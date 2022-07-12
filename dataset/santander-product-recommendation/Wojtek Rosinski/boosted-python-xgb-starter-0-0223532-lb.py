# Work in progress. I will update when I incorporate more features.
# I've based my processing on scripts by SRK. Thanks!


""" __author__ : Wrosinski """


import numpy as np
import pandas as pd
import time
import sys

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score, f1_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import csr_matrix, hstack

import xgboost as xgb
from xgboost import XGBClassifier


numerical_cols = ['age', 'antiguedad', 'renta']

feature_cols = ['ind_actividad_cliente', 
                "ind_empleado", "pais_residencia" ,"sexo" , "ind_nuevo", 
                 "nomprov", "segmento", 'indrel', 'tiprel_1mes', 'indresi', 'indext',
               'conyuemp', 'indfall', 'canal_entrada']

dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1'] 


data_path = "../input/"
train_file = data_path + "train_ver2.csv"
test_file = data_path + "test_ver2.csv"
train_size = 13647309
nrows = 1000000 # change this value to read more rows from train
start_index = train_size - nrows




for ind, col in enumerate(feature_cols):
    print(col)
    train = pd.read_csv(train_file, usecols=[col])
    test = pd.read_csv(test_file, usecols=[col])
    train.fillna(-99, inplace=True)
    test.fillna(-99, inplace=True)
    if train[col].dtype == "object":
        le = LabelEncoder()
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
print ("Categorical features loaded.")


for ind, col in enumerate(numerical_cols):
    print(col)
    train = pd.read_csv(train_file, usecols=[col])
    test = pd.read_csv(test_file, usecols=[col])
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    if train[col].dtype == "object":

        temp_train_X = pd.to_numeric(train[col], 'coerce').fillna(-1).astype('float64').reshape(-1,1)[start_index:,:]
        temp_test_X = pd.to_numeric(test[col], 'coerce').fillna(-1).astype('float64').reshape(-1,1)
    else:
        temp_train_X = np.array(pd.to_numeric(train[col], 'coerce').fillna(0).astype('float64')).reshape(-1,1)[start_index:,:]
        temp_test_X = np.array(pd.to_numeric(test[col], 'coerce').fillna(0).astype('float64')).reshape(-1,1)
    if ind == 0:
        train_X_f = temp_train_X.copy()
        test_X_f = temp_test_X.copy()
    else:
        train_X_f = np.hstack([train_X_f, temp_train_X])
        test_X_f = np.hstack([test_X_f, temp_test_X])
        
print ("Numeric features loaded.")


#Dense encoding:
full_train = np.hstack([train_X, train_X_f])

full_test = np.hstack([test_X, test_X_f])

#Sparse encoding (OneHot):
scaler = RobustScaler()
temp = csr_matrix(scaler.fit_transform(train_X_f))

enc = OneHotEncoder()
OH = enc.fit_transform(train_X)
OH_csr = csr_matrix(OH)

sparse_train = hstack([temp, OH_csr], format = "csr")


#Read target data:
train_y = pd.read_csv(train_file, usecols=['ncodpers']+target_cols, dtype=dtype_list)
last_instance_df = train_y.drop_duplicates('ncodpers', keep='last')
train_y = np.array(train_y.fillna(0)).astype('int')[start_index:,1:]
print(train_y.shape)


#Train/validation split, optional.
X_train2, X_val2, y_train2, y_val2 = train_test_split(full_train, train_y, test_size = 0.2, random_state = 669)
nb_classes = train_y.shape[1]
print (sys.getsizeof(full_train))


#Model
xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.5,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'max_depth': 10,
    'min_child_weight': 100,
    'booster': 'gbtree',
    'eval_metric': 'logloss'
    }


preds = []
preds_test = []

folds = 5
kf = KFold(full_train.shape[0], n_folds = folds)

for ind, col in enumerate(train_y.T):
    
    fold_preds = []
    fold_test = []
    
    for i, (train_index, test_index) in enumerate(kf):
        
        X_train, X_val = full_train[train_index], full_train[test_index]
        y_train, y_val = col[train_index], col[test_index]

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        d_test = xgb.DMatrix(full_test)

        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        clf = xgb.train(xgb_params,
                    d_train,
                    300,
                    watchlist,
                    early_stopping_rounds = 50, verbose_eval = False)

        probs = clf.predict(d_valid)
        probs_test = clf.predict(d_test)
        
        fold_preds.append(probs)
        fold_test.append(probs_test)
        
    preds.append(fold_preds)
    preds_test.append(fold_test)

preds_a = np.asarray(preds)
preds_at = np.asarray(preds_test)

means = []
for i in preds_a:
    j = np.mean(i, axis = 0)
    means.append(j)
    
means_t = []
for i in preds_at:
    j = np.mean(i, axis = 0)
    means_t.append(j)

    
means_b = np.asarray(means).T
means_tb = np.asarray(means_t).T

ROC = roc_auc_score(y_val2, means_b)

means_b[means_b >= 0.5] = 1
means_b[means_b < 0.5] = 0
F1 = f1_score(y_val2, means_b, average = "macro")

print ('\n', "F1 score: ", F1, "ROC AUC score: ", ROC)


#Making proper predictions for test data.
preds = means_tb
print ("Test set predictions done.")


print("Getting last instance dict..")
last_instance_df = last_instance_df.fillna(0).astype('int')
cust_dict = {}
target_cols = np.array(target_cols)
for ind, row in last_instance_df.iterrows():
    cust = row['ncodpers']
    used_products = set(target_cols[np.array(row[1:])==1])
    cust_dict[cust] = used_products


print("Creating submission..")
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)
test_id = np.array(pd.read_csv(test_file, usecols=['ncodpers'])['ncodpers'])
final_preds = []
for ind, pred in enumerate(preds):
    cust = test_id[ind]
    top_products = target_cols[pred]
    used_products = cust_dict.get(cust,[])
    new_top_products = []
    for product in top_products:
        if product not in used_products:
            new_top_products.append(product)
        if len(new_top_products) == 7:
            break
    final_preds.append(" ".join(new_top_products))

len(final_preds[0])
len(final_preds)
final_preds[0]

out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv('XGBoost_try_13.11.csv', index=False)