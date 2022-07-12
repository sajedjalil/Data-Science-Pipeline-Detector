# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time,datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
def reduce_mem_usage(transaction_path,identity_path, verbose=True):
     # merge the two datasets
    t = transaction_path
    i = identity_path
    Transaction_dataset = pd.read_csv("%s"%t).drop_duplicates(['TransactionID'])
    Identity_dataset = pd.read_csv("%s"%i).drop_duplicates(['TransactionID'])
    df= pd.merge(Transaction_dataset,Identity_dataset,on='TransactionID',how='outer')
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


train = reduce_mem_usage('../input/ieee-fraud-detection/train_transaction.csv','../input/ieee-fraud-detection/train_identity.csv')
test = reduce_mem_usage('../input/ieee-fraud-detection/test_transaction.csv','../input/ieee-fraud-detection/test_identity.csv')
########################drop the V columns which have different null frequency between train and test###########################
local = train.drop(['isFraud'],axis = 1)
null_frq_train = local.isnull().sum()/len(local)
null_frq_test = test.isnull().sum()/len(test)
diff = null_frq_train - null_frq_test

null_freq_to_drop = []
for i,v in diff.items():
    if abs(v) > .1:
        null_freq_to_drop.append(i)
null_freq_to_drop = [v for v in null_freq_to_drop if v[0] is 'V']   

#################################################################################################################
def feature_transform(merged_data):
    ##Feature Engineering##
    # merge the two datasets 
    #*********************** handle the transaction time(TranscationDT)**********************
    def handle_time(timestamp):
        initial = 1329973999
        timeArray = time.localtime(initial+timestamp)
        otherStyleTime = int(time.strftime("%H%M%S", timeArray))  # 23:40:00
        return otherStyleTime
    
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    merged_data['TransactionDT'] = merged_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
    merged_data['month'] = merged_data['TransactionDT'].apply(lambda x: (x.month))
    merged_data['day'] = merged_data['TransactionDT'].apply(lambda x: (x.day))
    merged_data['hour'] = merged_data['TransactionDT'].apply(lambda x: (x.hour))
    merged_data['weekday'] = merged_data['TransactionDT'].apply(lambda x: (x.weekday()))
#     merged_data['TransactionDT'] = merged_data['TransactionDT'].map(lambda x:handle_time(x))
    merged_data = merged_data.drop(['TransactionDT'],axis = 1)
    #*************************columns to drop************************************************
    columns_to_drop = []
    #***************************** drop the column which have different distribution between train and test dataset#####################################
    diff_distribution_to_drop = ['V4','V7','V12','V79','V53','D13','D9','id_01','id_31','id_13','V75','C12','id_34','id_30','D14']
   #************************ handle the features with too many nulls***************************************
    def get_too_many_null_attr(data):
        many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
        return many_null_cols
    temp = get_too_many_null_attr(merged_data)
    many_null_attr_to_drop = temp+['D7','id_18']

    #*************************** too many repeated value*******************
   
    #**********************cleaning the data*************************************************************
    columns_to_drop = list(set(null_freq_to_drop).union(set(diff_distribution_to_drop)).union(set(many_null_attr_to_drop)))
    merged_data = merged_data.drop(columns_to_drop,axis = 1)
#     ['V25', 'V92', 'id_27', 'V87', 'V68', 'V93', 'V13', 'V38', 'V67', 'V53', 'V49', 'V80', 'V23', 'V31', 'V84', 'id_34',
#      'V44', 'V39', 'V74', 'V83', 'V20', 'V55', 'V41', 'id_25', 'id_26', 'V76', 'V16', 'V66', 'V65', 'V37', 'V46', 'V11', 'V82', 
#      'V9', 'V15', 'D13', 'V5', 'V4', 'V8', 'V63', 'V72', 'V56', 'V69', 'V59', 'V14', 'V91', 'V51', 'V36', 'dist2', 'V45', 
#      'id_18', 'D9', 'V58', 'D14', 'V34', 'V86', 'V12', 'V64', 'V24', 'V62', 'id_24', 'V3',  'V27', 'V78', 'V52', 'V79', 'V75', 
#      'V50', 'V43', 'V28', 'id_13', 'id_08', 'V6', 'id_30', 'V21', 'V54', 'V22', 'V89', 'V70', 'V71', 'V77', 'id_31', 'V40', 'V94', 
#      'V10', 'D7', 'id_22', 'V48', 'V85', 'id_01', 'V26', 'V18', 'V42', 'V60', 'V88', 'V19', 'id_07', 'V29', 'V35', 'V61', 'V32', 
#      'V1', 'V47', 'C12', 'V73', 'V7', 'V81', 'V33', 'V30', 'V17', 'V2', 'V90', 'id_23', 'id_21', 'V57']
#     print(columns_to_drop)
#     print([x for x in merged_data.columns if x[0] == 'V'])
    #*************************handle the Transaction amount***********************************
    #logged TransactionAmt 
    merged_data["Amt_log"] = np.log(merged_data['TransactionAmt'])
    merged_data['Amt_decimal'] = ((merged_data['TransactionAmt'] - merged_data['TransactionAmt'].astype(int)) * 1000).astype(int)
    merged_data = merged_data.drop(['TransactionAmt'],axis=1)
    
#     merged_data['month_Amt_std'] = merged_data['Amt_log'] / merged_data.groupby(['month'])['Amt_log'].transform('std')
#     merged_data['day_Amt_std'] = merged_data['Amt_log'] / merged_data.groupby(['day'])['Amt_log'].transform('std')
#     merged_data['hour_Amt_std'] = merged_data['Amt_log'] / merged_data.groupby(['hour'])['Amt_log'].transform('std')
#     merged_data['weekday_Amt_std'] = merged_data['Amt_log'] / merged_data.groupby(['weekday'])['Amt_log'].transform('std')                                                          
                                                         
    #merged_data = merged_data.sort_values('TransactionDT')
    #**************************handle the card1-6***********************************************
    #**************************handle product CD*******************************************
#     merged_data['TransactionAmt_to_mean_CD'] = merged_data['Amt_log'] / merged_data.groupby(['ProductCD'])['Amt_log'].transform('mean')
#     merged_data['TransactionAmt_to_std_CD'] = merged_data['Amt_log'] / merged_data.groupby(['ProductCD'])['Amt_log'].transform('std')
    
    #******************************handle the NA in addr and dist*********************************************
    #merged_data[['addr1', 'addr2', 'dist1','dist2']] = merged_data[['addr1', 'addr2', 'dist1','dist2']].fillna(method="bfill")
    #merged_data['addr1_card1'] = merged_data.groupby(['card1'])['addr1'].agg(pd.Series.mode)
#     add_dis_col = [s for s in merged_data.columns if s.startswith('add') or s.startswith("dist")]
#     card_col = [s for s in merged_data.columns if s.startswith('card')]

#     for card,add in zip(card_col,add_dis_col):
#         merged_data['%s_Amt_std'%card] = merged_data['Amt_log'] / merged_data.groupby(['%s'%card])['Amt_log'].transform('std')
# #         merged_data['%s_DT_std'%add] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%add])['TransactionDT'].transform('std')
#         merged_data['%s_Amt_mean'%card] = merged_data['Amt_log'] / merged_data.groupby(['%s'%card])['Amt_log'].transform('mean')
# #         merged_data['%s_DT_mean'%add] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%add])['TransactionDT'].transform('mean')
#     temp =  ['card3','card5','dist1','addr1','addr2']  
#     for each in temp:
#         merged_data['%s_card1_std'%each] = merged_data['%s'%each] / merged_data.groupby(['card1'])['%s'%each].transform('std')
# #     
    
#     merged_data['addr1'].fillna(0, inplace=True)
#     merged_data['addr1_card1'] = merged_data['addr1'].astype(str) + '_' + merged_data['card1'].astype(str)

    #*******************************handle the p emaildomain**************************************
#     merged_data[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = merged_data['P_emaildomain'].str.split('.', expand=True)
#     # print(merged_data['TranscationAmt_to_std_Pemaildomain2'])
#     # print(merged_data['TransactionAmt_to_std_Pemaildomain1'])

#     #*******************************handle the R emaildomain**************************************

#     merged_data[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = merged_data['R_emaildomain'].str.split('.', expand=True)

#     merged_data = merged_data.drop(['P_emaildomain','R_emaildomain'],axis=1)
    #********************************handle C column################################################
#     merged_data['TransactionAmt_to_std_c5'] = merged_data['Amt_log'] / merged_data.groupby(['C5'])['Amt_log'].transform('std')
#     merged_data['TransactionAmt_to_std_c9'] = merged_data['Amt_log'] / merged_data.groupby(['C9'])['Amt_log'].transform('std')
#     merged_data['TransactionAmt_to_std_c13'] = merged_data['Amt_log'] / merged_data.groupby(['C13'])['Amt_log'].transform('std')

#     C_col = [s for s in merged_data.columns if s.startswith('C')]
#     for d in C_col:
#         merged_data['%s_std_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('std')
#         merged_data['%s_std_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('std')
#         merged_data['%s_mean_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('mean')
#         merged_data['%s_mean_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('mean')
#     for c in C_col:
#         for card in card_col:
#             merged_data['%s_std_amt_%s'%(c,card)] = merged_data['%s'%c] / merged_data.groupby(['%s'%card])['%s'%c].transform('std')
#             merged_data['%s_std_amt_%s'%(c,card)] = merged_data['%s'%c] / merged_data.groupby(['%s'%card])['%s'%c].transform('mean')
          
        
    #*******************************handle D column##################################################
    ### delete#######3
#     add_dis_col = [s for s in merged_data.columns if s.startswith('add') or s.startswith("dist")]
#     D_col = [s for s in merged_data.columns if not s.startswith('De') and s.startswith('D')]
# #     card_col = [s for s in merged_data.columns if s.startswith('card')]
#     for d in D_col:
#         merged_data['%s_std_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('std')
#         merged_data['%s_std_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('std')
#         merged_data['%s_mean_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('mean')
#         merged_data['%s_mean_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('mean')
# #             merged_data['%s%s_mean'%(d,card)] = merged_data['%s'%d] / merged_data.groupby(['%s'%card])['%s'%d].transform('mean')
#             merged_data['%s%s_mean'%(d,add)] = merged_data['%s'%d] / merged_data.groupby(['%s'%add])['%s'%d].transform('mean')

    #*******************************handle M column##################################################
#     M_col = [s for s in merged_data.columns if s.startswith('M')]
#     for d in M_col:
#         merged_data['%s_std_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('std')
#         merged_data['%s_std_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('std')
#         merged_data['%s_mean_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('mean')
#         merged_data['%s_mean_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('mean')
    ################################handle the V column################################################
#     g1 = [V95, V96, V97, V98, V99, V100, V101, V102, V103, V104, V105, V106, V107, V108, V109, V110, V111, V112, V113, V114,
#          V115, V116, V117, V118, V119, V120, V121, V122, V123, V124, V125, V126, V127, V128, V129, V130, V131, V132, V133,
#          V134, V135, V136, V137]#0.053172
#     g2 = [V217, V218, V219, V223, V224, V225, V226, V228, V229, V230, V231, V232, V233, V235, V236, V237, V240, V241, V242, V243,
#          V244, V246, V247, V248, V249, V252, V253, V254, V257,V258, V260, V261, V262, V263, V264, V265, V266, V267, V268, V269,
#          V273, V274, V275, V276, V277, V278]#77.913
#     g3 = [V167, V168, V172, V173, V176, V177, V178, V179, V181, V182, V183, V186, V187, V190, V191, V192, V193, V196, V199, 
#          V202, V203, V204, V205, V206, V207, V211, V212, V213, V214, V215, V216]#76.355370
#     g4 = [V169, V170, V171, V174, V175, V180, V184, V185, V188, V189, V194, V195, V197, V198, V200, V201, V208, V209, V210]#	76.323534
#     g5 = [V220, V221, V222, V227, V234, V238, V239, V245, V250, V251, V255, V256, V259, V270, V271, V272]#76.053104
#     g6 = [V279, V280, V284, V285, V286, V287, V290, V291, V292, V293, V294, V295, V297, V298, V299, V302, V303, V304, V305, V306, V307,
#          V308, V309, V310, V311, V312, V316, V317, V318, V319, V320, V321]#0.002032

    merged_data['g1'] = merged_data['V95'].apply(lambda x: (x.isnull()))
    merged_data['g2'] = merged_data['V217'].apply(lambda x: (x.isnull()))
    merged_data['g3'] = merged_data['V167'].apply(lambda x: (x.isnull()))
    merged_data['g4'] = merged_data['V169'].apply(lambda x: (x.isnull()))
    merged_data['g5'] = merged_data['V220'].apply(lambda x: (x.isnull()))
    merged_data['g6'] = merged_data['V279'].apply(lambda x: (x.isnull()))
    merged_data.drop(merged_data.loc[:,'V95':'V321'],axis = 1)
#     print(merged_data['ddd'])
#     print(nan_rows)
    ########################################handle the devices type and device info#########################
    #print(merged_data['DeviceInfo'])
#     merged_data['DeviceInfo'] = merged_data['DeviceInfo'].fillna('unknown_device').str.lower()
#     #print(merged_data['DeviceInfo'])
#     merged_data[['DeviceInfo_name', 'DeviceInfo_version']] = merged_data['DeviceInfo'].str.split('/',n=1, expand=True)
#     # print(merged_data['DeviceInfo'])
#     merged_data['TransactionAmt_to_std_type'] = merged_data['Amt_log'] / merged_data.groupby(['DeviceType'])['Amt_log'].transform('std')
#     merged_data['TransactionAmt_to_std_name'] = merged_data['Amt_log'] / merged_data.groupby(['DeviceInfo_name'])['Amt_log'].transform('std')
#     merged_data['TransactionAmt_to_std_version'] = merged_data['Amt_log'] / merged_data.groupby(['DeviceInfo_version'])['Amt_log'].transform('std')
#     #*******************************handle the R emaildomain**************************************
#     merged_data = merged_data.drop(['DeviceInfo','DeviceType'],axis=1)
#     print(merged_data[['DeviceInfo_name', 'DeviceInfo_version']])
#######################################handle the id column#############################################
#     id_col = [s for s in merged_data.columns if s.startswith('id')]
#     for d in id_col:
#         merged_data['%s_std_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('std')
#         merged_data['%s_std_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('std')
#         merged_data['%s_mean_amt'%d] = merged_data['Amt_log'] / merged_data.groupby(['%s'%d])['Amt_log'].transform('mean')
#         merged_data['%s_mean_DT'%d] = merged_data['TransactionDT'] / merged_data.groupby(['%s'%d])['TransactionDT'].transform('mean')

#     for each in range(1,10):h
#         merged_data['TransactionAmt_to_std_id%i'%each] =  merged_data['Amt_log'] / merged_data.groupby(['id_0%i'%each])['Amt_log'].transform('std')
#         merged_data['TransactionDT_to_std_id%i'%each] =  merged_data['TransactionDT'] / merged_data.groupby(['id_0%i'%each])['TransactionDT'].transform('std')
#     for each in range(10,39):
#         merged_data['TransactionAmt_to_std_id%i'%each] =  merged_data['Amt_log'] / merged_data.groupby(['id_%i'%each])['Amt_log'].transform('std')
#         merged_data['TransactionDT_to_std_id%i'%eac] =  merged_data['TransactionDT'] / merged_data.groupby(['id_%i'%each])['TransactionDT'].transform('std')
# #     feature_importances = pd.DataFrame()
#     feature_importances['feature'] = X.columns
    
    for each in merged_data.columns:
        if merged_data[each].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            merged_data[each] = lbl.fit_transform(merged_data[each].astype(str))
    X = pd.DataFrame()
    y = pd.DataFrame()
    if 'isFraud' in merged_data.columns:
        X = merged_data.drop(['isFraud','TransactionID'],axis =1)
        y = merged_data['isFraud']
    else:
        X = merged_data.drop(['TransactionID'], axis=1)
    print(merged_data.shape)
    return X,y

params = {'num_leaves': 423,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
         }


# local = reduce_mem_usage('../input/ieee-fraud-detection/train_transaction.csv','../input/ieee-fraud-detection/train_identity.csv')
# test = reduce_mem_usage('../input/ieee-fraud-detection/test_transaction.csv','../input/ieee-fraud-detection/test_identity.csv')
local_df = feature_transform(train)
test_df = feature_transform(test)
print("transform is finish,start training...")
NFOLDS = 5
folds = KFold(n_splits=NFOLDS,shuffle = False)
X1 = local_df[0]
y1 = local_df[1]
test = test_df[0]
#columns = local_df.columns
splits = folds.split(X1, y1)
score = 0
aucs = list()
# training_start_time = time()
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X1.iloc[train_index], X1.iloc[valid_index]
    y_train, y_valid = y1.iloc[train_index], y1.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 5000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    aucs.append(clf.best_score['valid_1']['auc'])
   
#     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
#     print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)
best_iter = clf.best_iteration
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(X1, y1)
sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
sub['isFraud'] = clf.predict_proba(test)[:, 1]
sub.to_csv('submission_cis_fraud_detection_v2.csv', index=False)


