#-*- coding: UTF-8 -*-
# xgboost+lr,excited!!!!!!
# final lb on the Santander Customer Match: 1) whole train data gbdt feature generation,lb: 0.822072; 2) sampled train data gbdt feature generation,lb: 0.821391
import time
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print("start time: %s" % time.ctime())
check_time = time.time()
time_name = str(time.ctime())
file_time = "_".join(time_name.split()[:3])
GBDT_FILE = 'gbdt_imbalance_%s.txt' % file_time    #tree node file
#GBDT_FILE = 'gbdt_imbalance_Fri_May_20.txt'   #test
COEF_FILE = 'Logistic_coef_%s.txt' % file_time
IMPORTANCE = 3    #最后输出相关性要保留的小数点位数
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print('Load data...')
train = pd.read_csv("../input/train.csv")       #update
train_id = train['ID'].values
target = train['TARGET'].values
#train = train.drop(['ID','TARGET'],axis=1)

test = pd.read_csv("../input/test.csv")          #update
test_id = test['ID'].values
#test = test.drop(['ID'],axis=1)
print('load data complete, train records count: <<<<%s, test records count: <<<<%s, train columns count: <<<<%s, test columns count: <<<<%s, time: <<<<%s' % (train.shape[0], test.shape[0], train.shape[1], test.shape[1], round(((time.time() - check_time)/60),2)))
check_time = time.time()
#------------------------------------------data process-------------------------------------------------------------
#removing outliers, MANUALLY
#train = train.replace(-999999,2)
#test = test.replace(-999999,2)

#replace na
train = train.fillna(-1)
test = test.fillna(-1)

# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
len1 = len(remove)
print(train.shape, test.shape)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
len2 = len(remove)

#label encoder
len3 = 0
cols = train.columns
origin_cols = train.columns[1:-1]        #original features
print("origin_cols: %s" % origin_cols)
for col in origin_cols:
     if train[col].dtype=='object':
            print(col)
            len3 += 1
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[col].values) + list(test[col].values))
            train[col] = lbl.transform(list(train[col].values))
            test[col] = lbl.transform(list(test[col].values))

print("data process ended, row droped:<<<<%s, row encoded:<<<<%s, time spend: <<<<%s" %(len1+len2, len3, round(((time.time() - check_time)/60),2)))
check_time = time.time()

#-----------------------------------------------------GBDT model on sample------------------------------------------------------------------------
"""
folds = 10
skf = StratifiedKFold(target,
                          n_folds=folds,
                          shuffle=False,
                          random_state=1580)  
a, sample_index = list(skf)[0]
train_sample = train.loc[sample_index,:]
target_sample = target[sample_index]
xgbc = xgb.XGBClassifier(n_estimators=500, seed=1580)
xgbc.fit(train_sample[origin_cols],target_sample)
xgbc.booster().dump_model(GBDT_FILE)
print('gbdt trained on sample, time spend:<<<<%s' % round(((time.time() - check_time)/60),2))
check_time = time.time()
"""
#-----------------------------------------------------GBDT model------------------------------------------------------------------------
xgbc = xgb.XGBClassifier(n_estimators=500, seed=1580)
xgbc.fit(train[origin_cols],target)
xgbc.booster().dump_model(GBDT_FILE)
print('gbdt trained on sample, time spend:<<<<%s' % round(((time.time() - check_time)/60),2))
check_time = time.time()

#-----------------------------------------------------READ FEATURE----------------------------------------------------------------------
f = open(GBDT_FILE,'r')
feature_set = set()
for line in f.readlines():
    if '<' in line:           #feature line
           line = line.split(':')[1].strip()
           feature_re = re.match('\[(.*)?\]', line)
           info = feature_re.group(0)              #should be only one group
           info = re.sub('\[|\]','',info)
           feature_set.add(info)

#feature encoder
for info in feature_set:
    key = info.split("<")[0].strip()
    value = float(info.split("<")[1].strip())
    train[info] = train[key].apply(lambda x: 1 if x<value else 0)
    test[info] = test[key].apply(lambda x: 1 if x<value else 0)

#remove original feature
train = train.drop(origin_cols, axis=1)
test = test.drop(origin_cols, axis=1)
print('feature generated base on gbdt sub-model, <<<<%s feature generated, time spend:<<<<%s' % (test.shape[1]-1, round(((time.time() - check_time)/60),2)))
check_time = time.time()

#--------------------------------------------------------UNDERSAMPLE-------------------------------------------------------------------
train1 = train[train['TARGET']==1]               #positive train samples
train2 = train[train['TARGET']==0]               #negative train samples
#train2 suppose to be the majority type, if not change it
if train2.shape[0]<train1.shape[0]:
    train1 = train[train['TARGET']==0]               #positive train samples
    train2 = train[train['TARGET']==1]               #negative train samples
train1 = train1.reset_index(drop=True)
train2 = train2.reset_index(drop=True)
fold = train2.shape[0] / train1.shape[0]
folds = int(fold / 4.)
skf1 = StratifiedKFold(train1.TARGET.values,
                          n_folds=folds,
                          shuffle=False,
                          random_state=1580)
skf2 = StratifiedKFold(train2.TARGET.values,
                          n_folds=folds,
                          shuffle=False,
                          random_state=1580)    
#------------------------------------------------------LVL1 Logistic Regression--------------------------------------------------------
clf = LogisticRegression(penalty='l1', C=0.25, random_state=1580, n_jobs=-1)
features = list(train.columns)
features.remove('ID')
features.remove('TARGET')
df_train_pred = []
df_pred = []
result = []
for i, (a, neg_index) in enumerate(skf2):
        fold_tag = "fold_%s" % i
        pos_index = list(skf1)[i][0]
        train_pos = train1.loc[pos_index, :]
        trainner = pd.concat((train_pos, train2.loc[neg_index,:]),axis=0, ignore_index=True)
        y = trainner.TARGET.values
        X = trainner[features]
        clf.fit(X,y)
        train_pred = clf.predict_proba(train[features])[:,1]
        y_pred = clf.predict_proba(test[features])[:,1]
        df_train_pred.append(train_pred)
        df_pred.append(y_pred)
        result.append(clf.coef_[0])

#average results
train_preds = np.average(np.array(df_train_pred), axis=0)
test_preds = np.average(np.array(df_pred), axis=0)

#output feature weights
coef = np.average(np.array(result), axis=0)
result = sorted(zip(map(lambda x:round(x,IMPORTANCE), coef), features), reverse=True)
o = open(COEF_FILE, "w")
o.write("len of feature: %s\n" % len(result))
for (value, key) in result:
    if 0==value:
       pass
    else:
       writeln = "%s: %.3f\n" % (key, value)
       o.write(writeln)
o.close()

print('LVL1 Logistic Model trained, Average AUC: <<<<%s, time spend:<<<<%s' % (roc_auc_score(train.TARGET.values, train_preds), round(((time.time() - check_time)/60),2)))
check_time = time.time()
#-----------------------------------------------------OUTPUT----------------------------------------------------------------------------
pd.DataFrame({"ID": test_id, "TARGET": test_preds}).to_csv('gbdt_lr_baseline.csv',index=False)
print("end time: %s" % time.ctime())