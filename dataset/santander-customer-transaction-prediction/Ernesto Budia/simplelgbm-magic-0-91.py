import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,RepeatedKFold
import warnings
from six.moves import urllib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
from scipy.stats import norm, skew
from tqdm import tqdm_notebook as tqdm
#Load the Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


test.drop(['ID_code'], axis=1, inplace=True)
test = test.values

unique_samples = []
unique_count = np.zeros_like(test)
for feature in range(test.shape[1]):
    _, index_, count_ = np.unique(test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]


test = pd.read_csv('../input/test.csv')
test=test.iloc[real_samples_indexes]

df=pd.concat([test, train])

train=df[~df.target.isnull()]
test=df[df.target.isnull()]
print(train.shape)
print(test.shape)

features = [c for c in train.columns if c not in ['ID_code', 'target']]
df['c_neg_6']=(df<-6).sum(axis=1)



def create_freq(x,data):
    aux=data.groupby(x).agg({'ID_code':'count'})
    aux['freq'+x]=aux.ID_code/aux.ID_code.sum()
    del aux['ID_code']
    data = data.merge(aux, on=x,how='left')
    return(data)

for r in features:
    df=create_freq(r,df)

train=df[~df.target.isnull()]
test=df[df.target.isnull()]
print(train.shape)
print(test.shape)



nval=20000
val=train[:nval]
train=train[nval:]





features = [c for c in train.columns if c not in ['ID_code', 'target']]

param = {
    #'bagging_freq': 5,
    #'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    #'feature_fraction': 0.041,
    'learning_rate': 0.03,
    'max_depth': -1,
    'metric':'auc',
    #'min_data_in_leaf': 80,
    #'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 3,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}


trn_data = lgb.Dataset(train[features], label= train['target'])
val_data= lgb.Dataset(val[features], label= val['target'])
clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
predictions = clf.predict(test[features], num_iteration=clf.best_iteration)
test["target"]= predictions
sub = pd.read_csv('../input/test.csv')
sub = sub[['ID_code']].merge(test[['ID_code','target']], on='ID_code',how='left')
sub = sub.fillna(0.5)
print(sub.shape)
sub.to_csv('submission.csv', index=False)


