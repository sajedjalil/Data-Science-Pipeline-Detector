### In this version we try to find Z score using the mean and std for target 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy import stats
from scipy.stats import rankdata

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
sns.set(font_scale=1)
random_state = 42
np.random.seed(random_state)
#df_train = pd.read_csv('../input/train.csv')
#df_test = pd.read_csv('../input/test.csv')
lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 9,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state,
    "num_threads": 4
}


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

mean = []
std = []
base_feat = []

for i in range(0,200):
    mean_tmp = df_train[df_train.target==1]['var_'+str(i)].mean()
    std_tmp = df_train[df_train.target==1]['var_'+str(i)].std()
    mean.append(mean_tmp)
    std.append(std_tmp)
    base_feat.append('var_'+str(i))

def get_features(df):
    #df['var_QOLSCORE'] = 0
    df['var_ZOLSCORE'] = 0
    for i in range(0,200):
        #### Get the Z-score outliers
        df['var_'+str(i)+'_Z'] = stats.norm.cdf((df['var_'+str(i)] - mean[i])/std[i])
        df['var_'+str(i)+'_ZOL'] = 0
        df.loc[df['var_'+str(i)+'_Z'] > 0.95 ,'var_'+str(i)+'_ZOL'] = 1
        df.loc[df['var_'+str(i)+'_Z'] < 0.05 ,'var_'+str(i)+'_ZOL'] = 1
#         df['var_QOLSCORE'] += df['var_'+str(i)+'_OL'] 
        df['var_ZOLSCORE'] += df['var_'+str(i)+'_ZOL']
        df = df.drop(['var_'+str(i),'var_'+str(i)+'_ZOL'],1)
    df = df.fillna(0)
    return df
    
def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y





print("########################################----Feature Set:2 ---########################")
features = [col for col in df_train.columns if col in (base_feat)]
X_test = get_features(df_test)
features1 = X_test.iloc[:,1:].columns
print(features1)
X_test = X_test.iloc[:,1:].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = df_test[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
a = label_encoder.fit_transform(df_train['var_68'])

for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train,df_train['target'].values)):
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    X_valid = get_features(X_valid)
    print("########################### fold: "+str(fold)+"####################")
    N = 4
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
        X_t = get_features(X_t)
        print("########################### sub fold: "+str(i)+"####################")
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        35000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features1
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    predictions['fold{}'.format(fold+1)] = yp/N
    
mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv('lgb_submission.csv', index=False)
oof.to_csv('lgb_oof.csv', index=False)
feature_importance_df.to_csv('feat_imp.csv', index=False)