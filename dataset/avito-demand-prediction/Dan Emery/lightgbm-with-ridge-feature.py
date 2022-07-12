import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

NFOLDS = 5
SEED = 42

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
class XgbWrapperEarly(object):
    def __init__(self, seed = SEED, params = None):
        self.param = params
        self.param['seed'] = seed
    
    def train(self, x_train, y_train):
        tr, val, y_tr, y_val = train_test_split(x_train, y_train, train_size = 0.9, random_state = SEED)
        dtrain = xgb.DMatrix(tr, label = y_tr)
        dval = xgb.DMatrix(val, label = y_val)
        watchlist = [(dval,'valid')]
        self.gbdt = xgb.train(self.param, dtrain, 10000, watchlist, early_stopping_rounds = 200, verbose_eval = 200)
    
    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def get_oof_xgb(clf, x_train,y_train, x_test):
    preds = np.array([])
    for i in range(NFOLDS):
        split = x_train.shape[0] / NFOLDS
        test_fold = x_train[int(i*split):int((i+1)*split)]
        y = np.concatenate((y_train[0:int(i*split)], y_train[int((i+1)*split):]))
        train_fold = pd.concat([x_train[0:int(i*split)], x_train[int((i+1)*split):]], axis = 0)
        clf.train(train_fold, y)
        pred = clf.predict(test_fold)
        print('\nCompleted Fold {}'.format(i))
        
    clf.train(x_train, y_train)
    pred = clf.predict(x_test)
    return (preds, pred)
    
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

print("\nData Load Stage")
training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
traindex = training.index
testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testing.index

ntrain = training.shape[0]
ntest = testing.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1"]
print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")

# Feature Engineering 
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features
    
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

# Meta Text Features
textfeats = ["description","text_feat", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    'strip_accents':'unicode',
    "dtype": np.float32,
    "norm": 'l2',
    "smooth_idf":False
}


def get_col(col_name): return lambda x: x[col_name]
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=45000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            preprocessor=get_col('title'))),
        ('text_feat',CountVectorizer(
            ngram_range=(1, 2),
            preprocessor=get_col('text_feat')))
    ])
    
start_vect=time.time()
vectorizer.fit(df.loc[traindex,:].to_dict('records'))

X = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description","text_feat", "title"]
df.drop(textfeats, axis=1,inplace=True)

from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
                
ridge_params_2 = {'alpha':10.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
                
ridge_params_3 = {'alpha':1.0, 'fit_intercept':False, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
                
xgb_params = {'objective' : "reg:logistic",
          'booster' : "gbtree",
          'eval_metric' : "rmse",
          'nthread' : 4,
          'eta':0.2,
          'max_depth':15,
          'min_child_weight': 2,
          'gamma' :0,
          'subsample':0.7,
          'colsample_bytree':0.7,
          'aplha':0,
          'lambda':0}  


import xgboost as xgb
xg = XgbWrapperEarly(seed=SEED, params=xgb_params)
xgb_oof_train, xgb_oof_test = get_oof_xgb(xg,X[:ntrain], y, X[ntrain:])

ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, X[:ntrain], y, X[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

ridge_2 = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params_2)
ridge_oof_train_2, ridge_oof_test_2 = get_oof(ridge_2, X[:ntrain], y, X[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train_2))
print('Ridge OOF RMSE: {}'.format(rms))

ridge_3 = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params_3)
ridge_oof_train_3, ridge_oof_test_3 = get_oof(ridge_3, X[:ntrain], y, X[ntrain:])


rms = sqrt(mean_squared_error(y, ridge_oof_train_3))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
ridge_preds_2 = np.concatenate([ridge_oof_train_2, ridge_oof_test_2])
ridge_preds_3 = np.concatenate([ridge_oof_train_3, ridge_oof_test_3])
xgb_preds = np.concatenate([xgb_oof_train, xgb_oof_test])

df['ridge_preds'] = ridge_preds
df['ridge_preds_2'] = ridge_preds_2
df['ridge_preds_3'] = ridge_preds_3
df['xgb_preds'] = xgb_preds
# Combine Dense Features with Sparse Text Bag of Words Features
X = df[:ntrain]
testing = df[ntrain:]

for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();

print("\nModeling Stage")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=23)
    
print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves':255,
    'max_depth': 8,
    'bagging_fraction': 0.8,
    'learning_rate': 0.02,
    'verbose': 0
} 

lgtrain = lgb.Dataset(X_train, y_train,
                categorical_feature = categorical)
                
lgvalid = lgb.Dataset(X_valid, y_valid,
                categorical_feature = categorical)

modelstart = time.time()
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=16000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=200,
    verbose_eval=200
)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))