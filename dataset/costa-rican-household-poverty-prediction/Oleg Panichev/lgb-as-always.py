import datetime
import gc
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
import category_encoders as ce
import lightgbm as lgb

def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + \
        " ".join(map(str,args)), **kwargs)

id_name = 'Id'
target_name = 'Target'

# Load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['is_test'] = 0
test['is_test'] = 1
df_all = pd.concat([train, test], axis=0)

dprint('Clean features...')
cols = ['dependency']
for c in tqdm(cols):
    x = df_all[c].values
    strs = []
    for i, v in enumerate(x):
        try:
            val = float(v)
        except:
            strs.append(v)
            val = np.nan
        x[i] = val
    strs = np.unique(strs)

    for s in strs:
        df_all[c + '_' + s] = df_all[c].apply(lambda x: 1 if x == s else 0)

    df_all[c] = x
    df_all[c] = df_all[c].astype(float)
dprint("Done.")

# dprint("Dummy features...")
# cols = ['idhogar']
# dprint("len(cols) = {}".format(len(cols)))
# df_all = pd.get_dummies(df_all, dummy_na=True, columns=cols)
# dprint("df_all.shape = {}".format(df_all.shape))
# dprint("Done.")

train = df_all.loc[df_all['is_test'] == 0].drop(['is_test'], axis=1)
test = df_all.loc[df_all['is_test'] == 1].drop(['is_test'], axis=1)

dprint('Label Encoder...')
cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object' and f_ != id_name]
print(cols)
for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
del le
gc.collect()
dprint("Done.")

dprint("Extracting features...")
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['rent_to_bedrooms'] = df['v2a1']/df['bedrooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['tamhog_to_bedrooms'] = df['tamhog']/df['bedrooms']
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['r4t3_to_bedrooms'] = df['r4t3']/df['bedrooms']
    df['rent_to_r4t3'] = df['v2a1']/df['r4t3']
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1'])
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms']
    df['hhsize_to_bedrooms'] = df['hhsize']/df['bedrooms']
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize']
    df['qmobilephone_to_r4t3'] = df['qmobilephone']/df['r4t3']
    df['qmobilephone_to_v18q1'] = df['qmobilephone']/df['v18q1']
    

extract_features(train)
extract_features(test)
dprint("Done.")         

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 5
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
err_buf = []   

cols_to_drop = [
    id_name, 
    target_name,
]
X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train[target_name].values

classes = np.unique(y)
dprint('Number of classes: {}'.format(len(classes)))
c2i = {}
i2c = {}
for i, c in enumerate(classes):
    c2i[c] = i
    i2c[i] = c

y_le = np.array([c2i[c] for c in y])

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test[id_name].values

dprint(X.shape, y.shape)
dprint(X_test.shape)

n_features = X.shape[1]

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 32,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 6,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0,
    'num_class': len(np.unique(y)),
}

# For encoders
enc_cols = [
    'idhogar', 
    'rooms', 
    'bedrooms',
    'r4t3', 
    'hogar_adul', 
    'dependency',
]
feature_names_initial = list(X.columns)

for train_index, valid_index in kf.split(X, y):
    print('Fold {}/{}*{}'.format(cnt + 1, n_splits, n_repeats))
    params = lgb_params.copy() 

    sampler = RandomUnderSampler(random_state=0)
    X_train_index, y_train = sampler.fit_sample(train_index.reshape(-1, 1), y_le[train_index])
    X_valid_index, y_valid = sampler.fit_sample(valid_index.reshape(-1, 1), y_le[valid_index])

    # print(train_index, X_train_index)
    X_train = X.loc[X_train_index.ravel()].copy()
    X_valid = X.loc[X_valid_index.ravel()].copy()
    X_test_te = X_test.copy()

    dprint('Encoders...')  
    print(X_train.shape) 
    
    def encode(encoder, cols, X_train, X_valid, X_test, y_train, suffix='enc'):
        print('X_train: {}, X_valid: {}, X_test: {}'.format(
            X_train.shape, 
            X_valid.shape, 
            X_test.shape))
        encoder.fit(X_train, y_train)
        X_train_enc = encoder.transform(X_train.copy())
        X_valid_enc = encoder.transform(X_valid.copy())
        X_test_enc = encoder.transform(X_test.copy())

        X_train_enc.columns = [c + '_' + suffix for c in X_train_enc.columns]
        X_valid_enc.columns = [c + '_' + suffix for c in X_valid_enc.columns]
        X_test_enc.columns = [c + '_' + suffix for c in X_test_enc.columns]

        print('X_train_enc: {}, X_valid_enc: {}, X_train_enc: {}'.format(
            X_train_enc.shape, 
            X_valid_enc.shape, 
            X_train_enc.shape))

        return X_train_enc, X_valid_enc, X_test_enc

    print('TE')
    encoder = ce.TargetEncoder(cols=enc_cols)
    X_train_enc0, X_valid_enc0, X_test_enc0 = encode(
        encoder, 
        enc_cols, 
        X_train[enc_cols], 
        X_valid[enc_cols], 
        X_test_te[enc_cols], 
        y_train,
        suffix='te')

    X_train = pd.concat([X_train, X_train_enc0], axis=1)
    X_valid = pd.concat([X_valid, X_valid_enc0], axis=1)
    X_test_te = pd.concat([X_test_te, X_test_enc0], axis=1)

    print('HE0')
    encoder = ce.HashingEncoder(cols=feature_names_initial, n_components=4)
    X_train_enc1, X_valid_enc1, X_test_enc1 = encode(
        encoder, 
        feature_names_initial, 
        X_train[feature_names_initial], 
        X_valid[feature_names_initial], 
        X_test_te[feature_names_initial], 
        y_train,
        suffix='he0')

    X_train = pd.concat([X_train, X_train_enc1], axis=1)
    X_valid = pd.concat([X_valid, X_valid_enc1], axis=1)
    X_test_te = pd.concat([X_test_te, X_test_enc1], axis=1)

    print('HE1')
    encoder = ce.HashingEncoder(cols=enc_cols, n_components=4)
    X_train_enc2, X_valid_enc2, X_test_enc2 = encode(
        encoder, 
        enc_cols, 
        X_train[enc_cols], 
        X_valid[enc_cols], 
        X_test_te[enc_cols], 
        y_train,
        suffix='he1')

    X_train = pd.concat([X_train, X_train_enc2], axis=1)
    X_valid = pd.concat([X_valid, X_valid_enc2], axis=1)
    X_test_te = pd.concat([X_test_te, X_test_enc2], axis=1)

    print(X_train.shape)
    feature_names = list(X_train.columns)
    dprint("Done.")

    X_train = X_train.values
    X_valid = X_valid.values
    X_test_te = X_test_te.values

    lgb_train = lgb.Dataset(
        X_train, 
        y_train, 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X_valid, 
        y_valid,
        feature_name=feature_names,
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=99999,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100, 
        verbose_eval=100, 
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(20):
            if i < len(tuples):
                print(i, tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X_valid, num_iteration=model.best_iteration)

    err = f1_score(y_valid, np.argmax(p, axis=1), average='macro')

    dprint('{} F1: {}'.format(cnt + 1, err))

    p = model.predict(X_test_te, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    err_buf.append(err)

    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break

    del model, lgb_train, lgb_valid, p
    gc.collect()


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.6f} +/- {:.6f}'.format(err_mean, err_std))

preds = p_buf/cnt

# Prepare probas
subm = pd.DataFrame()
subm[id_name] = id_test
for i in range(preds.shape[1]):
    subm[i2c[i]] = preds[:, i]
subm.to_csv('submission_{:.6f}_probas.csv'.format(err_mean), index=False)

# Prepare submission
subm = pd.DataFrame()
subm[id_name] = id_test
subm[target_name] = [i2c[np.argmax(p)] for p in preds]
subm[target_name] = subm[target_name].astype(int)
subm.to_csv('submission_{:.6f}.csv'.format(err_mean), index=False)