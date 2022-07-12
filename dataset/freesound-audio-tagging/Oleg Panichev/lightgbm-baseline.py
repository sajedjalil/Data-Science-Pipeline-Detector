import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import scipy
import scipy.io.wavfile

from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm

data_path = '../input/'
train = pd.read_csv(os.path.join(data_path, 'train.csv'))
ss = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

# MAPk from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def extract_features(files, path):
    features = {}

    cnt = 0
    for f in tqdm(files):
        features[f] = {}

        fs, data = scipy.io.wavfile.read(os.path.join(path, f))

        def calc_part_features(data, n=2, prefix=''):
            f_i = 1
            step = len(data)//n
            for i in range(0, len(data), step):
                features[f]['{}mean_{}_{}'.format(prefix, f_i, n)] = np.mean(data[i:i + step])
                features[f]['{}std_{}_{}'.format(prefix, f_i, n)] = np.std(data[i:i + step])
                features[f]['{}min_{}_{}'.format(prefix, f_i, n)] = np.min(data[i:i + step])
                features[f]['{}max_{}_{}'.format(prefix, f_i, n)] = np.max(data[i:i + step])
                # features[f]['{}range_{}_{}'.format(prefix, f_i, n)] = features[f]['{}max_{}_{}'.format(prefix, f_i, n)] - features[f]['{}min_{}_{}'.format(prefix, f_i, n)]
                # features[f]['{}p25_{}_{}'.format(prefix, f_i, n)] = np.percentile(data[i:i + step], 25)
                # features[f]['{}p75_{}_{}'.format(prefix, f_i, n)] = np.percentile(data[i:i + step], 75)
                
                # h, _ = np.histogram(data, bins=10, normed=True)
                # for h_i, h_v in enumerate(h):
                #     features[f]['{}hist{}_{}_{}'.format(prefix, h_i, f_i, n)] = h_i
                f_i += 1

        features[f]['len'] = len(data)
        if features[f]['len'] > 0:
            abs_data = np.abs(data)
            diff_data = np.diff(data)
            # fft_data = np.fft.rfft(data)
            cs_data = np.cumsum(data)
        
            for n in [1, 2, 3]:
                calc_part_features(data, n=n)
                calc_part_features(abs_data, n=n, prefix='abs_')
                calc_part_features(diff_data, n=n, prefix='diff_')
                calc_part_features(cs_data, n=n, prefix='cs_')
            
            # n = 10
            # calc_part_features(fft_data, n=n, prefix='fft_')


        cnt += 1

        # if cnt >= 1000:
        #     break

    features = pd.DataFrame(features).T.reset_index()
    features.rename(columns={'index': 'fname'}, inplace=True)
    
    return features


def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids

path = os.path.join(data_path, 'audio_train')
train_files = train.fname.values
train_features = extract_features(train_files, path)

path = os.path.join(data_path, 'audio_test')
test_files = ss.fname.values
test_features = extract_features(test_files, path)

train_features = train_features.merge(train, on='fname', how='left')
# print(train_features.head())
# print(test_features.head())

# Construct features set
X = train_features.drop(['fname', 'manually_verified', 'label'], axis=1)
feature_names = list(X.columns)
X = X.values
labels = np.sort(np.unique(train.label.values))
num_class = len(labels)
c2i = {}
i2c = {}
for i, c in enumerate(labels):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_features.label.values])

X_test = test_features.drop(['fname'], axis=1)
ids = test_features.fname.values

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
map3_buf = []   

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'max_depth': 5,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0,
        'num_class': num_class,
    }  

    lgb_train = lgb.Dataset(
        X[train_index], 
        y[train_index], 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X[valid_index], 
        y[valid_index],
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
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
        for i in range(10):
            if i < len(tuples):
                print(tuples[i])
            else:
                break
            
        del importance, model_fnames, tuples

    p = model.predict(X[valid_index], num_iteration=model.best_iteration)
    _, label_ids = proba2labels(p, i2c, k=3)
    actual = [[v] for v in y[valid_index]]
    map3 = mapk(actual, label_ids, k=3)

    print('{} MAP3: {:.6f}'.format(cnt, map3))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    map3_buf.append(map3)

    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break


map3_mean = np.mean(map3_buf)
map3_std = np.std(map3_buf)
print('MAP3 = {:.6f} +/- {:.6f}'.format(map3_mean, map3_std))

preds = p_buf/cnt
str_preds, _ = proba2labels(preds, i2c, k=3)

# Prepare submission
subm = pd.DataFrame()
subm['fname'] = ids
subm['label'] = str_preds
subm.to_csv('submission.csv', index=False)
