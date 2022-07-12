import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer


# ***********************************
# * CONST                           *
# ***********************************
NUMBER_OF_FOLDS = 20
# ***********************************


def train_pred_xgb(model_params, train_x, train_y, valid_x, valid_y, test_x=None, verbose=True):
    train_data = xgb.DMatrix(data=train_x, label=train_y)
    valid_data = xgb.DMatrix(data=valid_x, label=valid_y)

    watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

    tr_model = xgb.train(
        dtrain=train_data,
        num_boost_round=100000,
        evals=watchlist,
        early_stopping_rounds=4000,
        verbose_eval=2000,
        params=model_params)

    y_valid_pred = tr_model.predict(xgb.DMatrix(valid_x), ntree_limit=tr_model.best_ntree_limit)

    if verbose:
        model_score = roc_auc_score(valid_y, y_valid_pred)
        print('XGB ROC AUC: {:07.6f}'.format(round(model_score, 6)))

    if test_x is not None:
        y_test_pred = tr_model.predict(xgb.DMatrix(test_x), ntree_limit=tr_model.best_ntree_limit)
        return y_valid_pred, y_test_pred
    else:
        return y_valid_pred


print('+------------+')
print('| Let\'s go!  |')
print('+------------+')

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ", train.shape)
print("Test shape : ", test.shape)

y = train.target.reset_index(drop=True).values
x = train.drop(['ID_code', 'target'], axis=1).values.astype('float64')
x_test = test.drop(['ID_code'], axis=1).values.astype('float64')

ss = StandardScaler()
x = ss.fit_transform(x)
x_test = ss.transform(x_test)

qt = QuantileTransformer(output_distribution='normal')
x = qt.fit_transform(x)
x_test = qt.transform(x_test)

spw = float(len(y[y == 1])) / float(len(y[y == 0]))
print('scale_pos_weight: ' + str(spw))

params_xgb = {
    'eta': 0.02,
    'max_depth': 1,
    'subsample': 0.29,
    'colsample_bytree': 0.04,
    'lambda': 0.57,
    'alpha': 0.08,
    'min_child_weight': 5.45,
    'max_delta_step': 1.53,
    'scale_pos_weight': spw,
    'tree_method': 'gpu_hist',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_gpus': 1,
    'verbosity': 0,
    'silent': True
}

valid_scores = []
test_meta = np.zeros(x_test.shape[0])
splits = list(StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=42).split(x, y))
for idx, (train_idx, valid_idx) in enumerate(splits):
    print('+-------------------------+')
    print('| Fold: {:03d}               |'.format(idx + 1))
    print('+-------------------------+')
    
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    
    pred_val_y, pred_test_y = train_pred_xgb(params_xgb, x_train, y_train, x_valid, y_valid, x_test)
    valid_scores.append(roc_auc_score(y_valid, pred_val_y))
    test_meta += pred_test_y.reshape(-1) / len(splits)

auc_score = np.asarray(valid_scores, dtype='float64').mean()

print('+-----------------------------+')
print('| ROC AUC SCORE: {:07.6f} |'.format(round(auc_score, 6)))
print('+-----------------------------+')

samp = pd.read_csv("../input/sample_submission.csv")

out_df = pd.DataFrame(columns=['ID_code', 'target'])
out_df['ID_code'] = samp["ID_code"].values
out_df['target'] = test_meta
out_df.to_csv("submission.csv", index=False)

print('submission.csv file created')
