import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from math import exp, log
import xgboost as xgb


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))


def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def mymae(preds, dtrain):
	labels = dtrain.get_label()
	return 'mymae', mean_absolute_error(labels, preds)

DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'


def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train_loader = pd.read_csv(path_train, dtype={'id': np.int32})
    train = train_loader.drop(['id', 'loss'], axis=1)
    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    test = test_loader.drop(['id'], axis=1)
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index

    # compute skew and do Box-Cox transformation
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    # transform features with skew > 0.25 (this can be varied to find optimal value)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index
    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    features = train.columns
    cats = [feat for feat in features if 'cat' in feat]
    # factorize categorical features
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    x_train = train_test.iloc[:ntrain, :]
    x_test = train_test.iloc[ntrain:, :]
    train_test_scaled, scaler = scale_data(train_test)
    train, _ = scale_data(x_train, scaler)
    test, _ = scale_data(x_test, scaler)

    train_labels = np.log(np.array(train_loader['loss']))
    train_ids = train_loader['id'].values.astype(np.int32)
    test_ids = test_loader['id'].values.astype(np.int32)

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    
    x_train.set_index(train_ids, inplace=True)
    train.set_index(train_ids, inplace=True)
    x_test.set_index(test_ids, inplace=True)
    test.set_index(test_ids, inplace=True)

    train = pd.concat((train, x_train), axis=1)
    test = pd.concat((test, x_test), axis=1)
    print("Train/test:")
    print(train.head(1))
    print(test.head(1))
    return train.as_matrix(), train_labels, test.as_matrix(), train_ids, test_ids

################################## Actual Run Code ##################################

# enter the number of folds from xgb.cv
folds = 5
cv_sum = 0
early_stopping = 25
fpred = []
xgb_rounds = []
random_state = 1001
bagging_rounds = 1
start_time = timer(None)

# Load data set and target values
train, target, test, _, ids = load_data()
d_train_full = xgb.DMatrix(train, label=target)
d_test = xgb.DMatrix(test)

bag_avg = pd.read_csv('../input/sample_submission.csv', dtype={'id': np.int32}, index_col='id')

for bag in range(0, bagging_rounds):
    print('\n Bag %d\n' % (bag + 1)) 
    my_random_state = random_state + bag
    np.random.seed(my_random_state)

    # set up KFold that matches xgb.cv number of folds
    cv_sum = 0
    kf = KFold(train.shape[0], n_folds=folds, random_state=my_random_state)
    for i, (train_index, test_index) in enumerate(kf):
        print('Fold %d' % (i + 1))
        X_train, X_val = train[train_index], train[test_index]
        y_train, y_val = target[train_index], target[test_index]

    #######################################
    #
    # Define cross-validation variables
    #
    #######################################
    
        params = {}
        params['booster'] = 'gbtree'
        params['objective'] = "reg:linear"
        #params['eval_metric'] = 'mae'
        params['eta'] = 0.0999
        params['gamma'] = 0.5290
        params['min_child_weight'] = 4.2922
        params['colsample_bytree'] = 0.3085
        params['subsample'] = 0.9930
        params['max_depth'] = 7
        params['max_delta_step'] = 0
        params['silent'] = 1
        params['random_state'] = my_random_state
    
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    
    ####################################
    #  Build Model
    ####################################
    
        clf = xgb.train(params,
                        d_train,
                        100000,
                        watchlist,
                        early_stopping_rounds=early_stopping,
                        verbose_eval=False,
                        feval=mymae)
    
    ####################################
    #  Evaluate Model and Predict
    ####################################
    
        xgb_rounds.append(clf.best_iteration)
        scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
        cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
        print(' eval-MAE: %.6f' % cv_score)
        y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))
    
    ####################################
    #  Add Predictions and Average Them
    ####################################
    
        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred
        cv_sum = cv_sum + cv_score
    
    mpred = pred / folds
    score = cv_sum / folds
    print('Average eval-MAE: %.6f' % score)
    n_rounds = int(np.mean(xgb_rounds))
    
    ####################################
    #  Make Full Dataset Predictions
    ####################################
    
    print('Training full dataset for %d rounds ...' % n_rounds)
    watchlist = [(d_train_full, 'train')]
    clf_full = xgb.train(
        params, d_train_full,
        n_rounds,
        watchlist,
        verbose_eval=False,
        feval=mymae)
    y_pred_full = np.exp(clf_full.predict(d_test))
    
    # enter the number of iterations from xgb.cv with early_stopping turned on
    n_fixed = 376
    
    nfixed = int(n_fixed * (1 + (1. / folds)))
    print('Training full dataset for %d rounds ...\n' % nfixed)
    clf_fixed = xgb.train(
        params, d_train_full,
        nfixed,
        watchlist,
        verbose_eval=False,
        feval=mymae)
    y_pred_fixed = np.exp(clf_fixed.predict(d_test))
    timer(start_time)

    print("# Writing results")
    result = pd.DataFrame(mpred, columns=['loss'])
    result["id"] = ids
    result = result.set_index("id")
    print("%d-fold average prediction:\n" % folds)
    print(result.head())
    result_full = pd.DataFrame(y_pred_full, columns=['loss'])
    result_full["id"] = ids
    result_full = result_full.set_index("id")
    print(" Full dataset prediction:\n")
    print(result_full.head())
    result_fixed = pd.DataFrame(y_pred_fixed, columns=['loss'])
    result_fixed["id"] = ids
    result_fixed = result_fixed.set_index("id")
    print("Full datset (at CV #iterations) prediction:\n")
    print(result_fixed.head())

    avg = (result + result_full + result_fixed) / 3
    bag_avg = ((bag_avg * bag) + avg) / (1 + bag)


now = datetime.now()
sub_file = 'submission_bags_' + str(bagging_rounds) + '_' + str(
    now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission: %s" % sub_file)
print(bag_avg.head())
bag_avg.to_csv(sub_file, index=True, index_label='id')

print("Done")
