import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pickle
os.system('pip install xgboost')

import xgboost as xgb

def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def custom_gini_eval(estimator, X, y):
    pred = estimator.predict_proba(X)[:, 1]
    return gini_normalized(y, pred)


def process(frame, drop_na=False):
    """
    Fix missing values and apply one-hot encoding.
    """
    result = []
    for col_name, col in frame.iteritems():
        col_data = col.copy(deep=True)
        missing_indicators = (col_data == -1)
        if col_name.endswith('bin'):
            if missing_indicators.sum() > 0:
                if drop_na:
                    col_data[missing_indicators] = np.nan
                else:
                    mode = col_data[~missing_indicators].mode().item()
                    col_data[missing_indicators] = mode
            result.append(pd.DataFrame(col_data))
        elif col_name.endswith('cat'):
            if missing_indicators.sum() > 0:
                col_data[missing_indicators] = np.nan
                one_hot = pd.get_dummies(col_data, prefix=col_name)
                if drop_na:
                    for one_hot_name, _ in one_hot.iteritems():
                        one_hot.loc[missing_indicators, one_hot_name] = np.nan
            else:
                one_hot = pd.get_dummies(col_data, prefix=col_name)
            result.append(one_hot)
        else:
            if missing_indicators.sum() > 0:
                if drop_na:
                    col_data[missing_indicators] = np.nan
                else:
                    mode = col_data[~missing_indicators].mode().item()
                    col_data[missing_indicators] = mode
            result.append(pd.DataFrame(col_data))
    processed = pd.concat(result, axis=1)
    if drop_na:
        processed.dropna(inplace=True)
    return processed

def preprocess(data, force=False):
    path_train, path_test = '/kaggle/working/train_processed.csv', '/kaggle/working/test_processed.csv'
    if force or not os.path.exists(path_train) or not os.path.exists(path_test):
        train_raw, test_raw = data['raw_train'], data['raw_test']
        train_frame, test_frame = process(train_raw), process(test_raw)
        train_frame.to_csv(path_train)
        test_frame.to_csv(path_test)
    else:
        train_frame, test_frame = pd.read_csv(path_train), pd.read_csv(path_test)
    data['processed_train'] = train_frame
    data['processed_test'] = test_frame
    return data

def cross_validation(train_x, train_y):
    """
    Five-fold cross validation.
    """
    d = train_x.shape[1]
    path = f'/kaggle/working/model_{d}.pkl'
    if os.path.exists(path):
        with open(path, mode='rb') as file:
            cv_results = pickle.load(file)
    else:
        param = {'max_depth': 1, 'eta': 1, 'objective': 'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        param['gpu_id'] = 0
        param['tree_method'] = 'gpu_hist'
        parameters = {'n_estimators': [10, 20, 50, 100]}
        boost = xgb.XGBClassifier(**param)
        clf = GridSearchCV(boost, parameters, cv=5, scoring=custom_gini_eval)
        clf.fit(train_x, train_y)
        cv_results = clf.cv_results_
        cv_results['best_params'] = clf.best_params_
        cv_results['best_score'] = clf.best_score_
        with open(path, mode='wb') as file:
            pickle.dump(cv_results, file)
    return cv_results

def fit(train_x, train_y, n_estimator, verbose=True):
    """
    Train the final model.
    """
    param = {'max_depth': 1, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    param['gpu_id'] = 0
    param['tree_method'] = 'gpu_hist'
    param['n_estimators'] = n_estimator
    model = xgb.XGBClassifier(**param)
    model.fit(train_x, train_y)
    if verbose:
        y_pred = model.predict(train_x)
        y_pred_prob = model.predict_proba(train_x)[:, 1]
        train_acc = accuracy_score(train_y, y_pred)
        gini_index = gini_normalized(train_y, y_pred_prob)
        print(f'Training accuracy is {train_acc}, gini={gini_index}.')
    return model


def predict(model, test_x):
    y_pred = model.predict_proba(test_x)
    return y_pred

def report(data, predictions):
    """
    Output the final submission file.
    """
    ids = data['raw_test']['id'].to_numpy().astype(np.int64)
    target = predictions[:, 1]
    path = '/kaggle/working/submission.csv'
    output = pd.DataFrame(data={'id': ids, 'target': target})
    output.to_csv(path, index=False)


def read():
    train_frame, test_frame = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv'), pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
    data = {
        'raw_train': train_frame,
        'raw_test': test_frame
    }
    return data

def dimension_reduction(data, n_components, alias='train'):
    path = f'/kaggle/working/pca_feature_{alias}_{n_components}.npz'
    if not os.path.exists(path):
        frame = data[f'processed_{alias}']
        x = frame[frame.columns.difference(['id', 'target'])].to_numpy()
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(x)
        np.savez_compressed(path, features=pca_features)
    else:
        file = np.load(path)
        pca_features = file['features']
    return pca_features

def run():
    n_components = [3, 4, 5, 6, 7, 8, 9, 10]
    print('Read csv files.')
    data = read()
    print('Preprocess features')
    processed_data = preprocess(data)
    train_y = processed_data['processed_train']['target'].to_numpy()
    print('Model selection.')
    train_xs = {}
    model_selection = []
    for n_component in n_components:
        train_x = dimension_reduction(processed_data, n_component, 'train')
        test_x = dimension_reduction(processed_data, n_component, 'test')
        train_xs[n_component] = (train_x, test_x)
        cv_results = cross_validation(train_x, train_y)
        model_selection.append((cv_results['best_score'], cv_results['best_params'], n_component))
        print(fr'PCA dimension={n_component}, n_estimators={cv_results["best_params"]["n_estimators"]}, best score ={cv_results["best_score"]}.')
    model_selection = sorted(model_selection, key=lambda x: x[0], reverse=True)
    print(f'Best parameters to train the final model: {model_selection[0]}')
    model = fit(train_xs[model_selection[0][2]][0], train_y, n_estimator=model_selection[0][1]['n_estimators'])
    predictions = predict(model, train_xs[model_selection[0][2]][1])
    report(processed_data, predictions)

run()