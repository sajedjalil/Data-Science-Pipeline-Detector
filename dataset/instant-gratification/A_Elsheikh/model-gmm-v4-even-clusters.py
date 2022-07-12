import numpy as np
import pandas as pd
import os
import time

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
import _pickle as cPickle

import warnings
warnings.filterwarnings("ignore")


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
            os.makedirs(dir)


def save_pickle(obj, filename, filepath, protocol=3):
    assure_path_exists(filepath)
    with open(filepath + filename, 'wb') as pickle_file:
            cPickle.dump(obj, pickle_file, protocol=protocol)  # , pickle.HIGHEST_PROTOCOL)


def load_pickle(filename, filepath):
    with open(filepath + filename, 'rb') as pickle_file:
        obj = cPickle.load(pickle_file)
    return obj


if __name__ == '__main__':
    # rsync -avhz --stats --progress -e 'ssh -p 22' /media/second/kaggle_IG/ ahmed@santaka:/media/second/kaggle_IG/ --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.csv' --exclude='*.pdf' --exclude='*.h5' --exclude='*.zip' --exclude='*.ipynb' --exclude='*.pkl' --exclude='*.feather'
    kaggle_run = True

    try:
        import feather
        print('Loading data from feather file')
        t0 = time.time()
        train_df = feather.read_dataframe('../input/train.feather')
        test_df = feather.read_dataframe('../input/test.feather')
        print('Done loading data in {} sec'.format(time.time() - t0))
    except Exception as e:
        print('Failed in ldata from feather file, try CSV !!')
        t0 = time.time()
        train_df = pd.read_csv('../input/train.csv')
        test_df = pd.read_csv('../input/test.csv')
        print('Done loading data in {} sec'.format(time.time() - t0))

        # import feather
        # train_df.to_feather('../input/train.feather')
        # test_df.to_feather('../input/test.feather')

    n_train_samples = train_df.shape[0]

    if kaggle_run:
        # early termination
        sub_df = pd.read_csv('../input/sample_submission.csv')
        if sub_df.shape[0] == 131073 and sub_df.iloc[0].id == '1c13f2701648e0b0d46d8a2a5a131a53':
            sub_df['target'] = -99
            sub_df.to_csv('submission.csv', index=False)
            print('Early exit to save run time on Kaggle servers')
            exit(0)
    else:
        os.nice(18)

    cols = [c for c in train_df.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
    n_features = len(cols)

    model_idx_list = train_df['wheezy-copper-turtle-magic'].unique()
    model_idx_list = np.sort(model_idx_list)

    # build or load GMM models
    n_components_min = 2
    n_components_max = 10

    gmm_filename = 'gmm_models_rand.pkl'
    features_path = '../features/'

    if kaggle_run:
        build_gmm_dict = True
    else:
        build_gmm_dict = False
        try:
            print('Loading GMM features')
            t0 = time.time()
            gmm_models_dict = load_pickle(gmm_filename, features_path)
            print('Done loading features in {} sec'.format(time.time() - t0))

        except Exception as e:
            print('Failed in loading GMM features, recompute !!')
            build_gmm_dict = True

    if build_gmm_dict:
        print('Building GMM features')
        t0 = time.time()

        # dictionary inside a dictionary
        gmm_models_dict = {}  # indexed by the wheezy number
        for model_idx in model_idx_list:
            gmm_models_dict[model_idx] = {}

        for model_idx in model_idx_list:
            if model_idx % 64 == 0:
                print('working on GMM for model: {}'.format(model_idx))

            local_train_df = train_df[train_df['wheezy-copper-turtle-magic'] == model_idx]
            local_test_df = test_df[test_df['wheezy-copper-turtle-magic'] == model_idx]

            local_X_train = local_train_df[cols].values
            local_X_test = local_test_df[cols].values

            local_train_idx = local_train_df.index
            local_test_idx = local_test_df.index

            vc = VarianceThreshold(threshold=1.5).fit(local_X_train)
            local_X_train = vc.transform(local_X_train)
            local_X_test = vc.transform(local_X_test)

            joint_data = np.concatenate([local_X_train, local_X_test], axis=0)

            # # may be flip the data
            # joint_data_fliped = 2 * joint_data.mean(axis=0) - joint_data
            # joint_data = np.concatenate([
            #     joint_data,
            #     joint_data_fliped], axis=0)

            for n_components in np.arange(n_components_min, n_components_max, 2):
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    n_init=5,
                    init_params='random',  # 'kmeans',  # ,
                    random_state=42)
                gmm.fit(joint_data)
                gmm_models_dict[model_idx][n_components] = gmm
        print('Done building features in {} sec'.format(time.time() - t0))

        if not kaggle_run:
            save_pickle(gmm_models_dict, gmm_filename, features_path, protocol=1)

    # reg_param_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]  # , 0.2, 0.4, 0.6]
    reg_param_list = [0.02, 0.03, 0.04]  # , 0.2, 0.4, 0.6]

    preds = np.zeros((len(test_df), len(reg_param_list)))
    oof = np.zeros((len(train_df), len(reg_param_list)))
    for reg_idx, reg_param in enumerate(reg_param_list):
        print('current reg_param: {}'.format(reg_param))
        model_idx_list = train_df['wheezy-copper-turtle-magic'].unique()
        model_idx_list = np.sort(model_idx_list)
        for model_idx in model_idx_list:
            local_gmm_dict = gmm_models_dict[model_idx]
            local_train_df = train_df[train_df['wheezy-copper-turtle-magic'] == model_idx]
            local_test_df = test_df[test_df['wheezy-copper-turtle-magic'] == model_idx]

            local_X_train = local_train_df[cols].values
            local_y_train = local_train_df['target'].values

            local_X_test = local_test_df[cols].values

            local_train_idx = local_train_df.index
            local_test_idx = local_test_df.index

            vc = VarianceThreshold(threshold=1.5).fit(local_X_train)
            local_X_train = vc.transform(local_X_train)
            local_X_test = vc.transform(local_X_test)
            local_n_train = local_X_train.shape[0]

            embedding_train = [local_X_train]
            embedding_test = [local_X_test]

            for n_components in np.arange(n_components_min, n_components_max, 2):
                current_gmm_model = local_gmm_dict[n_components]
                embedding_train.append(current_gmm_model.predict_proba(local_X_train))
                embedding_test.append(current_gmm_model.predict_proba(local_X_test))

            local_X_train = np.concatenate(embedding_train, axis=1)
            local_X_test = np.concatenate(embedding_test, axis=1)

            # sc = StandardScaler().fit(local_X_train)
            # local_X_train = sc.transform(local_X_train)
            # local_X_test = sc.transform(local_X_test)

            # X_train = data2[:local_train_df.shape[0]]
            # X_test = data2[local_train_df.shape[0]:]

            skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
            for cv_idx, (train_index, val_index) in enumerate(skf.split(local_X_train, local_y_train)):

                # local_X_train2 = embedding[:local_n_train]
                # local_X_test2 = embedding[local_n_train:]

                X_train = local_X_train[train_index, :]
                X_val = local_X_train[val_index, :]

                y_train = local_y_train[train_index]
                y_val = local_y_train[val_index]

                base_clf = QuadraticDiscriminantAnalysis(reg_param=reg_param)
                bagged_clf = BaggingClassifier(
                    base_clf, n_estimators=100,
                    max_samples=0.8, max_features=1.0,
                    bootstrap=True, bootstrap_features=False,
                    random_state=42)
                bagged_clf.fit(X_train, y_train)
                current_predictions_val = bagged_clf.predict_proba(X_val)[:, 1]
                oof[local_train_idx[val_index], reg_idx] = current_predictions_val
                current_predicitons_test = bagged_clf.predict_proba(local_X_test)[:, 1]
                preds[local_test_idx, reg_idx] += current_predicitons_test / skf.n_splits

            if model_idx % 32 == 0:
                print('model_index: {}, reg_idx: {}, reg_param: {}, oof auc : {}'.format(
                    model_idx, reg_idx, reg_param, roc_auc_score(local_y_train, oof[local_train_idx, reg_idx])))

        print('size of training data: {}, reg_idx {}: , reg_param: {}, QDA auc: {}'.format(
            train_df.shape[0], reg_idx, reg_param,
            roc_auc_score(train_df['target'], oof[:, reg_idx])))

    selected_preds = preds.mean(axis=1)

    sub_df = pd.read_csv('../input/sample_submission.csv')
    sub_df['target'] = selected_preds
    sub_df.to_csv('submission.csv', index=False)
    try:
        submission_filename = '../submissions/submission_{}.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
        sub_df.to_csv(submission_filename, index=False)
    except Exception as e:
        print(e)
        pass
