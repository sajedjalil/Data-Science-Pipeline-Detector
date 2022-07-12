# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn import svm, tree, metrics
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from copy import deepcopy


def clf_training_method(clf, features_train, features_test, labels_train, labels_test):
    print("####### TRAIN & PREDICT #######")
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = metrics.accuracy_score(labels_test, pred)
    print("The prediction accuracy of classification is [%f]." % acc)

    print("-> DONE [TRAIN & PREDICT].\n")

    return acc


def clf_cross_validation(clf, features, labels):
    print("####### CROSS VALIDATION #######")

    auc_arr_scores = cross_val_score(clf, features, labels, scoring='roc_auc', cv=5)
    auc_acc = auc_arr_scores.mean()

    print("The prediction accuracy of classification is [%f]." % auc_acc)

    print("-> DONE [CROSS VALIDATION].\n")
    
    return auc_acc


def clf_train_submit(clf, features, labels, tests):
    print("####### TRAIN & PREDICT #######")

    clf.fit(features, labels)
    pred = clf.predict_proba(tests)

    print("-> DONE [TRAIN & PREDICT].\n")

    return pred


def check_values(data):
    columns = ["var_" + str(i) for i in range(200)]

    big_values = []
    small_values = []
    for col in columns:
        big_values.append(data.loc[data[col].idxmax()][col])
        small_values.append(data.loc[data[col].idxmin()][col])

    print("MAX VALUE:", max(big_values))
    print("MIN VALUE:", min(small_values))

def get_columns(data):
    col_list = list(data)
    if 'ID_code' in col_list:
        col_list.remove('ID_code')
    
    if 'target' in col_list:
        col_list.remove('target')
    
    return col_list

def find(arr):
    v = [x for x in arr if x < 0]
    
    return len(v)

def adjust_features(data):
    col_list = get_columns(data)
    for col in col_list:
        data[col] *= -1
    #data["sum_up"] = data[col_list].sum(axis=1)
    #data["average"] = data[col_list].sum(axis=1) / len(col_list)
    # print(data.head())

    """for col in col_list:
        new_col = col + "_squared"
        data[new_col] = data[col] ** 2

    data = data.drop(col_list, axis=1)
    
    col_list = get_columns(data)

    #data["sum_up_squared"] = data[col_list].sum(axis=1)
    print(data.head())"""
    #print(list(data))


def load_dataset(csv_file, submit=False):
    print("####### DATA PROCESSING #######")
    data = pd.read_csv(csv_file)

    if not submit:
        data = data.sample(frac=0.1)

    print("There are %d rows to process in dataset." % data.shape[0])

    # print(data.head())

    missing_values = data.isnull().sum()
    # print(missing_values)
    missing_values = missing_values[missing_values > 0] / data.shape[0]
    print("Percent of missing values\n%s\n" % (missing_values * 100 if len(missing_values) > 0 else 0))

    if len(missing_values) > 0:
        # replace NaN values with 0
        data = data.fillna(0)

    # print(data.dtypes)

    check_values(data)

    if not submit:
        if "target" in data:
            print("# of 0 in target: %d" % list(data["target"]).count(0))
            print("# of 1 in target: %d" % list(data["target"]).count(1))
    
    adjust_features(data)

    print("-> DONE [DATA PROCESSING].\n")

    return data


def train_classifier(train_data, test_data, submit=False, use_lgbm=False):
    print("####### PREPARE TRAIN / VALIDATION DATA #######")

    # adjust the classifier using train.csv
    # set initial features and labels
    features = train_data.drop(['ID_code', 'target'], axis=1)
    labels = train_data['target']
    
    random_state = 42
    np.random.seed(random_state)

    if not submit:
        # train / test split
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                    test_size=0.25, random_state=42)

        # standardization of training data
        scaler = StandardScaler().fit(features_train)
        features_train = scaler.transform(features_train)
        features_test = scaler.transform(features_test)
        
        if use_lgbm:
            scaler = StandardScaler().fit(features)
            features = scaler.transform(features)

        # print(features_train)
        # print(features_test)
    else:
        orig_test_data = deepcopy(test_data)
        test_data = test_data.drop(['ID_code'], axis=1)
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)
        tests = scaler.transform(test_data)

    print("-> DONE [PREPARE TRAIN / VALIDATION DATA].\n")

    print("####### PREPARE CLASSIFIER #######")
    
    if not use_lgbm:

        param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "n_estimators": [i * 10 for i in [1, 10, 100, 1000]],
                      "base_estimator__min_samples_split": [i * 10 for i in [1, 5, 8, 10]],
                      "base_estimator__min_samples_leaf": [i * 10 for i in [1, 5, 8, 10]],
                      "base_estimator__max_depth": [None, 1, 2, 3, 4, 5]
                      }
    
        """clf = AdaBoostClassifier(
            base_estimator=tree.DecisionTreeClassifier(criterion="entropy", splitter="random",
                                                       min_samples_split=80, min_samples_leaf=80, max_depth=1,
                                                       ), n_estimators=1000, learning_rate=1)
    
        clf = AdaBoostClassifier(
            base_estimator=LinearSVC(max_iter = 10000), algorithm='SAMME', n_estimators=100, learning_rate=1)"""
    
        """clf = AdaBoostClassifier(
            base_estimator=GaussianNB(var_smoothing=1e-15), algorithm='SAMME', n_estimators=1000, learning_rate=1, random_state=99999)"""
    
        clf = BaggingClassifier(
            base_estimator=GaussianNB(var_smoothing=1e-15), n_estimators=100, random_state=99999, n_jobs = -1, bootstrap_features = True, oob_score = True)

        """clf = AdaBoostClassifier(
            base_estimator=LogisticRegression(solver='newton-cg', max_iter=100, n_jobs=-1, tol=1e-4, C=1), 
                                                algorithm='SAMME.R', n_estimators=100, learning_rate=1)"""
    
        # clf = GradientBoostingClassifier(min_samples_split=80, min_samples_leaf=80, n_estimators=100, learning_rate=1)
    
        # clf = tree.DecisionTreeClassifier()
    
        """clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=100, learning_rate=1)
        grid_search_ABC = GridSearchCV(clf, param_grid=param_grid, scoring='roc_auc', cv=5)
        grid_search_ABC.fit(features, labels)"""
    
    else:
        params = {
    "objective" : "binary", "metric" : "auc", "boosting": 'gbdt', "max_depth" : -1, "num_leaves" : 13,
    "learning_rate" : 0.01, "bagging_freq": 5, "bagging_fraction" : 0.4, "feature_fraction" : 0.05,
    "min_data_in_leaf": 80, "tree_learner": "serial", "boost_from_average": "true",
    "bagging_seed" : random_state, "verbosity" : 1, "seed": random_state
    }
        train_data = lgb.Dataset(features, label=labels)
        num_round = 10000

    if submit:
        if not use_lgbm:
            pred = clf_train_submit(clf, features, labels, tests)
        else:
            bst = lgb.train(params, train_data, num_round)
            pred = bst.predict(tests)
        submission = pd.DataFrame({"ID_code": orig_test_data.ID_code.values})
        if not use_lgbm:
            submission["target"] = pred[:, 1]
        else:
            submission["target"] = pred[:]
        submission.to_csv("submission.csv", index=False)
    else:
        
        if use_lgbm:
            print(lgb.cv(params, train_data, num_round, nfold=5)['auc-mean'][-1])
        else:
            # used for tweaking
            clf_training_method(clf, features_train, features_test, labels_train, labels_test)
            # clf_cross_validation(clf, features, labels)

            # print("###BEST###")
            # print(grid_search_ABC.best_params_)

    print("-> DONE [PREPARE CLASSIFIER].\n")


def main():
    submit = True
    use_lgbm = True
    train_data = load_dataset("../input/train.csv", submit=submit)
    test_data = None
    test_data = load_dataset("../input/test.csv", submit=submit)
    train_classifier(train_data, test_data=test_data, submit=submit, use_lgbm=use_lgbm)


if __name__ == '__main__':
    main()
