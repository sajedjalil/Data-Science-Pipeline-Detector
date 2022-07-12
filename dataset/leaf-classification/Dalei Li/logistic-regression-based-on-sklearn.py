# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import math
import pickle

DATA_FOLDER = "../input/"

FIT_LOG_REG = False

FIT_GRID_SEARCH_LOG_REG = True

FIT_GRID_SEARCH_SVC = not FIT_GRID_SEARCH_LOG_REG

WRITE_TEST_PREDICTION = True


def neg_cross_entropy(estimator, X, y):
    """
    define grid search evaluator (scoring function)
    THe larger, the better. Hence, return the negative value.
    :param estimator:
    :param X:
    :param y:
    :return: cross entropy
    """
    x_entropy = 0.0
    pred_proba = estimator.predict_proba(X)
    for prob, label in zip(pred_proba, y):
        x_entropy += math.log(max(min(1 - 1e-15, prob[int(label)]), 1e-15))

    return x_entropy / len(y)

# read train data
# train = []
train_X, train_y = [], []
with open(DATA_FOLDER + "train.csv", "r") as f:
    fieldnames = f.readline().splitlines()[0].split(",")
    for line in f:
        cols = line.splitlines()[0].split(",")
#         train.append(cols)
        train_X.append(cols[2:])
        train_y.append(cols[1])
f.close()

# train[0:20]
print("train X: {}\ntrain y: {}".format(train_X[0:2], train_y[0:2]))
# to numpy array
train_X_np = np.array(train_X, dtype=np.float64)

# read unique labels
with open(DATA_FOLDER + "sample_submission.csv", "r") as f:
    unique_labels = f.readline().splitlines()[0].split(",")[1:]
f.close()
print("unique_labels: {}".format(unique_labels))

# transform labels to index
label_encoder = LabelEncoder()
label_encoder.fit(unique_labels)
print("label encoder classes: {}".format(label_encoder.classes_))

train_y_indexed = label_encoder.transform(train_y)
print("train_y_indexed: {}".format(train_y_indexed[0:2]))

train_y_np = np.array(train_y_indexed, dtype=np.float32)

# read test data
test_ins_ids, test_X = [], []
with open(DATA_FOLDER + "test.csv", "r") as f:
        test_cols = f.readline().splitlines()[0].split(",")
        for line in f:
            columns = line.splitlines()[0].split(",")
            test_ins_ids.append(columns[0])
            test_X.append(columns[1:])
f.close()
print("test X: {}".format(test_X[0:2]))
test_X_np = np.array(test_X, dtype=np.float64)

# scale train X
X_scaler = StandardScaler()
train_X_scaler = X_scaler.fit_transform(train_X_np)
test_X_scaler = X_scaler.transform(test_X_np)

# fit logistic regression
if FIT_LOG_REG:
    log_reg = LogisticRegression(penalty="l2", max_iter=400, solver="lbfgs")
    log_reg.fit(train_X_scaler, train_y_np)

    predictions = log_reg.predict(test_X_scaler)
    print("prediction: {}".format(predictions[0:2]))

    predict_proba = log_reg.predict_proba(test_X_scaler)
    print("predict_proba: {}".format(predict_proba[0:2]))

    print("training accuracy: {}".format(log_reg.score(train_X_scaler, train_y_np)))

    print("training cross entropy: {}".format(neg_cross_entropy(log_reg, train_X_scaler, train_y_np)))

if FIT_GRID_SEARCH_LOG_REG:
    log_reg = LogisticRegression(penalty="l2", max_iter=400, solver="lbfgs")
    # build grid search hyper-parameter grid
    param_grid = {
        "multi_class": ["multinomial"],
        "C": np.linspace(15000, 20000, num=3)
    }

    grid_search = GridSearchCV(log_reg, param_grid=param_grid, scoring="neg_log_loss", n_jobs=-1, cv=10)

    grid_search.fit(train_X_scaler, train_y_indexed)

    print("mean test scores: {}\nparas: {}".format(grid_search.cv_results_["mean_test_score"],
                                                   grid_search.cv_results_["params"]))

    print("best score: {}".format(grid_search.best_score_))

    test_pred_proba = grid_search.predict_proba(test_X_scaler)
    print("test_pred_proba: {}".format(test_pred_proba[0:2]))

    # with open(DATA_FOLDER + "test_pred_proba.pickle", "wb") as f:
    #     pickle.dump(test_pred_proba, f)
    # f.close()


if FIT_GRID_SEARCH_SVC:
    svc_clf = SVC(probability=True)
    # build grid search hyper-parameter grid
    param_grid = {
        "kernel": ["linear"],
        "C": np.linspace(1.0, 100.0, num=1),
        "gamma": np.linspace(0.00001, 0.01, num=10)
    }

    grid_search = GridSearchCV(svc_clf, param_grid=param_grid, scoring=neg_cross_entropy, n_jobs=-1, cv=10)

    grid_search.fit(train_X_scaler, train_y_indexed)

    print("mean test scores: {}\nparas: {}".format(grid_search.cv_results_["mean_test_score"],
                                                   grid_search.cv_results_["params"]))
    print("best score: {}".format(grid_search.best_score_))

    test_pred_proba = grid_search.predict_proba(test_X_scaler)
    print("test_pred_proba: {}".format(test_pred_proba[0:2]))

    # with open(DATA_FOLDER + "test_pred_proba_svc.pickle", "wb") as f:
    #     pickle.dump(test_pred_proba, f)
    # f.close()

if WRITE_TEST_PREDICTION:
    # write to csv
    with open("test_pred.csv", "w") as f:
        # write header
        f.write("id," + ",".join(unique_labels) + "\n")
        # write data
        for ins_id, prob in zip(test_ins_ids, test_pred_proba):
            f.write(str(ins_id) + "," + ",".join(["{:.18f}".format(float(val)) for val in prob]) + "\n")
    f.close()