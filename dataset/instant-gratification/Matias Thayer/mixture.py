import time
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score as roc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.svm import NuSVC
from sklearn.ensemble import ExtraTreesClassifier as ET
from scipy.stats import pearsonr, multivariate_normal, rankdata

import scipy
import lightgbm as lgb
import numpy as np
import pandas as pd

# pylint: disable=invalid-name
# pylint: disable=missing-docstring

SEED = 10
np.random.seed(SEED)
N_FOLD = 12


class SSGaussianMixture():
    def __init__(self, n_features, n_categories):
        self.n_features = n_features
        self.n_categories = n_categories

        self.mus = np.array([np.random.randn(n_features)] * n_categories)
        self.sigmas = np.array([np.eye(n_features)] * n_categories)
        self.pis = np.array([1 / n_categories] * n_categories)

    def fit(self, X_train, y_train, X_test, threshold=0.00001, max_iter=500):
        Z_train = np.eye(self.n_categories)[y_train]

        for i in range(max_iter):
            # EM algorithm
            # M step
            Z_test = np.array(
                [self.gamma(X_test, k) for k in range(self.n_categories)]
            ).T
            Z_test /= Z_test.sum(axis=1, keepdims=True)

            # E step
            datas = [X_train, Z_train, X_test, Z_test]
            mus = np.array([self._est_mu(k, *datas) for k in range(self.n_categories)])
            sigmas = np.array(
                [self._est_sigma(k, *datas) for k in range(self.n_categories)]
            )
            pis = np.array([self._est_pi(k, *datas) for k in range(self.n_categories)])

            diff = max(
                np.max(np.abs(mus - self.mus)),
                np.max(np.abs(sigmas - self.sigmas)),
                np.max(np.abs(pis - self.pis)),
            )

            self.mus = mus
            self.sigmas = sigmas
            self.pis = pis
            if diff < threshold:
                break

    def predict_proba(self, X):
        Z_pred = np.array([self.gamma(X, k) for k in range(self.n_categories)]).T
        Z_pred /= Z_pred.sum(axis=1, keepdims=True)
        return Z_pred

    def gamma(self, X, k):
        # X is input vectors, k is feature index
        return multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])

    def _est_mu(self, k, X_train, Z_train, X_test, Z_test):
        mu = (Z_train[:, k] @ X_train + Z_test[:, k] @ X_test).T / (
            Z_train[:, k].sum() + Z_test[:, k].sum()
        )
        return mu

    def _est_sigma(self, k, X_train, Z_train, X_test, Z_test):
        cmp1 = (
            (X_train - self.mus[k]).T @ np.diag(Z_train[:, k]) @ (X_train - self.mus[k])
        )
        cmp2 = (X_test - self.mus[k]).T @ np.diag(Z_test[:, k]) @ (X_test - self.mus[k])
        sigma = (cmp1 + cmp2) / (Z_train[:, k].sum() + Z_test[:k].sum())
        return sigma

    def _est_pi(self, k, X_train, Z_train, X_test, Z_test):
        pi = (Z_train[:, k].sum() + Z_test[:, k].sum()) / (Z_train.sum() + Z_test.sum())
        return pi


def ensemble_predictions(predictions, weights, type_="linear"):
    assert np.isclose(np.sum(weights), 1.0)
    if type_ == "linear":
        res = np.average(predictions, weights=weights, axis=0)
    elif type_ == "harmonic":
        res = np.average([1 / p for p in predictions], weights=weights, axis=0)
        return 1 / res
    elif type_ == "geometric":
        numerator = np.average(
            [np.log(p) for p in predictions], weights=weights, axis=0
        )
        res = np.exp(numerator / sum(weights))
        return res
    elif type_ == "rank":
        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)
        return res / (len(res) + 1)
    return res


if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    cols = [
        c
        for c in train.columns
        if c not in ["id", "target", "wheezy-copper-turtle-magic"]
    ]
    oof_qda = np.zeros(len(train))
    pred_te_qda = np.zeros(len(test))

    for i in range(512):
        train2 = train[train["wheezy-copper-turtle-magic"] == i]
        test2 = test[test["wheezy-copper-turtle-magic"] == i]
        idx1 = train2.index
        idx2 = test2.index
        train2.reset_index(drop=True, inplace=True)
        target = train2["target"]

        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
        data2 = StandardScaler().fit_transform(
            VarianceThreshold(threshold=2).fit_transform(data[cols])
        )
        train3, test3 = data2[: train2.shape[0]], data2[train2.shape[0] :]

        folds = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=11)
        for train_index, test_index in folds.split(train2, train2["target"]):
            clf = QuadraticDiscriminantAnalysis(0.5)
            clf.fit(train3[train_index, :], train2.loc[train_index]["target"])
            oof_qda[idx1[test_index]] = clf.predict_proba(train3[test_index, :])[:, 1]
            pred_te_qda[idx2] += clf.predict_proba(test3)[:, 1] / N_FOLD

    print("qda pseudo label", roc(train["target"], oof_qda))
    test["target"] = pred_te_qda
    test.loc[test["target"] > 0.99, "target"] = 1
    test.loc[test["target"] < 0.01, "target"] = 0

    usefull_test = test[(test["target"] == 1) | (test["target"] == 0)]
    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)
    new_train.loc[oof_qda > 0.99, "target"] = 1
    new_train.loc[oof_qda < 0.01, "target"] = 0

    oof_qda = np.zeros(len(train))
    oof_qda2 = np.zeros(len(train))
    oof_svnu = np.zeros(len(train))
    pred_te_qda = np.zeros(len(test))
    pred_te_qda2 = np.zeros(len(test))
    pred_te_svnu = np.zeros(len(test))
    for i in range(512):
        t0 = time.time()
        train2 = new_train[new_train["wheezy-copper-turtle-magic"] == i]
        test2 = test[test["wheezy-copper-turtle-magic"] == i]
        idx1 = train[train["wheezy-copper-turtle-magic"] == i].index
        idx2 = test2.index
        train2.reset_index(drop=True, inplace=True)

        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
        data2 = StandardScaler().fit_transform(
            VarianceThreshold(threshold=2).fit_transform(data[cols])
        )
        train3, test3 = data2[: train2.shape[0]], data2[train2.shape[0] :]

        target = train2["target"].astype(int)
        skf = StratifiedKFold(n_splits=N_FOLD, random_state=42)
        for train_index, test_index in skf.split(train2, train2["target"]):
            oof_test_index = [t for t in test_index if t < len(idx1)]
            clf = QuadraticDiscriminantAnalysis(0.5)
            clf.fit(train3[train_index, :], train2.loc[train_index]["target"])
            if len(oof_test_index) > 0:
                oof_qda[idx1[oof_test_index]] = clf.predict_proba(
                    train3[oof_test_index, :]
                )[:, 1]
            pred_te_qda[idx2] += clf.predict_proba(test3)[:, 1] / skf.n_splits

            clf = SSGaussianMixture(n_features=train3.shape[1], n_categories=2)
            clf.fit(train3[train_index, :], target[train_index], test3)
            if len(oof_test_index) > 1:
                oof_qda2[idx1[oof_test_index]] = clf.predict_proba(
                    train3[oof_test_index, :]
                )[:, 1]
            else:
                oof_qda2[idx1[oof_test_index]] = oof_qda[idx1[oof_test_index]]

            pred_te_qda2[idx2] += clf.predict_proba(test3)[:, 1] / skf.n_splits

            clf = NuSVC(
                probability=True,
                kernel="poly",
                degree=4,
                gamma="auto",
                random_state=48,
                nu=0.59,
                coef0=0.053,
                max_iter=9000,
            )
            clf.fit(train3[train_index, :], train2.loc[train_index]["target"])
            if len(oof_test_index) > 0:
                oof_svnu[idx1[oof_test_index]] = clf.predict_proba(
                    train3[oof_test_index, :]
                )[:, 1]
            pred_te_svnu[idx2] += clf.predict_proba(test3)[:, 1] / skf.n_splits

        if i < 4 or i % 100 == 0:
            print(f"qda_{i}", roc(train["target"].iloc[idx1], oof_qda[idx1]))
            print(f"qda2_{i}", roc(train["target"].iloc[idx1], oof_qda2[idx1]))
            print(f"svcnu_{i}", roc(train["target"].iloc[idx1], oof_svnu[idx1]))
            t1 = time.time()
            print(f"time iteration {i}:", t1 - t0)
    auc = roc(train["target"], oof_qda)
    print(f"AUC qda: {auc:.5}")
    auc = roc(train["target"], oof_qda2)
    print(f"AUC qda2: {auc:.5}")
    auc = roc(train["target"], oof_svnu)
    print(f"AUC svnu: {auc:.5}")

    weights = [0.35, 0.35, 0.3]
    oof_ens = ensemble_predictions([oof_qda, oof_qda2, oof_svnu], weights, type_="rank")
    pred_te_ens = ensemble_predictions(
        [pred_te_qda, pred_te_qda2, pred_te_svnu], weights, type_="rank"
    )
    auc = roc(train["target"], oof_ens)
    print(f"AUC blend: {auc:.5}")

    oof_lrr = np.zeros(len(train))
    pred_te_lrr = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLD, random_state=42)
    tr = np.concatenate(
        (oof_svnu.reshape(-1, 1), oof_qda.reshape(-1, 1), oof_qda2.reshape(-1, 1)),
        axis=1,
    )
    te = np.concatenate(
        (
            pred_te_svnu.reshape(-1, 1),
            pred_te_qda.reshape(-1, 1),
            pred_te_qda2.reshape(-1, 1),
        ),
        axis=1,
    )
    print(tr.shape, te.shape)
    for tr_idx, val_idx in skf.split(tr, train["target"]):
        # lrr = svm.NuSVC(probability=True, kernel='poly', degree=2, gamma='auto', random_state=42, nu=0.6, coef0=0.6)
        lrr = linear_model.LogisticRegression(
            solver="liblinear"
        )  # solver='liblinear',penalty='l1',C=0.1
        # lrr = BayesianRidge(normalize=False, n_iter=1000)
        lrr.fit(tr[tr_idx], train["target"][tr_idx])
        oof_lrr[val_idx] = lrr.predict_proba(tr[val_idx, :])[:, 1]
        pred_te_lrr += lrr.predict_proba(te)[:, 1] / skf.n_splits

    score = round(roc(train["target"], oof_lrr), 6)
    print("stack CV score lr =", score)

    weights = [0.8, 0.2]
    oof_ens_stack = ensemble_predictions([oof_ens, oof_lrr], weights, type_="rank")
    score = round(roc(train["target"], oof_ens_stack), 6)
    print("stack + blend CV score =", score)
    pred_te_ens_stack = ensemble_predictions(
        [pred_te_ens, pred_te_lrr], weights, type_="rank"
    )

    sub = pd.read_csv("../input/sample_submission.csv")
    sub["target"] = pred_te_ens_stack
    sub.to_csv("submission.csv", index=False)
