import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import xgboost as xgb


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]



class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))

#
# Model/pipeline with scaling,pca,svm
#
svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            PCA(),
                                            SVR(kernel='rbf', C=1.0, epsilon=0.05)]))
                                            
results = cross_val_score(svm_pipe, train, y_train, cv=5, scoring='r2')
print("SVM score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()
                                            
#
# Model/pipeline with scaling,pca,ElasticNet
#
en_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                           PCA(n_components=125),
                                           ElasticNet(alpha=0.001, l1_ratio=0.1)]))

#
# XGBoost model
#
xgb_model = xgb.sklearn.XGBRegressor(max_depth=3, learning_rate=0.005, subsample=0.9,
                                     colsample_bytree=0.4, objective='reg:linear', n_estimators=1300)

results = cross_val_score(xgb_model, train, y_train, cv=5, scoring='r2')
print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Random Forest
#
rf_model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                                 min_samples_leaf=25, max_depth=3)

results = cross_val_score(rf_model, train, y_train, cv=5, scoring='r2')
print("RF score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Extra Trees
#
et_model = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35,
                               max_features=150)

results = cross_val_score(et_model, train, y_train, cv=5, scoring='r2')
print("ET score: %.4f (%.4f)" % (results.mean(), results.std()))

# Code below does out-of-fold
# training/predictions and then we combine the final results.
#
# Read here for more explanation (This code was borrowed/adapted) :
#

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                print ("Model %d fold %d score %f" % (i, j, r2_score(y_holdout, y_pred)))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

stack = Ensemble(n_splits=5,
                 #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 stacker=ElasticNet(l1_ratio=0.1, alpha=400), #alpha was 1.4
                 base_models=(svm_pipe, en_pipe, xgb_model, rf_model, et_model))

y_test = stack.fit_predict(train, y_train, test)

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('submission.csv', index=False)