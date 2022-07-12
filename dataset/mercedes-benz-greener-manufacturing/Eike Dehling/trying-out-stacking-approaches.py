import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

class StackingCVRegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        for i, clf in enumerate(self.regressors):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)

                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)
        

class StackingCVRegressorRetrained(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5, use_features_in_secondary=False):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        # Create out-of-fold predictions for training meta-model
        for i, regr in enumerate(self.regr_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(regr)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])

        # Train meta-model
        if self.use_features_in_secondary:
            self.meta_regr_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_regr_.fit(out_of_fold_predictions, y)
        
        # Retrain base models on all data
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])

        if self.use_features_in_secondary:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(meta_features)
        
class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        
        # Train base models
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        return np.mean(predictions, axis=1)

en = make_pipeline(RobustScaler(), SelectFromModel(Lasso(alpha=0.03)), ElasticNet(alpha=0.001, l1_ratio=0.1))
    
rf = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)
                           
et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)

xgbm = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.9, base_score=y_mean,
                                objective='reg:linear', n_estimators=1000)
                           
stack_avg = StackingCVRegressorAveraged((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

stack_with_feats = StackingCVRegressorRetrained((en, rf, et), xgbm, use_features_in_secondary=True)

stack_retrain = StackingCVRegressorRetrained((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))

averaged = AveragingRegressor((en, rf, et, xgbm))

results = cross_val_score(en, train.values, y_train, cv=5, scoring='r2')
print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(rf, train.values, y_train, cv=5, scoring='r2')
print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(et, train.values, y_train, cv=5, scoring='r2')
print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(xgbm, train.values, y_train, cv=5, scoring='r2')
print("XGBoost score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(averaged, train.values, y_train, cv=5, scoring='r2')
print("Averaged base models score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(stack_with_feats, train.values, y_train, cv=5, scoring='r2')
print("Stacking (with primary feats) score: %.4f (%.4f)" % (results.mean(), results.std()))

results = cross_val_score(stack_retrain, train.values, y_train, cv=5, scoring='r2')
print("Stacking (retrained) score: %.4f (%.4f)" % (results.mean(), results.std()))
                 
results = cross_val_score(stack_avg, train.values, y_train, cv=5, scoring='r2')
print("Stacking (averaged) score: %.4f (%.4f)" % (results.mean(), results.std()))
