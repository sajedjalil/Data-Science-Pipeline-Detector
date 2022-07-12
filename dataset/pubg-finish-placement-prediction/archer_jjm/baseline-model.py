import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge

train_data = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv",encoding="utf-8")
test_data = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv",encoding="utf-8")

select_cols = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","kills","longestKill","revives",
               "rideDistance","swimDistance","walkDistance","weaponsAcquired"]

train_data.dropna(axis=0,how='any',inplace =True)

select_train_data = train_data[select_cols]
select_test_data = test_data[select_cols]

pca = PCA(n_components=14)

def pca_train_data(train_data):
    train_data_scaled=pca.fit_transform(train_data)
    return train_data_scaled

def pca_test_data(test_data):
    test_data_scaled=pca.transform(test_data)
    return test_data_scaled

train_data_processed = pca_train_data(select_train_data)
test_data_processed = pca_test_data(select_test_data)

X_train = train_data_processed
y_train = train_data["winPlacePerc"]
X_test = test_data_processed

RFR = RandomForestRegressor(n_estimators=90,min_samples_leaf=3, max_features='sqrt',n_jobs=-1,verbose=1)


RFR.fit(X_train,y_train)

predictions = np.clip(a = RFR.predict(test_data_processed), a_min = 0.0, a_max = 1.0)
pred_df = pd.DataFrame({'Id' : test_data['Id'], 'winPlacePerc' : predictions})
pred_df.to_csv("submission.csv", index=False)