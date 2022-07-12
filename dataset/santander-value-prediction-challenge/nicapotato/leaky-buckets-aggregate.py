# Catboost and Light Gradient Boost - Regressor Ensemble with Giba's Leaky Feature
# By Nick Brooks, July 2018
# https://www.kaggle.com/nicapotato

import time
notebookstart= time.time()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import random
from contextlib import contextmanager
from sklearn import preprocessing

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Model
import lightgbm as lgb

#CatBoost
import hyperopt 
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from catboost import cv as catcv

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

random.seed(2018)

id_col = "ID"
target_var = "target"
Debug = True

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s\n".format(title, time.time() - t0))    

# Specify index/ target name
def get_data(id_col, target_var,Debug = False):
    print("Load Data")
    nrows = None
    if Debug is True: nrows= 100
    train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv", index_col = id_col, nrows=nrows)
    train["log_compiled_leak"] = np.log1p(pd.read_csv("../input/breaking-lb-fresh-start-with-lag-selection/train_leak.csv", nrows=nrows)["compiled_leak"].values)
    traindex = train.index
    test_df = pd.read_csv("../input/santander-value-prediction-challenge/test.csv", index_col = id_col, nrows=nrows)
    test_df["log_compiled_leak"] = np.log1p(pd.read_csv("../input/breaking-lb-fresh-start-with-lag-selection/test_leak.csv", nrows=nrows)["compiled_leak"].values)
    testdex = test_df.index
    y = np.log1p(train[target_var]).copy()
    train.drop(target_var,axis=1,inplace=True)
    print('Train shape: {} Rows, {} Columns'.format(*train.shape))
    print('Test shape: {} Rows, {} Columns'.format(*test_df.shape))
    
    return train, traindex, test_df, testdex, y

def feature_engineering(train, test_df):
    # Combine Datasets for Processing
    df = pd.concat([train, test_df], axis = 0 )
    log_compiled_leak_array = df["log_compiled_leak"].copy()
    df.drop(["log_compiled_leak"], axis =1, inplace = True)

    # check and remove constant columns
    colsToRemove = []
    for col in train.columns:
        if col != 'ID' and col != 'target':
            if train[col].std() == 0: 
                colsToRemove.append(col)
    df.drop(colsToRemove, axis = 1, inplace= True)
    print("Columns Dropped: ", len(colsToRemove))

    # Scaling
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df.values)

    # Decomposition
    pca = PCA(random_state=23, n_components = 100)
    df = pd.DataFrame(pca.fit_transform(df))

    # Add the leak back
    df.loc[:, "log_compiled_leak"] = log_compiled_leak_array.values
    # Leaky Buckets
    n_buckets = 10
    labels = [i for i in range(n_buckets)]
    df["leaky_buckets"] = pd.cut(df["log_compiled_leak"], n_buckets, labels = labels)

    # Percentile Function
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    # Aggregate By Bucket
    leaky_buckets_agg = df.groupby('leaky_buckets').agg(["sum","mean","max","min","std","skew",percentile(80),percentile(20)])
    leaky_buckets_agg.columns = pd.Index([str(e[0]) +"_"+ str(e[1]) for e in leaky_buckets_agg.columns.tolist()])
    leaky_buckets_agg.head()
    df = pd.merge(df,leaky_buckets_agg, on="leaky_buckets", how= "left")
    
    # Split
    train = df.iloc[0:len(train), :]
    test_df = df.iloc[len(train):, :]

    return train, test_df

def lgbm_cv(y, lgtrain):
    print("Light Gradient Boosting Regressor: ")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        "learning_rate": 0.01,
        "num_leaves": 180,
        "feature_fraction": 0.50,
        "bagging_fraction": 0.50,
        'bagging_freq': 4,
        "max_depth": 5,
        "reg_alpha": 1,
        "reg_lambda": 0.1,
        "min_child_weight":10,
        'zero_as_missing':True
                    }
                    
    modelstart= time.time()
    # Find Optimal Parameters / Boosting Rounds
    lgb_cv = lgb.cv(
        params = lgbm_params,
        train_set = lgtrain,
        num_boost_round=2000,
        stratified=False,
        nfold = 5,
        verbose_eval= None,
        seed = 23,
        early_stopping_rounds=75)

    loss = lgbm_params["metric"]
    optimal_rounds = np.argmin(lgb_cv[str(loss) + '-mean'])
    best_cv_score = min(lgb_cv[str(loss) + '-mean'])

    print("\nOptimal Round: {}\nOptimal Score: {} + {}".format(
        optimal_rounds,best_cv_score,lgb_cv[str(loss) + '-stdv'][optimal_rounds]))

    return lgbm_params, optimal_rounds, best_cv_score

def lgbm_seed_diversification(y, lgtrain, train, test_df, lgbm_params, optimal_rounds, best_cv_score, target_var, testdex):
    print("Seed Diversification Stage:")
    allmodelstart= time.time()
    # Run Model with different Seeds
    multi_seed_pred = dict()
    all_feature_importance_df  = pd.DataFrame()
    
    all_seeds = [27,22,300,401]
    for seeds_x in all_seeds:
        modelstart= time.time()
        print("Seed: ", seeds_x,)
        # Go Go Go
        lgbm_params["seed"] = seeds_x
        lgb_reg = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round = optimal_rounds + 1,
            verbose_eval=None)

        # Feature Importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train.columns
        fold_importance_df["importance"] = lgb_reg.feature_importance()
        all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df], axis=0)

        multi_seed_pred[seeds_x] =  list(lgb_reg.predict(test_df))
        print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
        del lgb_reg

    cols = all_feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    best_features = all_feature_importance_df.loc[all_feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgb_finalm_importances.png')
    
    seed_preds = pd.DataFrame.from_dict(multi_seed_pred)
    lgbmpred = np.expm1(seed_preds.mean(axis=1).rename(target_var))
    lgbmpred.index = testdex
    lgbmpred.to_csv('seed_mean_sub_rounds_{}_score_{}.csv'.format(optimal_rounds,round(best_cv_score,5))
                    ,index = True, header=True)
    print("All Model Runtime: %0.2f Minutes"%((time.time() - allmodelstart)/60))
    
    return lgbmpred

# CatBoost Model
def catboost(train, y, test_df):
    cat_params = {"eval_metric":'RMSE',
               "iterations": 550,
               "random_seed": 42,
               "logging_level": "Verbose",
               "metric_period": 75,
             }
    print("Train Submission Model")
    model = CatBoostRegressor(**cat_params)
    model.fit(train,y)
    catpred = np.expm1(model.predict(test_df))
    
    return catpred

# Execute All
def main(Debug = False):
    id_col = "ID"
    target_var = "target"
    with timer("Load Data"):
        train, traindex, test_df, testdex, y = get_data(Debug=Debug, id_col = id_col, target_var = target_var)
    with timer("Pre-Processing"):
        train, test_df = feature_engineering(train, test_df)
    with timer("LGBM CV"):
        lgtrain = lgb.Dataset(train,y ,feature_name = "auto")
        print("Starting LightGBM. Train shape: {}, Test shape: {}".format(train.shape,test_df.shape))
        lgbm_params, optimal_rounds, best_cv_score = lgbm_cv(y, lgtrain)
    with timer("Seed Diversification"):
        lgbmpred = lgbm_seed_diversification(y, lgtrain, train, test_df, lgbm_params, optimal_rounds, best_cv_score, target_var, testdex)
        print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

if __name__ == '__main__':
    main(Debug = False)

# Don't Forget to check the LOG! Happy Kaggling.