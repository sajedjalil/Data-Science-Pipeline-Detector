# Term Frequency-Inverse Document Frequency, Singular Value Decomposition
# K-Means, Light Gradient Boosting Multi-Class Classifier
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition
from sklearn import cluster

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Model
import lightgbm as lgb
random.seed(2018)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s\n".format(title, time.time() - t0))    

def get_data(Debug = False):
    print("Load Data")
    train = pd.read_json('../input/train.json').set_index('id')
    if Debug is True: train = train.sample(300)
    traindex = train.index
    test_df = pd.read_json('../input/test.json').set_index('id')
    if Debug is True: test_df = test_df.sample(100)
    testdex = test_df.index
    # Label Encoding - Target 
    print ("Label Encode the Target Variable ... ")
    y = train['cuisine'].copy()
    train.drop("cuisine",axis=1,inplace=True)
    
    return train, traindex, test_df, testdex, y

def preprocess(train, test_df, traindex, testdex):
    print("Preprocessing Stage:")
    df = pd.concat([train,test_df],axis=0)
    dfdex = df.index
    # Label Encode
    vect = TfidfVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)
    dummies = vect.fit_transform(df['ingredients'].apply(','.join)) 
    df = pd.DataFrame(dummies.todense(),columns=vect.get_feature_names())
    print("Vocab Length: ", len(vect.get_feature_names()))
    print("All Data Shape: ", df.shape)
    
    # SVD- Dimensionality Reduction
    svd = decomposition.TruncatedSVD(n_components=500, n_iter=10, random_state=42)
    df = svd.fit_transform(df)
    print("After SVD: ",df.shape)
    
    # K-Means- Unsupervized Learning
    kmeans = cluster.KMeans(n_clusters=250, random_state=23)
    kmeans.fit(df)
    kmeans_pred = kmeans.predict(df)
    df = pd.DataFrame(df)
    df["k_means_cluster"] = kmeans_pred
    df.index= dfdex
   
    # Split
    train = df.iloc[0:len(traindex),:]
    test_df = df.iloc[len(traindex):,:]
    print("Train Shape: ", train.shape)
    print("Test Shape: ", test_df.shape)
    
    return train, test_df

def lgbm_cv(y, lgtrain):
    print("Light Gradient Boosting Classifier: ")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(set(y)),
        'metric': ['multi_logloss'],
        "learning_rate": 0.05,
         "num_leaves": 80,
         "max_depth": 6,
         "feature_fraction": 0.70,
         "bagging_fraction": 0.75,
         "reg_alpha": 0.15,
         "reg_lambda": 0.15,
          "min_child_weight": 0,
          "verbose":0
                    }
                    
    modelstart= time.time()
    # Find Optimal Parameters / Boosting Rounds
    lgb_cv = lgb.cv(
        params = lgbm_params,
        train_set = lgtrain,
        num_boost_round=2000,
        stratified=True,
        nfold = 5,
        verbose_eval=100,
        seed = 23,
        early_stopping_rounds=75)

    loss = lgbm_params["metric"][0]
    optimal_rounds = np.argmin(lgb_cv[str(loss) + '-mean'])
    best_cv_score = min(lgb_cv[str(loss) + '-mean'])

    print("\nOptimal Round: {}\nOptimal Score: {} + {}".format(
        optimal_rounds,best_cv_score,lgb_cv[str(loss) + '-stdv'][optimal_rounds]))

    return lgbm_params, optimal_rounds, best_cv_score

def lgbm_seed_diversification(y, lgtrain, train, test_df, lgbm_params, optimal_rounds, best_cv_score):
    print("Seed Diversification Stage:")
    allmodelstart= time.time()
    # Run Model with different Seeds
    multi_seed_pred = dict()
    all_feature_importance_df  = pd.DataFrame()

    # To submit each seed model seperately aswell
    def seed_submit(test_df, model,seed):
        # Output position with highest probability
        class_prediction = (pd.DataFrame(model.predict(test_df)).idxmax(axis=1) + 1).rename('Id')
        class_prediction.index = test_df.index

        # Submit
        class_prediction.to_csv('seed{}_sub_ep{}_sc{}.csv'.format(seed,optimal_rounds,round(best_cv_score,5))
                    ,index = True, header=True)

    all_seeds = [5,8,10,12]
    for seeds_x in all_seeds:
        modelstart= time.time()
        print("Seed: ", seeds_x,)
        # Go Go Go
        lgbm_params["seed"] = seeds_x
        lgb_final = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round = optimal_rounds + 1,
            verbose_eval=200)

        # Feature Importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train.columns
        fold_importance_df["importance"] = lgb_final.feature_importance()
        all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df], axis=0)

        multi_seed_pred[seeds_x] =  pd.DataFrame(lgb_final.predict(test_df))
        # Submit Model Individually
        seed_submit(test_df, model= lgb_final, seed= seeds_x)
        print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
        del lgb_final

    cols = all_feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    best_features = all_feature_importance_df.loc[all_feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgb_finalm_importances.png')
    print("All Model Runtime: %0.2f Minutes"%((time.time() - allmodelstart)/60))

    # Collapse Seed DataFrames
    panel = pd.Panel(multi_seed_pred)
    print("Seed Effect Breakdown: Classwise Statistics")
    for i,(std,mean) in enumerate(zip(panel.std(axis=0).mean(axis=0),panel.mean(axis=0).mean(axis=0))):
        print("Class {}:".format(i+1))
        print("Mean {0:.3f} (+/-) {1:.5f}\n".format(mean,std))
    
    return panel

def seed_ensemble(y, testdex, panel):
    print("Seed Ensemble Stage")
    # Take Mean over Seed prediction
    mean_prob = panel.mean(axis=0)
    # Output position with highest probability
    class_prediction = mean_prob.idxmax(axis=1)
    return class_prediction

# Execute All
def main(Debug = False):
    with timer("Load Data"):
        train, traindex, test_df, testdex, y= get_data(Debug=Debug)
        lb = preprocessing.LabelEncoder()
        lb.fit(y)
        y = lb.transform(y)
    with timer("Pre-Process"):
        train, test_df = preprocess(train, test_df, traindex, testdex)
    with timer("LGBM CV"):
        lgtrain = lgb.Dataset(train,y, categorical_feature= ["k_means_cluster"], free_raw_data=False)
        print("Starting LightGBM.\nTrain shape: {}\nTest shape: {}".format(train.shape,test_df.shape))
        lgbm_params, optimal_rounds, best_cv_score = lgbm_cv(y, lgtrain)
    with timer("Seed Diversification"):
        panel = lgbm_seed_diversification(y, lgtrain, train, test_df, lgbm_params, optimal_rounds, best_cv_score)
    with timer("Seed Ensemble"):
        class_prediction = seed_ensemble(y, testdex, panel)
        class_prediction = pd.Series(lb.inverse_transform(class_prediction))
        class_prediction.rename("cuisine",inplace=True)
        class_prediction.index = testdex
    with timer("Submit"):
        class_prediction.to_csv('seed_mean_sub_rounds_{}_score_{}.csv'.format(optimal_rounds,round(best_cv_score,5))
                    ,index = True, header=True)
        print(class_prediction.head())
        print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

if __name__ == '__main__':
    main(Debug = False)

# Don't Forget to check the LOG! Happy Kaggling.