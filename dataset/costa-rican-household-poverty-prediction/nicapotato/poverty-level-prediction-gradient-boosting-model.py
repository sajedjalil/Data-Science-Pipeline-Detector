# Poverty Level Prediction Light Gradient Boosting with CV and Seed Diversification
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
random.seed(2018)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s\n".format(title, time.time() - t0))    

def get_data(Debug = False):
    print("Load Data")
    nrows = None
    if Debug is True: nrows= 500
    train = pd.read_csv("../input/train.csv", index_col = "Id", nrows=nrows)
    traindex = train.index
    test_df = pd.read_csv("../input/test.csv", index_col = "Id", nrows=nrows)
    testdex = test_df.index
    y = train.Target.copy()
    y = y - 1
    train.drop("Target",axis=1,inplace=True)
    
    return train, traindex, test_df, testdex, y

def preprocess(train, test_df, traindex, testdex):
    print("Preprocessing Stage:")
    df = pd.concat([train,test_df],axis=0)
    dfdex = df.index
    # Label Encode
    lbl = preprocessing.LabelEncoder()
    for col in train.loc[:,train.dtypes == "object"].columns:
        df[col] = lbl.fit_transform(df[col].astype(str))
        
    # Create Room Features- https://www.kaggle.com/opanichev/lgb-as-always
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms']
    
    # Average by household ID
    householdagg = df.groupby("idhogar").agg({k:["sum","mean","max","min","std"] for k in train.columns if k not in "idhogar"})
    householdagg.columns = pd.Index(["household_agg_" + e[0] +"_"+ e[1] for e in householdagg.columns.tolist()])
    df = pd.merge(df,householdagg, left_on="idhogar", right_on="idhogar", how= "left")
    df.index = dfdex
    
    # Split
    train = df.loc[traindex,:]
    test_df = df.loc[testdex,:]
    
    return train, test_df

def lgbm_cv(y, lgtrain):
    print("Light Gradient Boosting Classifier: ")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': y.nunique(),
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
    class_prediction = mean_prob.idxmax(axis=1) + 1
    class_prediction.rename("Target",inplace=True)
    class_prediction.index = testdex
    print("Prediction Class Distribution:\n", class_prediction.value_counts(normalize=True))
    print("Dependent Variable Class Distribution:\n", y.value_counts(normalize=True))
    return class_prediction

# Execute All
def main(Debug = False):
    with timer("Load Data"):
        train, traindex, test_df, testdex, y = get_data(Debug=Debug)
    with timer("Pre-Process"):
        train, test_df = preprocess(train, test_df, traindex, testdex)
    with timer("LGBM CV"):
        # LGBM Dataset
        cat_vars = ['hacdor','hacapo','v14a','refrig','v18q','paredblolad','paredzocalo','paredpreb',
            'pareddes','paredmad', 'paredzinc','paredfibras','paredother','pisomoscer','pisocemento',
            'pisoother','pisonatur',  'pisonotiene', 'pisomadera','techozinc','techoentrepiso',
            'techocane','techootro','cielorazo','abastaguadentro','abastaguafuera','abastaguano',
            'public','planpri','noelec','coopele','sanitario1','sanitario2','sanitario3','sanitario5',
            'sanitario6','energcocinar1','energcocinar2','energcocinar3','energcocinar4','elimbasu1','elimbasu2','elimbasu3',
            'elimbasu4','elimbasu5','elimbasu6','epared1','epared2','epared3','etecho1',
            'etecho2','etecho3','eviv1','eviv2','eviv3','dis','male','female','estadocivil1','estadocivil2','estadocivil3','estadocivil4',
            'estadocivil5','estadocivil6','estadocivil7','parentesco1','parentesco2','parentesco3',
            'parentesco4','parentesco5','parentesco6','parentesco7','parentesco8',
            'parentesco9','parentesco10','parentesco11','parentesco12','dependency','edjefe',
            'edjefa','instlevel1','instlevel2','instlevel3','instlevel4','instlevel5',
            'instlevel6','instlevel7','instlevel8','instlevel9','tipovivi1','tipovivi2','tipovivi3','tipovivi4',
            'tipovivi5','computer','television','mobilephone','lugar1','lugar2','lugar3',
            'lugar4','lugar5','lugar6','area1','area2','idhogar']
        lgtrain = lgb.Dataset(train,y ,feature_name = "auto", categorical_feature = cat_vars, free_raw_data=False)
        print("Starting LightGBM. Train shape: {}, Test shape: {}".format(train.shape,test_df.shape))
        lgbm_params, optimal_rounds, best_cv_score = lgbm_cv(y, lgtrain)
    with timer("Seed Diversification"):
        panel = lgbm_seed_diversification(y, lgtrain, train, test_df, lgbm_params, optimal_rounds, best_cv_score)
    with timer("Seed Ensemble"):
        class_prediction = seed_ensemble(y, testdex, panel)
    with timer("Submit"):
        class_prediction.to_csv('seed_mean_sub_rounds_{}_score_{}.csv'.format(optimal_rounds,round(best_cv_score,5))
                    ,index = True, header=True)
        print(class_prediction.head())
        print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

if __name__ == '__main__':
    main(Debug = False)

# Don't Forget to check the LOG! Happy Kaggling.