import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

######
# Pre-processing
######
training = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train = reduce_mem_usage(training)
test_X = reduce_mem_usage(test) # Target not included.


######
# EDA
#######

# train.head(25)
# train.dtypes
# train.isnull().values.any()
# train[train.isna().any(axis=1)]
# train.dropna(how="any", inplace=True) # training data contains single null value for winPlacePerc

# train.dtypes
# Here we can see Id, groupId, matchId and matchType are non-numeric variables
# matchType being a categoric feature. Lets explore the unique match types in training data:


# Here we can see types include Squad, Duo and Solo each of which can be played in first person or 3rd person modes. 
# However we can also notice interesting values like "crashfpp" and "flarefpp" which represents different game types. 
# such as crash fpp is a game mode that allows damage delt to players through vehicles. 
# train.matchType.unique()
# np.sum(train["matchType"] == 'flarefpp')

# Now we want to explore what % of gametypes each of these catagories consist of:
# train.groupby("matchType").count()["Id"]
# percAllGameTypes = train.groupby("matchType").count()["Id"] / train["Id"].count() * 100
# percAllGameTypes.sort_values(ascending=False)

# interestingly we can see the Top 6 values of matchtypes by % are:
# squad-fpp           39.491788
# duo-fpp             22.412837
# squad               14.088845
# solo-fpp            12.070277
# duo                  7.051798
# solo                 4.091397

def simplify_matchtypes(value):
    if('squad' in value):
        return 'squad'
    elif('duo' in value):
        return 'duo'
    elif('solo' in value):
        return 'solo'
    elif('crash' in value): 
        return 'crash' 
    elif('flare' in value):
        return 'flare'
    else:
        return 'unknown' # catch out any other game types that may appear on test data.

train["matchTypeSimplified"] = train["matchType"].apply(lambda c: simplify_matchtypes(c))
perc_all_game_types = train.groupby("matchTypeSimplified").count()["Id"] / train["Id"].count() * 100
perc_all_game_types.sort_values(ascending=False)

# All 5-values now
# matchTypeSimplified
# squad    53.978432
# duo      29.592542
# solo     16.206829
# crash     0.149720
# flare     0.072476

def one_hot_encode_match_type(matchTypeName):
    train[matchTypeName] = train["matchTypeSimplified"].apply(lambda c: 1 if c == matchTypeName else 0)

one_hot_encode_match_type('squad')
one_hot_encode_match_type('solo')
one_hot_encode_match_type('duo')

# f,ax = plt.subplots(figsize=(20, 20))
# matrix = np.triu(train.corr())
# sns.heatmap(train.corr().sort_values(by = ["winPlacePerc"]), annot=True, mask=matrix,vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, ax=ax)
# plt.show()

#Correlation with output variable
# pearson_corr = train.corr()
# pearson_corr_target = cor["winPlacePerc"].sort_values(ascending=False)
# pearson_corr_target.plot.bar(figsize=(15, 10), align="center")


# # Using lasso regularization with Linear regression to determing important features
# # scikit learn's LassoCV selects the best model using Cross Validation.
# from sklearn.preprocessing import StandardScaler
# y = train["winPlacePerc"]
# X = train.drop(["Id", "groupId", "matchId", "matchType", "matchTypeSimplified", "winPlacePerc"], axis=1)
# scaler = StandardScaler() 
# scaler.fit(X)
# X = scaler.transform(X)

# lasso_reg = LassoCV()
# lasso_reg.fit(X, y)
# X2 = train.drop(["Id", "groupId", "matchId", "matchType", "matchTypeSimplified", "winPlacePerc"], axis=1)
# coef = pd.Series(lasso_reg.coef_, index = X2.columns)
# coef.abs().sort_values(ascending=True).plot.bar(figsize=(15, 10), align="center")

# plt.figure(figsize=(15,10))
# plt.title("Damage Dealt",fontsize=15)
# sns.distplot(training['damageDealt'])
# plt.show()

######
# Feature Engineering
######

def feature_engineering(dataset):
    # Total distance covered by player during game
    dataset["totalDistanceCovered"] = dataset["rideDistance"] + dataset["swimDistance"] + dataset["walkDistance"]
    # Head shots as percent of kills
    dataset["killsByHeadShot"] = dataset["headshotKills"]/dataset["kills"]
    # Fill NaN values with 0s. 

    # Kills, assists, DBNOs 
    dataset["killsAssistsDBNOs"] = dataset["kills"] + dataset["assists"] + dataset["DBNOs"]
    
    dataset['numberOfTeamMates'] = dataset.groupby('groupId')['Id'].transform('count')
    dataset['numberOfEnemies'] = dataset.groupby('matchId')['Id'].transform("count") - dataset["numberOfTeamMates"]
    dataset["normalizedDistanceTravelled"] = dataset['totalDistanceCovered']*((100-dataset['matchDuration'])/100 + 1)
    dataset["normalizedKills"] = dataset['kills']*((100-dataset['numberOfEnemies'])/100 + 1)
    dataset['itemsAcquiredForDistanceTravelled'] = (dataset['weaponsAcquired'] + dataset['boosts'] + dataset['heals']) / dataset['totalDistanceCovered']
    dataset['killPlaceOverMaxPlace'] = dataset['killPlace']/dataset['maxPlace']
    dataset["killsByKillStreak"] = dataset["killStreaks"]/dataset["kills"]
    dataset["killsForDistanceTravelled"] = dataset["kills"]/dataset["totalDistanceCovered"]
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.fillna(0, inplace=True)

feature_engineering(train)
feature_engineering(test_X)

train.drop(["killPoints", "matchDuration", "maxPlace", "numGroups", "rankPoints", "revives", "roadKills", "teamKills", "vehicleDestroys", "winPoints", "squad", "solo", "duo", "killsForDistanceTravelled", "itemsAcquiredForDistanceTravelled"], axis=1, inplace=True, errors='ignore')
test_X.drop(["killPoints", "matchDuration", "maxPlace", "numGroups", "rankPoints", "revives", "roadKills", "teamKills", "vehicleDestroys", "winPoints", "squad", "solo", "duo", "killsForDistanceTravelled", "itemsAcquiredForDistanceTravelled"], axis=1, inplace=True, errors='ignore')
train.drop(["Id", "groupId", "matchId", "matchTypeSimplified", "matchType"], axis=1, inplace=True, errors="ignore")
test_X.drop(["Id", "groupId", "matchId", "matchType"], axis=1, inplace=True, errors="ignore")
X = train.drop(["winPlacePerc"], axis=1)
y = train["winPlacePerc"]

X = reduce_mem_usage(X)
test_X = reduce_mem_usage(test_X)

scaler = StandardScaler() 
scaler.fit(X)
X_scaled = scaler.transform(X)
lin_reg = LinearRegression()
maes = cross_val_score(lin_reg, X_scaled, y, cv=10, scoring="neg_mean_absolute_error")
mean_maes = np.mean(maes)
print(mean_maes)

scaler.fit(test_X)
Test_scaled = scaler.transform(test_X)
lin_reg.fit(X_scaled, y)
y_Pred = lin_reg.predict(Test_scaled)
#print(mean_absolute_error(y_test, y_Pred)

ridge_reg = RidgeCV()

ridge_reg.fit(X_scaled, y)
y_Pred = ridge_reg.predict(Test_scaled)


import lightgbm as lgb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=26)
lgb_train = lgb.Dataset(X_train, label=y_train)
res = lgb.cv({'metric': 'mae'},lgb_train, nfold=5,stratified=False,seed=232)
print("Mean score Light Gradient Boost without Cross Validation:",res['l1-mean'][-1])


lgb_val = lgb.Dataset(X_test, label= y_test)
params = {"objective" : "regression", 
          "metric" : ["mae"], 
          'n_estimators':1500, 
          'early_stopping_rounds':100,
          "num_leaves" : 31, 
          "learning_rate" : 0.05, 
          "bagging_fraction" : 0.9,
           "bagging_seed" : 0, 
          "num_threads" : 4,
          "colsample_bytree" : 0.7
         }
model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=300, verbose_eval=100)


#Light Gradient Boost Method
# 
# gridParams = {
#     'learning_rate': [0.05,0.15], 
#     'max_depth': [-1,12,20], 
#     'min_data_in_leaf': [100,300,500], 
#     'max_bin': [250,500],
#     'n_estimators': [50,250],
#     'num_leaves': [30,80,200],
#     'lambda_l1': [0.01],
#     'metric' : ['mae'],
#     'num_iterations': [5],
#     'nthread': [4]    
# }

# model = lgb.LGBMRegressor()
# grid = GridSearchCV(model, gridParams,
#                     verbose=1,
#                     cv=KFold())

# train_data = lgb.Dataset(data=train_x, label=train_y)
# valid_data = lgb.Dataset(data=test_x, label=test_y)   
# params = {"objective" : "regression", "metric" : "mae", 'n_estimators':15000, 'early_stopping_rounds':100,
#           "num_leaves" : 100, "learning_rate" : 0.05, "bagging_fraction" : 0.9,
#            "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
#          }
# lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=100) 

# grid.fit(X, y)


# best_params = grid.best_params_
# best_params

best_params = {'lambda_l1': 0.01,
 'learning_rate': 0.15,
 'max_bin': 250,
 'max_depth': -1,
 'metric': 'mae',
 'min_data_in_leaf': 100,
 'n_estimators': 100,
 'nthread': 4,
 'num_iterations': 5,
 'num_leaves': 200}
best_params['num_iterations'] = 5000
model = lgb.train(best_params, lgb_train, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=200, verbose_eval=100)
y_Pred = model.predict(test_X, num_iteration=model.best_iteration)

subm = pd.read_csv('../input/pubg-finish-placement-prediction/sample_submission_V2.csv')

test_with_id = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
test_with_id['winPlacePerc'] = y_Pred
test_with_id['winPlacePerc'] = test_with_id['winPlacePerc'].clip(0, 1)
subm['winPlacePerc'] = test_with_id['winPlacePerc']
subm['Id']=test_with_id["Id"]
subm.to_csv('submission.csv', index = False)

