import numpy as np
import pandas as pd
import gc
import lightgbm as lgb

#print("Read Done")
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    # iterate through all the columns of a dataframe and modify the data type
    #   to reduce memory usage.        
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df





def featureModify(isTrain):
    if isTrain:
        all_data = pd.read_csv('../input/train_V2.csv') 
        all_data = all_data[all_data['maxPlace'] > 1]
        all_data = reduce_mem_usage(all_data)
        all_data = all_data[all_data['winPlacePerc'].notnull()]
    else:
        all_data = pd.read_csv('../input/test_V2.csv')


    all_data['matchType'] = all_data['matchType'].map({
    'crashfpp':1,
    'crashtpp':2,
    'duo':3,
    'duo-fpp':4,
    'flarefpp':5,
    'flaretpp':6,
    'normal-duo':7,
    'normal-duo-fpp':8,
    'normal-solo':9,
    'normal-solo-fpp':10,
    'normal-squad':11,
    'normal-squad-fpp':12,
    'solo':13,
    'solo-fpp':14,
    'squad':15,
    'squad-fpp':16
    })
    all_data = reduce_mem_usage(all_data)

    print("Match size")
    matchSizeData = all_data.groupby(['matchId']).size().reset_index(name='matchSize')
    all_data = pd.merge(all_data, matchSizeData, how='left', on=['matchId'])
    del matchSizeData
    gc.collect()
    
    
    all_data.loc[(all_data['rankPoints']==-1), 'rankPoints'] = 0
    all_data['_killPoints_rankpoints'] = all_data['rankPoints']+all_data['killPoints']


    all_data["_Kill_headshot_Ratio"] = all_data["kills"]/all_data["headshotKills"]
    all_data['_killStreak_Kill_ratio'] = all_data['killStreaks']/all_data['kills']
    all_data['_totalDistance'] = 0.25*all_data['rideDistance'] + all_data["walkDistance"] + all_data["swimDistance"]
    all_data['_killPlace_MaxPlace_Ratio'] = all_data['killPlace'] / all_data['maxPlace']
    all_data['_totalDistance_weaponsAcq_Ratio'] = all_data['_totalDistance'] / all_data['weaponsAcquired']
    all_data['_walkDistance_heals_Ratio'] = all_data['walkDistance'] / all_data['heals']
    all_data['_walkDistance_kills_Ratio'] = all_data['walkDistance'] / all_data['kills']
    all_data['_kills_walkDistance_Ratio'] = all_data['kills'] / all_data['walkDistance']
    all_data['_totalDistancePerDuration'] =  all_data["_totalDistance"]/all_data["matchDuration"]
    all_data['_killPlace_kills_Ratio'] = all_data['killPlace']/all_data['kills']
    all_data['_walkDistancePerDuration'] =  all_data["walkDistance"]/all_data["matchDuration"]
    all_data['walkDistancePerc'] = all_data.groupby('matchId')['walkDistance'].rank(pct=True).values
    all_data['killPerc'] = all_data.groupby('matchId')['kills'].rank(pct=True).values
    all_data['killPlacePerc'] = all_data.groupby('matchId')['killPlace'].rank(pct=True).values
    all_data['weaponsAcquired'] = all_data.groupby('matchId')['weaponsAcquired'].rank(pct=True).values
    all_data['_walkDistance_kills_Ratio2'] = all_data['walkDistancePerc'] / all_data['killPerc']
    all_data['_kill_kills_Ratio2'] = all_data['killPerc']/all_data['walkDistancePerc']
    all_data['_killPlace_walkDistance_Ratio2'] = all_data['walkDistancePerc']/all_data['killPlacePerc']
    all_data['_killPlace_kills_Ratio2'] = all_data['killPlacePerc']/all_data['killPerc']
    all_data['_totalDistance'] = all_data.groupby('matchId')['_totalDistance'].rank(pct=True).values
    all_data['_walkDistance_kills_Ratio3'] = all_data['walkDistancePerc'] / all_data['kills']
    all_data['_walkDistance_kills_Ratio4'] = all_data['kills'] / all_data['walkDistancePerc']
    all_data['_walkDistance_kills_Ratio5'] = all_data['killPerc'] / all_data['walkDistance']
    all_data['_walkDistance_kills_Ratio6'] = all_data['walkDistance'] / all_data['killPerc']

    all_data[all_data == np.Inf] = np.NaN
    all_data[all_data == np.NINF] = np.NaN
    all_data.fillna(0, inplace=True)
    
    features = list(all_data.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchSize")
    features.remove("matchType")
    if isTrain:
        features.remove("winPlacePerc")

    
    print("Mean Data")
    meanData = all_data.groupby(['matchId','groupId'])[features].agg('mean')
    meanData = reduce_mem_usage(meanData)
    meanData = meanData.replace([np.inf, np.NINF,np.nan], 0)
    meanDataRank = meanData.groupby('matchId')[features].rank(pct=True).reset_index()
    meanDataRank = reduce_mem_usage(meanDataRank)
    all_data = pd.merge(all_data, meanData.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
    del meanData
    gc.collect()
    all_data = all_data.drop(["vehicleDestroys_mean","rideDistance_mean","roadKills_mean","rankPoints_mean"], axis=1)
    all_data = pd.merge(all_data, meanDataRank, suffixes=["", "_meanRank"], how='left', on=['matchId', 'groupId'])
    del meanDataRank
    gc.collect()
    all_data = all_data.drop(["numGroups_meanRank","rankPoints_meanRank"], axis=1)
    
    all_data = all_data.join(reduce_mem_usage(all_data.groupby('matchId')[features].rank(ascending=False).add_suffix('_rankPlace').astype(int)))

    
    print("Std Data")
    stdData = all_data.groupby(['matchId','groupId'])[features].agg('std').replace([np.inf, np.NINF,np.nan], 0)
    stdDataRank = reduce_mem_usage(stdData.groupby('matchId')[features].rank(pct=True)).reset_index()
    del stdData
    gc.collect()
    all_data = pd.merge(all_data, stdDataRank, suffixes=["", "_stdRank"], how='left', on=['matchId', 'groupId'])
    del stdDataRank
    gc.collect()
    
    print("Max Data")
    maxData = all_data.groupby(['matchId','groupId'])[features].agg('max')
    maxData = reduce_mem_usage(maxData)
    maxDataRank = maxData.groupby('matchId')[features].rank(pct=True).reset_index()
    maxDataRank = reduce_mem_usage(maxDataRank)
    all_data = pd.merge(all_data, maxData.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
    del maxData
    gc.collect()
    all_data = all_data.drop(["assists_max","killPoints_max","headshotKills_max","numGroups_max","revives_max","teamKills_max","roadKills_max","vehicleDestroys_max"], axis=1)
    all_data = pd.merge(all_data, maxDataRank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])
    del maxDataRank
    gc.collect()
    all_data = all_data.drop(["roadKills_maxRank","matchDuration_maxRank","maxPlace_maxRank","numGroups_maxRank"], axis=1)


    print("Min Data")
    minData = all_data.groupby(['matchId','groupId'])[features].agg('min')
    minData = reduce_mem_usage(minData)
    minDataRank = minData.groupby('matchId')[features].rank(pct=True).reset_index()
    minDataRank = reduce_mem_usage(minDataRank)
    all_data = pd.merge(all_data, minData.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
    del minData
    gc.collect()
    all_data = all_data.drop(["heals_min","killStreaks_min","killPoints_min","maxPlace_min","revives_min","headshotKills_min","weaponsAcquired_min","_walkDistance_kills_Ratio_min","rankPoints_min","matchDuration_min","teamKills_min","numGroups_min","assists_min","roadKills_min","vehicleDestroys_min"], axis=1)
    all_data = pd.merge(all_data, minDataRank, suffixes=["", "_minRank"], how='left', on=['matchId', 'groupId'])
    del minDataRank
    gc.collect()
    all_data = all_data.drop(["killPoints_minRank","matchDuration_minRank","maxPlace_minRank","numGroups_minRank"], axis=1)

    
    print("group Size")
    groupSize = all_data.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    groupSize = reduce_mem_usage(groupSize)
    all_data = pd.merge(all_data, groupSize, how='left', on=['matchId', 'groupId'])
    del groupSize
    gc.collect()

    
    print("Match Mean")
    matchMeanFeatures = features
    matchMeanFeatures = [ v for v in matchMeanFeatures if v not in ["killPlacePerc","matchDuration","maxPlace","numGroups"] ]
    matchMeanData= reduce_mem_usage(all_data.groupby(['matchId'])[matchMeanFeatures].transform('mean')).replace([np.inf, np.NINF,np.nan], 0)
    all_data = pd.concat([all_data,matchMeanData.add_suffix('_matchMean')],axis=1)
    del matchMeanData,matchMeanFeatures
    gc.collect()

    print("matchMax")
    matchMaxFeatures = ["walkDistance","kills","_walkDistance_kills_Ratio","_kill_kills_Ratio2"]
    all_data = pd.merge(all_data, reduce_mem_usage(all_data.groupby(['matchId'])[matchMaxFeatures].agg('max')).reset_index(), suffixes=["", "_matchMax"], how='left', on=['matchId'])

    print("match STD")
    matchMaxFeatures = ["kills","_walkDistance_kills_Ratio2","_walkDistance_kills_Ratio","killPerc","_kills_walkDistance_Ratio"]
    all_data = pd.merge(all_data, reduce_mem_usage(all_data.groupby(['matchId'])[matchMaxFeatures].agg('std')).reset_index().replace([np.inf, np.NINF,np.nan], 0), suffixes=["", "_matchSTD"], how='left', on=['matchId'])


    all_data = all_data.drop(["Id","groupId"], axis=1)
    all_data = all_data.drop(["DBNOs","assists","headshotKills","heals","killPoints","_killStreak_Kill_ratio","killStreaks","longestKill","revives","roadKills","teamKills","vehicleDestroys","_walkDistance_kills_Ratio","weaponsAcquired"], axis=1)
    all_data = all_data.drop(["_walkDistance_heals_Ratio","_totalDistancePerDuration","_killPlace_kills_Ratio","_totalDistance_weaponsAcq_Ratio","_killPlace_MaxPlace_Ratio","_walkDistancePerDuration","rankPoints","rideDistance","boosts","winPoints","swimDistance","_kills_walkDistance_Ratio"], axis=1)
    all_data = all_data.drop(["_Kill_headshot_Ratio","maxPlace","_totalDistance","numGroups","walkDistance","killPlace"], axis=1)
    all_data = reduce_mem_usage(all_data)
    gc.collect()
    
    print("done")
    features_label = all_data.columns
    features_label = features_label.drop('matchId')
    if isTrain:
        features_label = features_label.drop('winPlacePerc')

    gc.collect()
    return all_data,features_label
    
X_train,features_label = featureModify(True) 

print("Split time")
def split_train_val(data, fraction):
    matchIds = data['matchId'].unique().reshape([-1])
    train_size = int(len(matchIds)*fraction)
    
    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))
    train_matchIds = matchIds[random_idx[:train_size]]
    val_matchIds = matchIds[random_idx[train_size:]]
    
    data_train = data.loc[data['matchId'].isin(train_matchIds)]
    data_val = data.loc[data['matchId'].isin(val_matchIds)]
    return data_train, data_val

# Split the Data by matchId. Thanks to Ivan Batalov for this. 
X_train, X_train_test = split_train_val(X_train, 0.91)
print("Y time")
y = X_train['winPlacePerc']
y_test = X_train_test['winPlacePerc']
print("X_train time")
X_train = X_train.drop(columns=['matchId', 'winPlacePerc'])
print("X test train time")
X_train_test = X_train_test.drop(columns='matchId')
print("X test train winPlace remove")
X_train_test = X_train_test.drop(columns='winPlacePerc')

print("X test np time")
X_train_test = np.array(X_train_test)
print("y test np time")
y_test = np.array(y_test)

#Split the Data again and then join it. I am doing this because If I turn the Pandas DataFrame into Numpy Array with 
# all rows at once, Kernel will be killed for exceeding 16GB Memory. 
from sklearn.model_selection import train_test_split
X_train, X_train2, y, y2 = train_test_split(X_train, y, test_size=0.1, shuffle=False)
print("X_train np time")
X_train = np.array(X_train)
print("y np time")
y = np.array(y)

print("X_train2 np time")
X_train2 = np.array(X_train2)
print("y2 np time")
y2 = np.array(y2)

y = np.concatenate((y, y2), axis=0)
del y2
gc.collect()
X_train = np.concatenate((X_train, X_train2), axis=0)
del X_train2
gc.collect()


train_set = lgb.Dataset(X_train, label=y)
del X_train,y
gc.collect()
valid_set = lgb.Dataset(X_train_test, label=y_test)
del X_train_test,y_test
gc.collect()

params = {
        "objective" : "regression", 
        "metric" : "mae", 
        "num_leaves" : 149, 
        "learning_rate" : 0.03, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 4,
        "colsample_bytree" : 0.5,
        'min_data_in_leaf':1900, 
        'min_split_gain':0.00011,
        'lambda_l2':9
}

model = lgb.train(  params, 
                    train_set = train_set,
                    num_boost_round=9400,
                    early_stopping_rounds=200,
                    verbose_eval=100, 
                    valid_sets=[train_set,valid_set]
                  )
  
del train_set,valid_set
gc.collect()
                
print("Calculating Feature Importance and save it in a file") 
featureImp = list(model.feature_importance())
featureImp, features_label = zip(*sorted(zip(featureImp, features_label)))
with open("FeatureImportance.txt", "w") as text_file:
    for i in range(len(featureImp)):
        print(f"{features_label[i]} =  {featureImp[i]}", file=text_file)

print("Done calculating")
del featureImp,features_label
gc.collect()

                  
X_test,features_label = featureModify(False) 
X_test = X_test.drop(columns=['matchId'])
X_test = np.array(X_test)
y_pred=model.predict(X_test, num_iteration=model.best_iteration)
del X_test
gc.collect()

# Insert ID and Predictions into dataframe
df_sub = pd.DataFrame()

df_test = pd.read_csv('../input/test_V2.csv')
df_test = reduce_mem_usage(df_test)
df_sub['Id'] = df_test['Id']
df_sub['winPlacePerc'] = y_pred

print(df_sub['winPlacePerc'].describe())


df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)
df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]


df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc
# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0

df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)
print(df_sub['winPlacePerc'].describe())