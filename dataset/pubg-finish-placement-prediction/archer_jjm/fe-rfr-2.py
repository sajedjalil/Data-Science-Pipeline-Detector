import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest

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

train_data = pd.get_dummies(train_data, columns=['matchType'])
test_data = pd.get_dummies(test_data, columns=['matchType'])

def add_feature(data):
    data['headshotrate'] = data['kills']/data['headshotKills']
    data['killStreakrate'] = data['killStreaks']/data['kills']
    data['healthitems'] = data['heals'] + data['boosts']
    data['totalDistance'] = data['rideDistance'] + data["walkDistance"] + data["swimDistance"]
    data['killPlace_over_maxPlace'] = data['killPlace'] / data['maxPlace']
    data['headshotKills_over_kills'] = data['headshotKills'] / data['kills']
    data['distance_over_weapons'] = data['totalDistance'] / data['weaponsAcquired']
    data['walkDistance_over_heals'] = data['walkDistance'] / data['heals']
    data['walkDistance_over_kills'] = data['walkDistance'] / data['kills']
    data['killsPerWalkDistance'] = data['kills'] / data['walkDistance']
    data["skill"] = data["headshotKills"] + data["roadKills"]
    data[data == np.Inf] = np.NaN
    data[data == np.NINF] = np.NaN
    data.fillna(0,inplace=True)
    return data

def outlier_detection(data):
    data["killswithoutmove"] = (data["kills"]>0) & (data["totalDistance"] ==0)
    data.drop(data[data['killswithoutmove'] == True].index, inplace=True)
    data.drop(data[(data["kills"]>=29) & (train_data["totalDistance"]<1000)].index,inplace =True)
    data.drop(data[data["headshotrate"] ==1].index,inplace=True)
    data.drop(data[data['longestKill'] >= 1000].index, inplace=True)
    data.drop(data[data['longestKill'] >= 1000].index, inplace=True)
    data.drop(data[data['rideDistance'] >= 20000].index, inplace=True)
    data.drop(data[data['swimDistance'] >= 1000].index, inplace=True)
    data.drop(data[data['weaponsAcquired'] >= 20].index, inplace=True)
    data.drop(data[data['heals'] >= 20].index, inplace=True)
    data.drop(columns=['killswithoutmove'],inplace=True)
    return data

train_data = add_feature(train_data)
test_data = add_feature(test_data)

train_data = outlier_detection(train_data)

select_train_data = train_data.drop(["Id","groupId","matchId","winPlacePerc"],axis=1)
select_test_data = test_data.drop(["Id","groupId","matchId"],axis=1)

pca = PCA(n_components=51)

def pca_train_data(train_data):
    train_data_scaled=pca.fit_transform(train_data)
    return train_data_scaled

def pca_test_data(test_data):
    test_data_scaled=pca.transform(test_data)
    return test_data_scaled

train_data_processed = pca_train_data(select_train_data)
test_data_processed = pca_test_data(select_test_data)

X_train = pd.DataFrame(train_data_processed,columns=select_train_data.columns)
y_train = train_data["winPlacePerc"]
X_test = pd.DataFrame(test_data_processed,columns=select_train_data.columns)

to_keep_1 = ['assists', 'damageDealt', 'numGroups', 'matchType_solo','headshotKills', 'DBNOs', 'matchDuration',
       'matchType_normal-solo-fpp', 'matchType_normal-solo', 'heals','boosts', 'matchType_normal-duo-fpp', 'winPoints', 
       'rankPoints','killPlace', 'longestKill', 'rideDistance', 'killPoints','revives', 'matchType_duo', 'maxPlace',
       'matchType_normal-squad-fpp', 'swimDistance', 'matchType_duo-fpp','weaponsAcquired', 'matchType_solo-fpp', 'healthitems']

X_train_keep_1 = X_train[to_keep_1]

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


X_train_keep_1 = reduce_mem_usage(X_train_keep_1)
X_test = reduce_mem_usage(X_test)

Id = test_data['Id']
del train_data
del test_data
del train_data_processed
del test_data_processed
del select_train_data
del select_test_data

print("end process,begin training")
RFR1 = RandomForestRegressor(n_estimators=80,min_samples_leaf=3, max_features='sqrt',n_jobs=-1,verbose=1)
RFR1.fit(X_train_keep_1,y_train)
print("end training")

predictions = np.clip(a = RFR1.predict(X_test[to_keep_1]), a_min = 0.0, a_max = 1.0)
pred_df = pd.DataFrame({'Id' : Id, 'winPlacePerc' : predictions})
pred_df.to_csv("submission.csv", index=False)