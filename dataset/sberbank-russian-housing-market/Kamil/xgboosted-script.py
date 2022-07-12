import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn import model_selection, preprocessing
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from xgboost import *
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
#import sklearn.linear_model as linear_model
#from sklearn.ensemble import RandomForestClassifier

# Reading all three files
train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

# Removing outliers for full_sq
train = train[train.full_sq<=265]

# Removing the outlier from state column
train.loc[train[train.state == 33].index[0],'state'] = 2.0
# removing the outlier from build_yead column
train.loc[train[train.build_year == 20052009].index[0],'build_year'] = 2007

# Merging macro data with train and test
train = pd.merge(train, macro, how='left', on='timestamp')
test = pd.merge(test, macro, how='left', on='timestamp')

# normalize prize feature
train["price_doc"] = np.log1p(train["price_doc"])
# store it as Y
Y_train = train["price_doc"]
# Dropping price column
train.drop("price_doc", axis=1, inplace=True)

# Droppign timestamp as we already joined data and datatype creating issue in skew
train.drop("timestamp", axis=1, inplace=True)
test.drop("timestamp", axis=1, inplace=True)

# Merging both dataframes
all_data = pd.concat((train.loc[:,'id':'apartment_fund_sqm'],test.loc[:,'id':'apartment_fund_sqm']))

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data['metro_km_walk']=all_data['metro_km_walk'].fillna(all_data.metro_km_avto)
all_data['metro_min_walk']=all_data['metro_min_walk'].fillna(all_data.metro_min_avto)
all_data['railroad_station_walk_km']=all_data['railroad_station_walk_km'].fillna(all_data.railroad_station_avto_km)
all_data['railroad_station_walk_min']=all_data['railroad_station_walk_min'].fillna(all_data.railroad_station_avto_min)
all_data['ID_railroad_station_walk']=all_data['ID_railroad_station_walk'].fillna(all_data.ID_railroad_station_avto)

# the magic feature ;)
all_data["apartment_name"]=all_data["sub_area"] + all_data['metro_km_avto'].astype(str)


#convert objects / non-numeric data types into numeric
for f in all_data.columns:
    if all_data[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(all_data[f].values)) 
        all_data[f] = lbl.transform(list(all_data[f].values))

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

print(X_train.shape)
print(X_test.shape)


n_iter_search = 1
param_grid_xgb = {
            'max_depth' : range(1, 20),
            'learning_rate' : np.arange(0.01, 0.1, 0.01),
            'n_estimators' : range(1, 150),
            'min_child_weight' : np.arange(0.5, 1., 0.1),
            'subsample' : np.arange(0.5, 1, 0.1),
            'colsample_bytree' : np.arange(0.5, 1, 0.1),
            'seed' : [0]
            }

steps = [("XGBoost", XGBRegressor())]
pipeline = Pipeline(steps)

random_search = RandomizedSearchCV(XGBRegressor(), 
                                       param_distributions = param_grid_xgb,
                                       n_iter = n_iter_search,
                                       scoring="neg_mean_squared_error", 
                                       n_jobs = -1)
                                       
start = time()
random_search.fit(X_train, Y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings. \n" % ((time() - start), n_iter_search))

Y_pred = np.expm1(random_search.predict(X_test))

# Preparing submission file
submission = pd.DataFrame({"id": test["id"],"price_doc": Y_pred})
submission.to_csv('submission.csv', index=False)