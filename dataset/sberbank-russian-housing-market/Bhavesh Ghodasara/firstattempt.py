import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn import model_selection, preprocessing
from scipy.stats import skew
from scipy.stats.stats import pearsonr
#import sklearn.linear_model as linear_model
#from sklearn.ensemble import RandomForestClassifier

# Reading all three files
train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

# Removing outliers for full_sq
train = train[train.full_sq<=265]

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


# Create dummy variables
#all_data = pd.get_dummies(all_data)


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

print(X_train.shape)
print(X_test.shape)


try:
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y_train)
    Y_pred = np.expm1(model_lasso.predict(X_test))
except Exception as e:
    print(type(e))
    print(e)
    

print("checkpoint -5")


# Preparing submission file
submission = pd.DataFrame({"id": test["id"],"price_doc": Y_pred})
submission.to_csv('submission.csv', index=False)