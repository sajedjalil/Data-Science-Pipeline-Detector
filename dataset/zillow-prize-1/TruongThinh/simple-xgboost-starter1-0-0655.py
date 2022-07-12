# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

##### READ IN RAW DATA

print( "\nReading data from disk ...")
properties = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")

for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

df_train = train.merge(properties, how='left', on='parcelid')
#life of property
df_train['N-life'] = 2018 - df_train['yearbuilt']

#error in calculation of the finished living area of home
df_train['N-LivingAreaError'] = df_train['calculatedfinishedsquarefeet']/df_train['finishedsquarefeet12']

#proportion of living area
df_train['N-LivingAreaProp'] = df_train['calculatedfinishedsquarefeet']/df_train['lotsizesquarefeet']
df_train['N-LivingAreaProp2'] = df_train['finishedsquarefeet12']/df_train['finishedsquarefeet15']

#Amout of extra space
df_train['N-ExtraSpace'] = df_train['lotsizesquarefeet'] - df_train['calculatedfinishedsquarefeet']
df_train['N-ExtraSpace-2'] = df_train['finishedsquarefeet15'] - df_train['finishedsquarefeet12']

#Total number of rooms
df_train['N-TotalRooms'] = df_train['bathroomcnt']*df_train['bedroomcnt']

#Average room size 62
# if df_train['roomcnt'] != 0.0:
#     df_train['N-AvRoomSize'] = df_train['calculatedfinishedsquarefeet']/df_train['roomcnt']
# else:
#     df_train['N-AvRoomSize'] = -1.0

# Number of Extra rooms
df_train['N-ExtraRooms'] = df_train['roomcnt'] - df_train['N-TotalRooms']

#Ratio of the built structure value to land area
df_train['N-ValueProp'] = df_train['structuretaxvaluedollarcnt']/df_train['landtaxvaluedollarcnt']

#Does property have a garage, pool or hot tub and AC?
df_train['N-GarPoolAC'] = ((df_train['garagecarcnt']>0) & (df_train['pooltypeid10']>0) & (df_train['airconditioningtypeid']!=5))*1

df_train["N-location"] = df_train["latitude"] + df_train["longitude"]
df_train["N-location-2"] = df_train["latitude"]*df_train["longitude"]
df_train["N-location-2round"] = df_train["N-location-2"].round(-4)

df_train["N-latitude-round"] = df_train["latitude"].round(-4)
df_train["N-longitude-round"] = df_train["longitude"].round(-4)

#Ratio of tax of property over parcel
df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt']/df_train['taxamount']

#TotalTaxScore
df_train['N-TaxScore'] = df_train['taxvaluedollarcnt']*df_train['taxamount']

#polnomials of tax delinquency year
df_train["N-taxdelinquencyyear-2"] = df_train["taxdelinquencyyear"] ** 2
df_train["N-taxdelinquencyyear-3"] = df_train["taxdelinquencyyear"] ** 3

#Length of time since unpaid taxes
df_train['N-life'] = 2018 - df_train['taxdelinquencyyear']

#Ratio of tax of property over parcel
df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt']/df_train['taxamount']

#TotalTaxScore
df_train['N-TaxScore'] = df_train['taxvaluedollarcnt']*df_train['taxamount']

#polnomials of tax delinquency year
df_train["N-taxdelinquencyyear-2"] = df_train["taxdelinquencyyear"] ** 2
df_train["N-taxdelinquencyyear-3"] = df_train["taxdelinquencyyear"] ** 3

#Length of time since unpaid taxes
df_train['N-life'] = 2018 - df_train['taxdelinquencyyear']

#Number of properties in the zip
zip_count = df_train['regionidzip'].value_counts().to_dict()
df_train['N-zip_count'] = df_train['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = df_train['regionidcity'].value_counts().to_dict()
df_train['N-city_count'] = df_train['regionidcity'].map(city_count) #76

#Number of properties in the city
region_count = df_train['regionidcounty'].value_counts().to_dict()
df_train['N-county_count'] = df_train['regionidcounty'].map(region_count) #77

#Indicator whether it has AC or not
df_train['N-ACInd'] = (df_train['airconditioningtypeid']!=5)*1

#Indicator whether it has Heating or not
df_train['N-HeatInd'] = (df_train['heatingorsystemtypeid']!=13)*1

#There's 25 different property uses - let's compress them down to 4 categories
df_train['N-PropType'] = df_train.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 248 : "Mixed", 260 : "Home", 261 : "Home", 262 : "Home", 263 : "Home", 264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home", 269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 291 : "Not Built" })

#polnomials of the variable
df_train["N-structuretaxvaluedollarcnt-2"] = df_train["structuretaxvaluedollarcnt"] ** 2
df_train["N-structuretaxvaluedollarcnt-3"] = df_train["structuretaxvaluedollarcnt"] ** 3

#Average structuretaxvaluedollarcnt by city
group = df_train.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
df_train['N-Avg-structuretaxvaluedollarcnt'] = df_train['regionidcity'].map(group)

#Deviation away from average
df_train['N-Dev-structuretaxvaluedollarcnt'] = abs((df_train['structuretaxvaluedollarcnt'] - df_train['N-Avg-structuretaxvaluedollarcnt']))/df_train['N-Avg-structuretaxvaluedollarcnt']

train_y = df_train['logerror'].values
df_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
feat_names = df_train.columns.values

for c in df_train.columns:
    if df_train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(df_train[c].values))
        df_train[c] = lbl.transform(list(df_train[c].values))

#import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}

for c, dtype in zip(df_train.columns, df_train.dtypes):
    if dtype == np.float64:
        df_train[c] = df_train[c].astype(np.float32)
for c in df_train.dtypes[df_train.dtypes == object].index.values:
    df_train[c] = (df_train[c] == True)
for c in df_train.columns:
    df_train[c]=df_train[c].fillna(-1)
    if df_train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(df_train[c].values))
        df_train[c] = lbl.transform(list(df_train[c].values))

#take sample data
numberOfSample = 100 #need to change here
df_train = df_train[:numberOfSample]
train_y = train_y[:numberOfSample]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# sklearn.preprocessing.normalize(df_train, norm='max', axis=1, copy=True, return_norm=False)
#converse data to numpy array
df_train = np.array(df_train)
train_y = np.array(train_y)
C_range = np.logspace(0, 10, 5)
gamma_range = np.logspace(-1, 3, 5)
param_grid = dict( C=C_range)
grid = GridSearchCV(SVR(), param_grid=param_grid, cv=10)
grid.fit(df_train, train_y)
#best params
text = "The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_)
print(text)
# plt.hist(train_y, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()

# numberOfFolds = 5 #need to change here
# folds = list(KFold(n_splits=numberOfFolds, shuffle=True, random_state=2016).split(df_train, train_y))
# clf = SVR(C=1.0, epsilon=0.2)
# #training
# print("Training:")
# for j, (train_idx, test_idx) in enumerate(folds):
#     X_train = df_train[train_idx]
#     y_train = train_y[train_idx]
#     X_holdout = df_train[test_idx]
#     y_holdout = train_y[test_idx]
#
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_holdout)[:]
#
#     print ("fold %d score %f" % ( j, r2_score(y_holdout, y_pred)))

