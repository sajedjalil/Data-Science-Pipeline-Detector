import numpy as np
import pandas as pd

def load(test=False):
    
    if test:
        X = pd.read_csv("../input/test.csv")
        y = X[['Id']].T.values[0]
        X.drop(['Id'], axis=1, inplace=True)
    else:
        X = pd.read_csv("../input/train.csv", dtype={'StateHoliday': object})
        y = X[['Sales']].T.values[0]
    
    return X, y   
    
    
def loadStore():
    return pd.read_csv("../input/store.csv")
    
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)/y_true) ** 2)

from sklearn.metrics import make_scorer

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)


def rmspe_xgb(y, predicted):
    predicted = predicted.get_label()
    return "rmspe", rmspe(y, predicted)



# Load Data

train, y = load()
test, ids = load(True)
store = loadStore()

train_df = pd.merge(train, store, left_on='Store', right_on='Store', how='inner')
test_df = pd.merge(test, store, left_on='Store', right_on='Store', how='inner')


# Getting features
#Fill NaN with Open = 1, because sales in this days > 0

test_df.Open.fillna(1, inplace=True)

#Getting some features from 'Date'

train_df['Year'] = pd.to_datetime(train_df.Date).map(lambda x: x.year)
train_df['DayOfYear'] = pd.to_datetime(train_df.Date).map(lambda x: x.dayofyear)
train_df['Month'] = pd.to_datetime(train_df.Date).map(lambda x: x.month)

test_df['Year'] = pd.to_datetime(test_df.Date).map(lambda x: x.year)
test_df['DayOfYear'] = pd.to_datetime(test_df.Date).map(lambda x: x.dayofyear)
test_df['Month'] = pd.to_datetime(test_df.Date).map(lambda x: x.month)

#Promo2 to 0 - if no Promo2, 1 - if in this day Promo2 is available

train_df.Promo2SinceYear.fillna(2016, inplace=True)
train_df.Promo2SinceWeek.fillna(1, inplace=True)
train_df.PromoInterval.fillna(0, inplace=True)

test_df.Promo2SinceYear.fillna(2016, inplace=True)
test_df.Promo2SinceWeek.fillna(1, inplace=True)
test_df.PromoInterval.fillna(0, inplace=True)


dictionary1 = {0:0, 'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3}
dictionary2 = {1:1, 4:1, 7:1, 10:1, 2:2, 5:2, 8:2, 11:2, 3:3, 6:3, 9:3, 12:3}

train_df['PromoInterval'] = train_df['PromoInterval'].map(dictionary1)
train_df['MonthInterval'] = train_df['Month'].map(dictionary2)

test_df['PromoInterval'] = test_df['PromoInterval'].map(dictionary1)
test_df['MonthInterval'] = test_df['Month'].map(dictionary2)



train_df['Promo2'] = np.sign((train_df.Year - train_df.Promo2SinceYear) * 365 + 
                              (train_df.DayOfYear - train_df.Promo2SinceWeek * 7))

test_df['Promo2'] = np.sign((test_df.Year - test_df.Promo2SinceYear) * 365 + 
                              (test_df.DayOfYear - test_df.Promo2SinceWeek * 7))

def binarizePromo2(df):

    promo2 = []
    
    for row in range(len(df)):

        if df.Promo2[row] == 1 and df.PromoInterval[row] == df.MonthInterval[row]:
            promo2.append(1)
        else:
            promo2.append(0)
    
    return promo2
        
train_df['Promo2'] = pd.DataFrame(binarizePromo2(train_df))
test_df['Promo2'] = pd.DataFrame(binarizePromo2(test_df))

#CompetitionOpen in days

train_df['CompetitionOpen'] = ((train_df.Year - train_df.CompetitionOpenSinceYear) * 365 + 
                                (train_df.DayOfYear - train_df.CompetitionOpenSinceMonth * 30))

test_df['CompetitionOpen'] = ((test_df.Year - test_df.CompetitionOpenSinceYear) * 365 + 
                                (test_df.DayOfYear - test_df.CompetitionOpenSinceMonth * 30))
                                
#fill NaN with median

med = train_df.CompetitionDistance.median()
train_df['CompetitionDistance'] = train_df.CompetitionDistance.fillna(med)

med = train_df.CompetitionOpen.median()
train_df['CompetitionOpen'] = train_df.CompetitionOpen.fillna(med)

med = test_df.CompetitionDistance.median()
test_df['CompetitionDistance'] = test_df.CompetitionDistance.fillna(med)

med = test_df.CompetitionOpen.median()
test_df['CompetitionOpen'] = test_df.CompetitionOpen.fillna(med)

train_df.CompetitionDistance = train_df.CompetitionDistance.astype(int)
train_df.CompetitionOpen = train_df.CompetitionOpen.astype(int)

test_df.CompetitionDistance = test_df.CompetitionDistance.astype(int)
test_df.CompetitionOpen = test_df.CompetitionOpen.astype(int)

train_df = train_df[train_df.Sales != 0]



# Category features
#Category featores to digits

from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()

enc_list = ['StoreType', 'Assortment', 'StateHoliday']

for name in enc_list:
    
    train_df[name] = np.array(label_enc.fit_transform(train_df[name]))
    test_df[name] = np.array(label_enc.fit_transform(test_df[name]))
    
print(train_df['StateHoliday'].unique())

train_df.drop(['Promo2SinceYear', 'Promo2SinceWeek', 'CompetitionOpenSinceYear', 
'CompetitionOpenSinceMonth', 'PromoInterval', 'MonthInterval', 'Year'], axis=1, inplace=True)

test_df.drop(['Promo2SinceYear', 'Promo2SinceWeek', 'CompetitionOpenSinceYear', 
'CompetitionOpenSinceMonth', 'PromoInterval', 'MonthInterval', 'Year'], axis=1, inplace=True)



# Metafeature
#Split dataset into the 2 parts 50/50

from sklearn.cross_validation import train_test_split

drop_cols = ['Open', 'Sales', 'Customers', 'Date']

X = train_df.drop(drop_cols, axis=1).values
y_c = train_df['Customers'].T.values

X_tr, X_te, y_tr, y_te = train_test_split(X, y_c, test_size=0.50)

test = test_df.drop(['Open', 'Date'], axis=1).values

#train first half to predict 'Customers'
#predict 'Customers' to the second half

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=25, random_state=0)

result_tr = [0, 0]
result_te = [0, 0]

clf.fit(X_tr, y_tr)
result_tr[0] = clf.predict(X_te) 
result_te[0] = clf.predict(test) 

clf.fit(X_te, y_te)
result_tr[1] = clf.predict(X_tr) 
result_te[1] = clf.predict(test) 


#for test avarage of 2 predictions
train_df['Meta'] = np.concatenate((result_tr[0], result_tr[1]), axis=0).astype(int)
test_df['Meta'] = ((result_te[0] + result_te[1]) / 2).astype(int)



# Scoring

import xgboost

def fit_xgb(X_tr, X_te, y_tr, y_te):
    xgb_tr, xgb_te, yxgb_tr, yxgb_te = train_test_split(X_tr, y_tr, test_size=0.2, random_state=10)

    dtrain = xgboost.DMatrix(xgb_tr, label = yxgb_tr)
    dval = xgboost.DMatrix(xgb_te, label = yxgb_te)


    params = {'objective': 'reg:linear',
              'booster': 'gbtree',
              'eta': 0.3,
              'max_depth': 13,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'silent': 1,
              'seed': 1001,
              'nthread': 4
              }
    num_round = 300

    plst = list(params.items())

    evallist  = [(dval,'eval'), (dtrain,'train')]

    bst = xgboost.train( plst, dtrain, num_round, evallist, early_stopping_rounds=100, feval=rmspe_xgb, verbose_eval=True)

    y_val = bst.predict(xgboost.DMatrix(X_te))

    print('Error', np.fabs(rmspe(y_te, y_val)))
    
    return bst
    
    
    
#split on time

drop_cols = ['Open', 'Sales', 'Customers', 'Date']

mask = [(pd.to_datetime(train_df.Date) < np.datetime64('2015-06-01T00:00:00.000000000+0000'),
        pd.to_datetime(train_df.Date) >= np.datetime64('2015-06-01T00:00:00.000000000+0000'))]

y_tr = train_df.loc[mask[0][0]].Sales.T.values
y_te = train_df.loc[mask[0][1]].Sales.T.values

X_tr = train_df.loc[mask[0][0], :].drop(drop_cols, axis=1).values
X_te = train_df.loc[mask[0][1], :].drop(drop_cols, axis=1).values

test = test_df.drop(['Open', 'Date'], axis=1).values

cols = train_df.drop(drop_cols, axis=1).columns

print()
print('XGBoost')
print('Split on time')
bst = fit_xgb(X_tr, X_te, y_tr, y_te)



#split on random

from sklearn.cross_validation import train_test_split

drop_cols = ['Open', 'Sales', 'Customers', 'Date']

y = train_df['Sales'].T.values
X = train_df.drop(drop_cols, axis=1).values

test = test_df.drop(['Open', 'Date'], axis=1).values

cols = train_df.drop(drop_cols, axis=1).columns

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=50)

print()
print('XGBoost')
print('Random split')
bst = fit_xgb(X_tr, X_te, y_tr, y_te)


#importance of features
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_jobs=-1, random_state=10, n_estimators = 100)
  
clf.fit(X_tr, y_tr)

predicted = clf.predict(X_te)
    
imp = clf.feature_importances_
   
print('Feature importance RF') 
for i in range(len(cols)):
    
    print(cols[i], imp[i])
    
print('Error RF', np.fabs(rmspe(y_te, predicted)))

# Predict

predicted = bst.predict(xgboost.DMatrix(test))
test_df['Sales'] = (predicted * test_df['Open'])
test_df['Id'] = ids
result = pd.DataFrame({"Id": test_df["Id"], 'Sales': test_df["Sales"]})
#result.to_csv('../output/output.csv', index=False, index_label=False)