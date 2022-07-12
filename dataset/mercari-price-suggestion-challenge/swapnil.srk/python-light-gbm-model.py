import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import explained_variance_score
import gc

# Importing the dataset
train = pd.read_csv('../input/train.tsv', sep='\t')
print(train.shape)
test = pd.read_csv('../input/test.tsv', sep='\t')
print(test.shape)


## Split the catagories
# reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
train['cat1'], train['cat2'], train['cat3'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
# train.head()
test['cat1'], test['cat2'], test['cat3'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))
# test.head()

## separate response variable and merge train and test datasets
y = train.price
train = train.drop('price', axis=1)
train.rename(columns={'train_id': 'id'}, inplace=True)
test.rename(columns={'test_id': 'id'}, inplace=True)
all = train.append(test, ignore_index=True)
del train 
del test
gc.collect()

## fill missing data
all.brand_name = all.brand_name.fillna("NoBrand")
all.cat1 = all.cat1.fillna("UnknowCat1")
all.cat2 = all.cat2.fillna("UnknowCat2")
all.cat3 = all.cat3.fillna("UnknowCat3")

## label encode brand variable
brle = LabelEncoder()
all['brand_label'] = brle.fit_transform(all['brand_name'])

## One hot encoding for catagorical features
all = pd.get_dummies(all, columns=["item_condition_id","cat1", "cat2","cat3"], sparse = True)
print(all.shape)

## remove unwanted variables
all = all.drop(['id', 'name', 'brand_name', 'category_name', 'item_description'], axis=1)
#print(all.shape)

## split train and submit\test datasets again
train = all[:len(y)]
submit = all[len(y):]

del all
gc.collect()

# Splitting the dataset into the Training set and Validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.4, random_state = 32)
del train
gc.collect()

X_train = X_train.fillna(-777)
X_test = X_test.fillna(-777)
submit = submit.fillna(-777)
print(X_train.shape, X_test.shape, submit.shape)

X_train = X_train.values
X_test = X_test.values
gc.collect()

## Fitting LightGBM on training data set
import lightgbm as lgb
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=5)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)