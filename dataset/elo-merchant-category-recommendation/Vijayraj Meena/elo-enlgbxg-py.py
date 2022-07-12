import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
import gc

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
print("shape of train : ",train.shape)

train.head()
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("shape of test : ",test.shape)

# Code in https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo
# SRK - Simple Exploration Notebook 

cnt_srs = train['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = test['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()

train.corr()
#histogram
f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(train['target'])

train['target'].describe()

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

data = pd.concat([train['target'], train['feature_1']], axis=1)
fig = sns.boxplot(x='feature_1', y="target", data=data,ax=ax1)

# feature 1
sns.violinplot(x="feature_1", y="target", data=data,ax=ax2)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()


fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

data = pd.concat([train['target'], train['feature_2']], axis=1)
fig = sns.boxplot(x='feature_2', y="target", data=data,ax=ax1)

# feature 1
sns.violinplot(x="feature_2", y="target", data=data,ax=ax2)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()


fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

data = pd.concat([train['target'], train['feature_3']], axis=1)
fig = sns.boxplot(x='feature_3', y="target", data=data,ax=ax1)

# feature 1
sns.violinplot(x="feature_3", y="target", data=data,ax=ax2)
plt.xticks(rotation='vertical')
plt.xlabel('feature_3', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.title("feature_3 distribution")
plt.show()


fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

sns.boxplot(x="feature_1", y="target", hue="feature_3",
               data=train, palette="Set3",ax=ax1)

sns.boxplot(x="feature_3", y="target", hue="feature_1",
               data=train, palette="Set3",ax=ax2)
               
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

sns.boxplot(x="feature_1", y="target", hue="feature_2",
               data=train, palette="Set3",ax=ax1)

sns.boxplot(x="feature_2", y="target", hue="feature_1",
               data=train, palette="Set3",ax=ax2)
               
# Code in https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo
# SRK - Simple Exploration Notebook 
train_low_m30 = train[train['target']<-30]
cnt_srs = train_low_m30['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in target low than -30")
plt.show()

train_low_m30.head()

# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# checking missing data
total = test.isnull().sum().sort_values(ascending = False)
percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

import datetime

for df in [train,test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

target = train['target']
del train['target']

train.head()
#Simple Exploration : historical_transactions
ht = pd.read_csv("../input/historical_transactions.csv")
print("shape of historical_transactions : ",ht.shape)
    
ht.head()

temp = ht["authorized_flag"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('authorized_flag - Y or N')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

temp = ht["category_1"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_1 - Y or N')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

temp = ht["category_3"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_3 - A B C')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

temp = ht["category_2"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_2 - 1,2,3,4,5')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


#histogram
f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(ht['month_lag'])

#histogram
f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(ht['purchase_amount'])

#histogram
f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(ht['installments'])

temp = ht["city_id"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (18,8))
plt.title('city_id')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

ht['authorized_flag'] = ht['authorized_flag'].map({'Y':1, 'N':0})

def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(ht)
del ht
gc.collect()


train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')

#Simple Exploration : merchants.csv

merchant = pd.read_csv("../input/merchants.csv")
print("shape of merchant : ",merchant.shape)

merchant.head()
# checking missing data
total = merchant.isnull().sum().sort_values(ascending = False)
percent = (merchant.isnull().sum()/merchant.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(merchant['numerical_1'])

#histogram
f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(merchant['numerical_2'])

temp = merchant["category_1"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_1 in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

#most_recent_sales_range, most_recent_purchases_range, category_4
temp = merchant["category_2"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_2 in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

#most_recent_sales_range, most_recent_purchases_range, category_4
temp = merchant["most_recent_sales_range"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('most_recent_sales_range in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

#most_recent_sales_range, most_recent_purchases_range, category_4
temp = merchant["most_recent_purchases_range"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('most_recent_purchases_range in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

#most_recent_sales_range, most_recent_purchases_range, category_4
temp = merchant["category_4"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_4 in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


#Simple Exploration : new_merchants.csv
new_merchant = pd.read_csv("../input/new_merchant_transactions.csv")
print("shape of new_merchant_transactions : ",new_merchant.shape)

new_merchant.head()
# checking missing data
total = new_merchant.isnull().sum().sort_values(ascending = False)
percent = (new_merchant.isnull().sum()/new_merchant.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

new_merchant['authorized_flag'] = new_merchant['authorized_flag'].map({'Y':1, 'N':0})

def aggregate_new_transactions(new_trans):    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_trans = aggregate_new_transactions(new_merchant)

train = pd.merge(train, new_trans, on='card_id', how='left')
test = pd.merge(test, new_trans, on='card_id', how='left')

#Make a Baseline model
#kernel : https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you

use_cols = [col for col in train.columns if col not in ['card_id', 'first_active_month']]

train = train[use_cols]
test = test[use_cols]

features = list(train[use_cols].columns)
categorical_feats = [col for col in features if 'feature_' in col]

for col in categorical_feats:
    print(col, 'have', train[col].value_counts().shape[0], 'categories.')

from sklearn.preprocessing import LabelEncoder
for col in categorical_feats:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))    

df_all = pd.concat([train, test])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = train.shape[0]

train = df_all[:len_train]
test = df_all[len_train:]

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


lgb_params = {"objective" : "regression", "metric" : "rmse", 
               "max_depth": 7, "min_child_samples": 20, 
               "reg_alpha": 1, "reg_lambda": 1,
               "num_leaves" : 64, "learning_rate" : 0.005, 
               "subsample" : 0.8, "colsample_bytree" : 0.8, 
               "verbosity": -1}

FOLDs = KFold(n_splits=20, shuffle=True, random_state=1989)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_splits
    

print(np.sqrt(mean_squared_error(oof_lgb, target)))


cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = predictions_lgb 
sub_df.to_csv("submission_lgb.csv", index=False)

import xgboost as xgb

xgb_params = {'eta': 0.005, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}



FOLDs = KFold(n_splits=20, shuffle=True, random_state=1989)

oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))


for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = xgb.DMatrix(data=train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(data=train.iloc[val_idx], label=target.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 10000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=50, verbose_eval=1000)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb += xgb_model.predict(xgb.DMatrix(test), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_splits

np.sqrt(mean_squared_error(oof_xgb, target))


print('lgb', np.sqrt(mean_squared_error(oof_lgb, target)))
print('xgb', np.sqrt(mean_squared_error(oof_xgb, target)))

total_sum = 0.5 * oof_lgb + 0.5 * oof_xgb
print("CV score: {:<8.5f}".format(mean_squared_error(total_sum, target)**0.5))

sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df.to_csv("submission_ensemble.csv", index=False)


    

