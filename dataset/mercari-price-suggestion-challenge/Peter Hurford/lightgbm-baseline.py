import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor


def rmsle(predictions, targets):
    predictions = np.exp(predictions) - 1
    targets = np.exp(targets) - 1
    return np.sqrt(((predictions - targets) ** 2).mean())

def rmsle_lgb(labels, preds):
    return 'rmsle', rmsle(preds, labels), False


print('Loading train...')
train = pd.read_csv('../input/train.tsv', delimiter='\t')


print('Loading test...')
test = pd.read_csv('../input/test.tsv', delimiter='\t')


print('Processing...')
# Get target and IDs
target = np.log(train['price'] + 1)
train_id = train['train_id']
test_id = test['test_id']
train.drop(['train_id', 'price'], axis=1, inplace=True)
test.drop('test_id', axis=1, inplace=True)

# Split category into subcategories by splitting on /
train_cats = train['category_name'].str.split('/', 0, expand=True)
train_cats.columns = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']
test_cats = test['category_name'].str.split('/', 0, expand=True)
test_cats.columns = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']
train = pd.concat([train.reset_index(), train_cats.reset_index()], axis=1)
test = pd.concat([test.reset_index(), test_cats.reset_index()], axis=1)
train.drop('category_name', axis=1, inplace=True)
test.drop('category_name', axis=1, inplace=True)

# Fast label encoding adapted from https://www.kaggle.com/lawrencechernin/rf-version3
print('Label Encoding...')
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], axis=0)
df.brand_name.fillna('missing', inplace=True)
df.cat1.fillna('missing', inplace=True)
df.cat1 = df.cat1.astype('category')
df.cat2.fillna('missing', inplace=True)
df.cat2 = df.cat2.astype('category')
df.cat3.fillna('missing', inplace=True)
df.cat3 = df.cat3.astype('category')
df.cat4.fillna('missing', inplace=True)
df.cat4 = df.cat4.astype('category')
df.cat5.fillna('missing', inplace=True)
df.cat5 = df.cat5.astype('category')
df.item_description = df.item_description.astype('category')
df.name = df.name.astype('category')
df.brand_name = df.brand_name.astype('category')
df.name = df.name.cat.codes
df.cat1 = df.cat1.cat.codes
df.cat2 = df.cat2.cat.codes
df.cat3 = df.cat3.cat.codes
df.cat4 = df.cat4.cat.codes
df.cat5 = df.cat5.cat.codes
df.brand_name = df.brand_name.cat.codes
df.item_description = df.item_description.cat.codes


# Split back up train and test
print('Processing...')
df_test = df.loc[df['is_train'] == 0]
df_train = df.loc[df['is_train'] == 1]
df_test = df_test.drop(['is_train'], axis=1)
df_train = df_train.drop(['is_train'], axis=1)


print('Training model...')
X, y = df_train.values, target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lgbm_params = {'n_estimators': 900, 'learning_rate': 0.15, 'max_depth': 5,
               'num_leaves': 31, 'subsample': 0.9, 'colsample_bytree': 0.8,
               'min_child_samples': 50, 'n_jobs': 3}
model = LGBMRegressor(**lgbm_params)
model.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         eval_metric=rmsle_lgb,
         early_stopping_rounds=None,
         verbose=True)

print('Generating submission...')
model = LGBMRegressor(**lgbm_params)
model.fit(X, y,
         early_stopping_rounds=None,
         verbose=True)
preds = model.predict(df_test)
preds = np.exp(preds) - 1
submit = pd.DataFrame({"test_id": test_id, "price": preds})
submit.to_csv('lgb_baseline.csv',index=False)