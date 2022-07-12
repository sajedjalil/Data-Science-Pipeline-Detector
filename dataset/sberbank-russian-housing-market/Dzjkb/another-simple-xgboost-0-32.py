import numpy as np
import pandas as pd
import xgboost as xgb
import math

def rmsle(preds, dtrain):
    labels = np.array(dtrain.get_label().tolist())
    preds = np.array(preds.tolist())

    return 'rmsle', math.sqrt(float(np.mean(np.square(np.log(labels + 1) - np.log(preds + 1)))))

def factorize(df):
    rest = df.select_dtypes(exclude=['object'])
    categorical = df.select_dtypes(include=['object'])
    categorical = categorical.apply(lambda x: pd.factorize(x)[0], axis=1)

    return pd.concat([rest, categorical], axis=1)

def preprocess(train_df, macro_df, test_df):
    y = train_df.loc[:, 'price_doc'].values
    test_id = test_df['id']

    train_df.drop(['price_doc', 'id'], inplace=True, axis=1)
    test_df.drop(['id'], inplace=True, axis=1)

    no_examples = len(train_df)
    all_df = pd.concat([train_df, test_df])

    # join with best macroeconomic features from
    # https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity

    merge_cols = ["timestamp", "balance_trade", "balance_trade_growth", "eurrub", \
                  "average_provision_of_build_contract", "micex_rgbi_tr", "micex_cbi_tr", \
                  "deposits_rate", "mortgage_value", "mortgage_rate", "income_per_cap", \
                  "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

    all_df = pd.merge(all_df, macro_df[merge_cols], on=['timestamp'], sort=False, how='left')

    # extract year, month, day from timestamp
    all_df['year'] = all_df['timestamp'].dt.year
    all_df['month'] = all_df['timestamp'].dt.month
    all_df['day'] = all_df['timestamp'].dt.day
    all_df.drop(['timestamp'], inplace=True, axis=1)

    # fill NaNs
    all_df.fillna(all_df.median(), inplace=True)

    # encode categorical variables
    all_df = factorize(all_df)

    train_df = all_df.iloc[:no_examples]
    test_df = all_df.iloc[no_examples:]

    return train_df, y, test_df, test_id

# ==========================
# Entry point
# ==========================

train_x_df = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test_x_df = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
macro_df = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])

train_x_df, train_y_np, test_x, test_id = preprocess(train_x_df, macro_df, test_x_df)

# get validation set

no_examples = len(train_x_df)
slice_index = no_examples * 2 // 3 # determines the split ratio

rng = np.random.RandomState(4283759) # random number, seed is hardcoded so
# the training/validation split is the same each run

idx = np.arange(no_examples)
rng.shuffle(idx)
train_x_np = train_x_df.values
print(train_x_np.shape)

train_x = train_x_np[idx[:slice_index], :]
train_y = train_y_np[idx[:slice_index]]
val_x = train_x_np[idx[slice_index:], :]
val_y = train_y_np[idx[slice_index:]]

# boost 'em

xgb_train = xgb.DMatrix(train_x, train_y, feature_names=train_x_df.columns)
xgb_val = xgb.DMatrix(val_x, val_y, feature_names=train_x_df.columns)

xgb_params = {
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0,
    'eta': 0.05,
    'lambda': 1.5, # l2 reg
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 1
}

num_rounds = 450

model = xgb.train(xgb_params,
                  xgb_train,
                  num_boost_round=num_rounds,
                  evals=[(xgb_val, 'val_set')],
                  feval=rmsle)
                  # early_stopping_rounds=20)
                  
# make predictions

xgb_test = xgb.DMatrix(test_x.values, feature_names=test_x.columns)
preds = model.predict(xgb_test)
preds_df = pd.DataFrame({'id': test_id, 'price_doc': preds})
preds_df.to_csv('preds.csv', index=False)
