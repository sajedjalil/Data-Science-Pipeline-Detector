import os

import pandas as pd
import xgboost as xgb
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

DATA_DIR = '../input'

train_raw = pd.read_json(os.path.join(DATA_DIR, 'train.json'))
test_raw = pd.read_json(os.path.join(DATA_DIR, 'test.json'))
full_raw = pd.concat([train_raw, test_raw])

interest_mapping = {'low': 0, 'medium': 1, 'high': 2}
target = train_raw.interest_level.apply(lambda x: interest_mapping[x])


def feature_processing(initial_df):
    df = initial_df.copy()
    baseline_columns = ['bathrooms', 'bedrooms', 'price', 'latitude', 'longitude']
    categorical = ['building_id', 'manager_id', 'street_address', 'display_address']
    to_count = ['features', 'description', 'photos']
    date_columns = ['year', 'month', 'day', 'hour']
    df.created = df.created.apply(pd.to_datetime)
    df['year'] = df.created.dt.year
    df['month'] = df.created.dt.month
    df['day'] = df.created.dt.day
    df['hour'] = df.created.dt.hour
    columns = baseline_columns + categorical + date_columns
    for col in categorical:
        df[col] = pd.factorize(df[col])[0]
    for col in to_count:
        df['cnt_{}'.format(col)] = df[col].apply(len)
        columns.append('cnt_{}'.format(col))
    cv = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, max_features=50)
    features = cv.fit_transform(df['features'])

    data = sp.hstack([df[columns].values, features]).tocsr()
    train, test = data[:len(train_raw)], data[len(train_raw):]
    return train, test

train, test = feature_processing(full_raw)

clf = xgb.XGBClassifier(n_estimators=500, max_depth=5, objective='multi:softprob', subsample=0.9, colsample_bytree=0.7)

# cv_scores = cross_val_score(clf, train, target, scoring='neg_log_loss', cv=3)
# print(cv_scores.mean())

clf.fit(train, target)
pred = clf.predict_proba(test)


def make_sumbission(path, pred):
    subm = pd.DataFrame(pred, columns=['low', 'medium', 'high'])
    subm['listing_id'] = test_raw.listing_id.values
    subm.to_csv(path, index=False)

make_sumbission('baseline.csv', pred)
