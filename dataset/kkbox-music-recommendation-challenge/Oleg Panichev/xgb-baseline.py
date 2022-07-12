import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tqdm import tqdm
import xgboost as xgb

data_path = '../input/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')

# Preprocess songs data
songs_genres = np.array(songs['genre_ids']\
    .apply(lambda x: [int(v) for v in str(x).split('|')]))
genres_list = songs_genres.ravel().unique()
print('Number of genres: ' + str(len(genres_list)))

ohe_genres = np.zeros((len(songs_genres), len(genres_list)))
for s_i, s_genres in enumerate(songs_genres):
    for genre in s_genres:
        g_i = genres_list.find(genre)
        ohe_genres[s_i, g_i] = 1
        
for g_i, g in enumerate(genres_list):
    songs['genre_' + str(g)] = ohe_genres[:, g_i]
print(songs.head())
songs = songs.drop(['genre_ids'], axis=1)

song_cols = songs.columns
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')
del songs

# Preprocess members data
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
member = members.drop(['registration_init_time', 'expiration_date'], axis=1)

members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')
del members

# Preprocess dataset
train = train.fillna(-1)
test = test.fillna(-1)

cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

        print(col + ': ' + str(len(train_vals)) + ', ' + str(len(test_vals)))

print(train.head())
print(test.head())

X_train = np.array(train.drop(['target'], axis=1))
y_train = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, \
    test_size=0.2, random_state=0)

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid) 
d_test = xgb.DMatrix(X_test)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
del X_train, X_valid, y_train, y_valid

# Train model, evaluate and make predictions
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.75
params['max_depth'] = 16
params['silent'] = 1
params['eval_metric'] = 'auc'

model = xgb.train(params, d_train, 100, watchlist, early_stopping_rounds=20, \
    maximize=True, verbose_eval=5)

p_test = model.predict(d_test)

# Prepare submission
print(len(ids), len(p_test))
subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv', index=False)
