import warnings
warnings.filterwarnings('ignore',category=UserWarning)

import gc
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgbm
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)

ID = 'id'
TARGET = 'revenue'
NFOLDS = 5
SEED = 32478
NROWS = None
DATA_DIR = '../input'

TRAIN_FILE = f'{DATA_DIR}/train.csv'
TEST_FILE = f'{DATA_DIR}/test.csv'

def reduceMemUsage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def uniqueValues(df, col, key):
    all_values = []
    for record in df[col]:
        lst = [d[key] for d in record]
        all_values.extend(lst)
    all_values = np.array(all_values)
    unique, counts = np.unique(all_values, return_counts=True)
    return pd.DataFrame({ 'Value': unique, 'Counts': counts })

def fixYear(row):
    year = int(row.split('/')[2])
    return row[:-2] + str(year + (2000 if year <= 19 else 1900))

def extractField(row, value):
    if row is np.nan: return 0
    return 1 if value in row else 0

def createCountMap(col, full, key=None, value=None):
    count_map = {}
    for row in full[col]:
        values = [i for i in row if i[key] == value] if key and value else row
        for ch in values:
            idx = str(ch['id'])
            if idx in count_map:
                count_map[idx] += 1
            else:
                count_map[idx] = 1
    return count_map

def countTopId(row, counts, top, key=None, value=None):
    count_sum = 0
    values = [i for i in row if i[key] == value] if key and value else row
    for ch in values[:top]:
        count_sum += counts[str(ch['id'])]
    return count_sum

def countTop(row, counts, top):
    count_sum = 0
    for ch in row[:top]:
        count_sum += counts[counts['Value'] == ch['id']]['Counts'].iloc[0]
    return count_sum

def prepareModelInput():
    train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
    test = pd.read_csv(TEST_FILE, nrows=NROWS)

    train = reduceMemUsage(train)
    test = reduceMemUsage(test)

    series_cols = ['belongs_to_collection', 'genres', 'production_companies',
                   'production_countries', 'spoken_languages', 'Keywords',
                   'cast', 'crew']

    for df in [train, test]:
        for column in series_cols:
            df[column] = df[column].apply(lambda s: [] if pd.isnull(s) else eval(s))

    full = pd.concat([train, test], sort=False)

    genres_unique = uniqueValues(full, 'genres', 'name').sort_values(by='Counts', ascending=False)
    languages_unique = uniqueValues(full, 'spoken_languages', 'iso_639_1').sort_values(by='Counts', ascending=False)
    top_languages = languages_unique.iloc[:8]
    cast_unique = uniqueValues(full, 'cast', 'id')
    companies_unique = uniqueValues(full, 'production_companies', 'id')
    countries_unique = uniqueValues(full, 'production_countries', 'iso_3166_1')
    top_countries = countries_unique.iloc[:5]
    keywords_unique = uniqueValues(full, 'Keywords', 'id')

    drct_count = createCountMap('crew', full, 'job', 'Director')
    test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/00'

    for df in [train, test]:
        df['genres_list'] = df['genres'].apply(lambda row: ','.join(d['name'] for d in row))
        df['genres_count'] = df['genres'].apply(lambda x: len(x))
        for g in genres_unique['Value'].values:
            df['genre_' + g] = df['genres_list'].apply(extractField, args=(g,))

        df['log_popularity'] = np.log1p(df['popularity'])
        df['log_budget'] = np.log1p(df['budget'])
        
        df['prod_countries_list'] = df['production_countries'].apply(lambda row: ','.join(d['iso_3166_1'] for d in row))
        for l in top_countries['Value'].values:
            df['country_' + l] = df['prod_countries_list'].apply(extractField, args=(l,))

        df['languages_list'] = df['spoken_languages'].apply(lambda row: ','.join(d['iso_639_1'] for d in row))
        for l in top_languages['Value'].values:
            df['lang_' + l] = df['languages_list'].apply(extractField, args=(l,))

        df['has_homepage'] = df['homepage'].apply(lambda v: pd.isnull(v) == False)
        
        df['release_date'] = df['release_date'].apply(fixYear)
        df['release_date'] = pd.to_datetime(df['release_date'])

        date_parts = ['year', 'weekday', 'month', 'weekofyear', 'day', 'quarter']
        for part in date_parts:
            part_col = 'release_date' + '_' + part
            df[part_col] = getattr(df['release_date'].dt, part).astype(int)

        df['budget_to_popularity'] = df['budget'] / df['popularity']
        df['budget_to_runtime'] = df['budget'] / df['runtime']
        df['budget_to_year'] = df['budget'] / (df['release_date_year'] ** 2)
        df['budget_to_month'] = df['budget'] / (df['release_date_month'] ** 2)
        
        df['collection'] = df['belongs_to_collection'].apply(lambda row: ','.join(d['name'] for d in row))
        df['has_collection'] = df['collection'].apply(lambda v: 1 if v else 0)

        ### START_TEST
        # df['releaseYear_popularity_ratio'] = df['release_date_year'] / df['popularity']
        # df['releaseYear_popularity_ratio2'] = df['popularity'] / df['release_date_year']
    
        # df['runtime_to_mean_year'] = df['runtime'] / df.groupby('release_date_year')['runtime'].transform('mean')
        # df['popularity_to_mean_year'] = df['popularity'] / df.groupby('release_date_year')['popularity'].transform('mean')
        # df['budget_to_mean_year'] = df['budget'] / df.groupby('release_date_year')['budget'].transform('mean')
        ###

        df['cast_counts'] = df['cast'].apply(lambda x: len(x))
        df['cast_freq'] = df['cast'].apply(countTop, args=(cast_unique, 4))
    
        df['companies_freq'] = df['production_companies'].apply(countTop, args=(companies_unique,100))
        df['directors_freq'] = df['crew'].apply(countTopId, args=(drct_count, 100,'job','Director'))

        df['keywords_counts'] = df['Keywords'].apply(lambda x: len(x))
        df['keywords_freq'] = df['Keywords'].apply(countTop, args=(keywords_unique, 20))

    # for col in train_texts.columns:
    #     vectorizer = TfidfVectorizer(
    #                 sublinear_tf=True,
    #                 analyzer='word',
    #                 token_pattern=r'\w{1,}',
    #                 ngram_range=(1, 2),
    #                 min_df=5
    #     )
    #     vectorizer.fit(list(train_texts[col].fillna('')) + list(test_texts[col].fillna('')))
    #     train_col_text = vectorizer.transform(train_texts[col].fillna(''))
    #     test_col_text = vectorizer.transform(test_texts[col].fillna(''))
    #     model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)
    #     oof_text, prediction_text = train_model(train_col_text, test_col_text, y, params=None, model_type='sklearn', model=model)
        
    #     X[col + '_oof'] = oof_text
    #     X_test[col + '_oof'] = prediction_text

    train.loc[train['imdb_id'] == 'tt1107828', 'runtime'] = 130
    test.loc[test['imdb_id'] == 'tt0082131', 'runtime'] = 93
    test.loc[test['imdb_id'] == 'tt3132094', 'runtime'] = 91
    test.loc[test['imdb_id'] == 'tt0078010', 'runtime'] = 100
    test.loc[test['imdb_id'] == 'tt2192844', 'runtime'] = 90

    # data fixes from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3
    train_fix_budget = [(90, 30000000), (118, 60000000), (149, 18000000), (464, 20000000), (470, 13000000),
                        (513, 930000), (797, 8000000), (819, 90000000), (850, 90000000), (1112, 7500000), (1131, 4300000),
                        (1359, 10000000), (1542, 1), (1542, 15800000), (1571, 4000000), (1714, 46000000), (1721, 17500000),
                        (2268, 17500000), (2602, 31000000), (2612, 15000000), (2696, 10000000), (2801, 10000000)]
    train_fix_revenue = [(16, 192864), (313, 12000000), (451, 12000000), (1865, 25000000), (2491, 6800000)]
    for idx, budget in train_fix_budget:
        train.loc[train['id'] == idx, 'budget'] = budget
    for idx, revenue in train_fix_revenue:
        train.loc[train['id'] == idx, 'revenue'] = revenue
    test_fix_budget = [(3889, 15000000), (6733, 5000000), (3197, 8000000), (6683, 50000000), (5704, 4300000),
                       (6109, 281756), (7242, 10000000), (7021, 17540562), (5591, 4000000), (4282, 20000000)]
    for idx, budget in test_fix_budget:
        test.loc[test['id'] == idx, 'budget'] = budget

    remove_cols = series_cols + [
        'original_title', 'poster_path', 'imdb_id', 'status',
        'homepage', 'title', 'release_date', 'genres_list',
        'overview', 'tagline', 'languages_list']

    train.drop(remove_cols, axis=1, inplace=True)
    test.drop(remove_cols, axis=1, inplace=True)

    del full, remove_cols, drct_count, series_cols
    del genres_unique, languages_unique, top_languages, cast_unique, companies_unique

    for col in train.columns[train.dtypes == 'object']:
        encoder = LabelEncoder()
        encoder.fit(np.concatenate([train[col], test[col]]))
        train[col] = encoder.transform(train[col].astype(str))
        test[col] = encoder.transform(test[col].astype(str))

    train = reduceMemUsage(train)
    test = reduceMemUsage(test)

    y = np.log1p(train[TARGET])

    del train[ID], train[TARGET]
    gc.collect()

    return train, test, y

def trainModel(x, test, y, folds, num_round=50, early_stopping_rounds=50):
    start_time = time.time()

    features = [f for f in x.columns if f is not ID]
    feature_importance_df = pd.DataFrame()
    oof_preds = np.zeros(x.shape[0])
    predictions = np.zeros(test.shape[0])
    classifier_models = []
    X_test = test.drop(ID, axis=1)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': SEED,
        'learning_rate': 0.05,
        'verbose': -1,
        'num_leaves': 60,
        'min_data_in_leaf': 60,
        'max_depth': 10,
    }

    for n_fold, (train_idx, validation_idx) in enumerate(folds.split(x)):
        X_train, X_validation = x.iloc[train_idx], x.iloc[validation_idx]
        y_train, y_validation = y.iloc[train_idx], y.iloc[validation_idx]

        clf = lgbm.LGBMRegressor(**params, n_estimators=num_round)
        clf.fit(X_train, y_train,  eval_set=[(X_train, y_train), (X_validation, y_validation)],
                eval_metric='rmse', verbose=False, early_stopping_rounds=early_stopping_rounds)
        
        preds = clf.predict(X_validation, num_iteration=clf.best_iteration_)
        score = mean_squared_error(y_validation, preds) ** 0.5
        oof_preds[validation_idx] = preds
        classifier_models.append(clf)

        predictions += clf.predict(X_test)

        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = features
        fold_importance_df['importance'] = clf.feature_importances_
        fold_importance_df['fold'] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold {:2d} - {:.6f}'.format(n_fold + 1, score))

        del clf
        gc.collect()

    score = mean_squared_error(y, oof_preds) ** 0.5
    print('Mean score: {:.6f}'.format(score))

    elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    print('Elapsed time: {}'.format(elapsed))

    predictions = np.expm1(predictions / folds.n_splits)

    return predictions, oof_preds, feature_importance_df

def displayImportances(feature_importance_df, save_to_csv=False):
    feats = feature_importance_df[['feature', 'importance']]
    cols = feats.groupby('feature').mean().sort_values(by='importance', ascending=False)[:50].index

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(8,10))
    sns.barplot(x='importance', y='feature',
                data=best_features.sort_values(by='importance', ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

    if save_to_csv:
        best_features.groupby('feature').mean().sort_values(by='importance', ascending=False) \
            .to_csv('feature_importances.csv', index=True)

def saveOutput(test, predictions):
    output = pd.DataFrame({ ID: test[ID].values, TARGET: predictions })
    output.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    gc.enable()
    train, test, y = prepareModelInput()

    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    predictions, oof_preds, feature_importance_df = trainModel(train, test, y, folds,
        num_round=5000, early_stopping_rounds=200)
    saveOutput(test, predictions)

    displayImportances(feature_importance_df)
