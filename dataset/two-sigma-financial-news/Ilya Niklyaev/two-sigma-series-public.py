import numpy as np
import pandas as p
import itertools
import functools
from kaggle.competitions import twosigmanews
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, LSTM, Conv1D, Reshape, Flatten, SpatialDropout1D, Lambda, Input, Average
from keras.optimizers import Adam, SGD, RMSprop
from keras import losses as ls
from keras import activations as act
import keras.backend as K
import lightgbm as lgb

# fix random
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

def cleanData(market_data, news_data):
    market_data = market_data[(market_data['returnsOpenNextMktres10'] <= 1) & (market_data['returnsOpenNextMktres10'] >= -1)]
    return market_data, news_data

def prepareData(marketdf, newsdf, scaler=None):
    print('Preparing data...')
    
    print('...preparing features...')
    marketdf = marketdf.copy()
    newsdf = newsdf.copy()
    # a bit of feature engineering
    marketdf['time'] = marketdf.time.dt.strftime("%Y%m%d").astype(int)
    marketdf['bartrend'] = marketdf['close'] / marketdf['open']
    marketdf['average'] = (marketdf['close'] + marketdf['open'])/2
    marketdf['pricevolume'] = marketdf['volume'] * marketdf['close']
    
    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']

    # filter pre-2012 data, no particular reason
    marketdf = marketdf.loc[marketdf['time'] > 20120000]
    
    # get rid of extra junk from news data
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider',
                'sentenceCount','bodySize','headlineTag', 'subjects','audiences',
                'assetName', 'wordCount','sentimentWordCount', 'companyCount',
                 'coverage']
    newsdf.drop(droplist, axis=1, inplace=True)
    marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)
    
    # unstack news
    newsdf['assetCodes'] = newsdf['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    codes = []
    indices = []
    for i, values in newsdf['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indices.extend(repeat_index)
    index_df = p.DataFrame({'news_index': indices, 'assetCode': codes})
    newsdf['news_index'] = newsdf.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(newsdf, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    
    # combine multiple news reports for same assets on same day
    newsgp = news_unstack.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    # join news reports to market data, note many assets will have many days without news data
    res = p.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'], copy=False)
    res.marketCommentary = res.marketCommentary.astype(float)
    
    targetcol = 'returnsOpenNextMktres10'
    target_presented = targetcol in res.columns
    features = [col for col in res.columns if col not in ['time', 'assetCode', 'universe', targetcol]]
    
    print('...scaling...')
    if(scaler == None):
        scaler = StandardScaler()
        scaler = scaler.fit(res[features])
    res[features] = scaler.transform(res[features])

    print('...done.')
    return type('', (object,), {
        'scaler': scaler,
        'data': res,
        'x': res[features],
        'y': (res[targetcol] > 0).astype(int).values if target_presented else None,
        'features': features,
        'samples': len(res),
        'assets': res['assetCode'].unique(),
        'target_presented': target_presented
    })

def generateTimeSeries(data, n_timesteps=1):
    
    data.data[data.features] = data.data[data.features].fillna(data.data[data.features].mean())
    assets = data.data.groupby('assetCode', sort=False)
    
    def grouper(n, iterable):
        it = iter(iterable)
        while True:
           chunk = list(itertools.islice(it, n))
           if not chunk:
               return
           yield chunk
    
    def sample_generator():
        while True:
            for assetCode, days in assets:
                x = days[data.features].values
                y = (days['returnsOpenNextMktres10'] > 0).astype(int).values if data.target_presented else None
                for i in range(0, len(days) - n_timesteps + 1):
                    yield (x[i: i + n_timesteps], y[i + n_timesteps - 1] if data.target_presented else 0)
    
    def batch_generator(batch_size):
        for batch in grouper(batch_size, sample_generator()):
            yield tuple([np.array(t) for t in zip(*batch)])
    
    n_samples = functools.reduce(lambda x,y : x + y, map(lambda t : 0 if len(t[1]) + 1 <= n_timesteps else len(t[1]) - n_timesteps + 1, assets))

    return type('', (object,), {
        'gen': batch_generator,
        'timesteps': n_timesteps,
        'features': len(data.features),
        'samples': n_samples,
        'assets': list(map(lambda x: x[0], filter(lambda t : len(t[1]) + 1 > n_timesteps, assets)))
    })


def buildRNN(timesteps, features):
    i = Input(shape=(timesteps, features))
    x1 = Lambda(lambda x: x[:,:,:13])(i)
    x1 = Conv1D(16,1, padding='valid')(x1)
    x1 = GRU(10, return_sequences=True)(x1)
    x1 = GRU(10, return_sequences=True)(x1)
    x1 = GRU(10, return_sequences=True)(x1)
    x1 = GRU(10)(x1)
    x1 = Dense(1, activation=act.sigmoid)(x1)
    x2 = Lambda(lambda x: x[:,:,13:])(i)
    x2 = Conv1D(16,1, padding='valid')(x2)
    x2 = GRU(10, return_sequences=True)(x2)
    x2 = GRU(10, return_sequences=True)(x2)
    x2 = GRU(10, return_sequences=True)(x2)
    x2 = GRU(10)(x2)
    x2 = Dense(1, activation=act.sigmoid)(x2)
    x = Average()([x1, x2])
    model = Model(inputs=i, outputs=x)
    return model

def train_model_time_series(model, data, val_data=None):
    print('Building model...')
    batch_size = 4096
    
    optimizer = RMSprop()
    
    # define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
    def auc_roc(y_true, y_pred):
        value, update_op = tf.metrics.auc(y_true, y_pred)
        metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value
    
    model.compile(loss=ls.binary_crossentropy, optimizer=optimizer, metrics=['binary_accuracy', auc_roc])
    
    print(model.summary())
    
    print('Training model...')
    
    if(val_data == None):
        model.fit_generator(data.gen(batch_size),
            epochs=8,
            steps_per_epoch=int(data.samples / batch_size),
            verbose=1)
    else:
        model.fit_generator(data.gen(batch_size),
            epochs=8,
            steps_per_epoch=int(data.samples / batch_size),
            validation_data=val_data.gen(batch_size),
            validation_steps=int(val_data.samples / batch_size),
            verbose=1)

    return type('', (object,), {
        'predict': lambda x: model.predict_generator(x, steps=1)
    })

def train_model(data, val_data=None):
    print('Building model...')
    
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "num_leaves" : 60,
        "max_depth": -1,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.9,  # subsample
        "feature_fraction" : 0.9,  # colsample_bytree
        "bagging_freq" : 5,        # subsample_freq
        "bagging_seed" : 2018,
        "verbosity" : -1 }
    
    ds, val_ds = lgb.Dataset(data.x.iloc[:,:13], data.y), lgb.Dataset(val_data.x.iloc[:,:13], val_data.y)
    print('...training...')
    model = lgb.train(params, ds, 2000, valid_sets=[ds, val_ds], early_stopping_rounds=100, verbose_eval=100)
    print('...done.')
    
    return type('', (object,), {
        'model': model,
        'predict': lambda x: model.predict(x.iloc[:,:13], num_iteration=model.best_iteration)
    })

def make_predictions(data, template, model):
    if(hasattr(data, 'gen')):
        prediction = (model.predict(data.gen(data.samples)) * 2 - 1)[:,-1]
    else:
        prediction = model.predict(data.x) * 2 - 1
    predsdf = p.DataFrame({'ast':data.assets,'conf':prediction})
    template['confidenceValue'][template['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values
    return template

#### Do things ######

n_timesteps = 30

market_data, news_data = cleanData(market_train_df, news_train_df)
dates = market_data['time'].unique()
train = range(len(dates))[:int(0.85*len(dates))]
val = range(len(dates))[int(0.85*len(dates)):]

train_data_prepared = prepareData(market_data.loc[market_data['time'].isin(dates[train])], news_data.loc[news_data['time'] <= max(dates[train])])
val_data_prepared = prepareData(market_data.loc[market_data['time'].isin(dates[val])], news_data.loc[news_data['time'] > max(dates[train])], scaler=train_data_prepared.scaler)

model_gbm = train_model(train_data_prepared, val_data_prepared)

train_data_ts = generateTimeSeries(train_data_prepared, n_timesteps=n_timesteps)
val_data_ts = generateTimeSeries(val_data_prepared, n_timesteps=n_timesteps)
rnn = buildRNN(train_data_ts.timesteps, train_data_ts.features)
model_rnn = train_model_time_series(rnn, train_data_ts, val_data_ts)

day = 1
days_data = p.DataFrame({})
days_data_len = []
days_data_n = p.DataFrame({})
days_data_n_len = []
for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    print(f'Predicting day {day}')
    days_data = p.concat([days_data, market_obs_df], ignore_index=True, copy=False, sort=False)
    days_data_len.append(len(market_obs_df))
    days_data_n = p.concat([days_data_n, news_obs_df], ignore_index=True, copy=False, sort=False)
    days_data_n_len.append(len(news_obs_df))
    data = prepareData(market_obs_df, news_obs_df, scaler=train_data_prepared.scaler)
    predictions_df = make_predictions(data, predictions_template_df.copy(), model_gbm)
    if(day >= n_timesteps):
        data = prepareData(days_data, days_data_n, scaler=train_data_prepared.scaler)
        data = generateTimeSeries(data, n_timesteps=n_timesteps)
        predictions_df_s = make_predictions(data, predictions_template_df.copy(), model_rnn)
        predictions_df['confidenceValue'] = (predictions_df['confidenceValue'] + predictions_df_s['confidenceValue']) / 2
        days_data = days_data[days_data_len[0]:]
        days_data_n = days_data_n[days_data_n_len[0]:]
        days_data_len = days_data_len[1:]
        days_data_n_len = days_data_n_len[1:]
    env.predict(predictions_df)
    day += 1

env.write_submission_file()