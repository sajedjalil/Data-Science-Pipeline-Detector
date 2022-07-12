# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import Sequence
import gc
from sklearn.preprocessing import QuantileTransformer,MinMaxScaler,StandardScaler
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()

# Merging market and train data sets.
def data_prep(market_train,news_train):
    market_train.time = market_train.time.dt.date
    news_train.time = news_train.time.dt.hour
    news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
    news_train.firstCreated = news_train.firstCreated.dt.date
    news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
    news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
    kcol = ['assetCodes','firstCreated']
    news_train = news_train.groupby(kcol, as_index=False).mean()
    market_train = pd.merge(market_train, news_train, how='left', left_on=['assetCode', 'time'], 
                            right_on=['assetCodes', 'firstCreated'])
    #Calculating relevant words based on urgency condition.
    market_train.loc[market_train['urgency'] != 1,'relevant_words'] = market_train['wordCount']*(market_train['relevance'])
    market_train.loc[market_train['urgency'] == 1,'relevant_words'] = (market_train['sentenceCount']-(market_train['firstMentionSentence']-1))*(market_train['wordCount']/market_train['sentenceCount'])
    market_train = market_train.sort_values(['assetCode', 'time_x'])
    market_train.sort_index(axis=1, inplace=True)
    market_train = market_train[market_train.columns[~market_train.columns.isin(['assetCodes','assetName','firstCreated','time_x','universe','time_y','sourceTimestamp','urgency','bodySize','companyCount','marketCommentary','sentenceCount','wordCount','firstMentionSentence','relevance','noveltyCount3D','noveltyCount5D','noveltyCount7D','volumeCounts3D' 'volumeCounts5D','volumeCounts7D','assetCodesLen'])]]
    return market_train
featureset = data_prep(market_train,news_train)

time_steps = 2
datashape = featureset.shape[1]
# Creating time series per assetcode.
featureset = featureset.groupby('assetCode').filter(lambda x: len(x['assetCode']) > time_steps)
scaler = StandardScaler()
scaler_dict = dict()
for column in featureset[featureset.columns[~featureset.columns.isin(['assetCode','returnsOpenNextMktres10'])]]:
    featureset[column] = scaler.fit_transform(featureset[[column]])
    scaler_dict[column] = scaler
# Removing outliers
featureset.drop(featureset['returnsOpenNextMktres10'].nlargest(10).index.values,inplace=True)
featureset.drop(featureset['returnsOpenNextMktres10'].nsmallest(10).index.values,inplace=True)
# Creating target data frame.
yTrain = featureset[['assetCode','returnsOpenNextMktres10']]
# Removing last time step rows per asset to prevent over-lapping in the 3d train sets.
yTrain = yTrain.groupby('assetCode').apply(lambda x: x.tail(-(time_steps-1)))
yTrain.reset_index(level=0,drop=True,inplace=True)
yTrain.drop(['assetCode'], axis=1,inplace=True)
# Converting target value to confidence value in range[-1,1].
predict_scalar = QuantileTransformer(output_distribution='normal')
predict_scalar1 = MinMaxScaler(feature_range=(-1, 1))
yTrain['returnsOpenNextMktres10'] = predict_scalar.fit_transform(yTrain[['returnsOpenNextMktres10']])
yTrain['returnsOpenNextMktres10'] = predict_scalar1.fit_transform(yTrain[['returnsOpenNextMktres10']])
xTrain = featureset[featureset.columns[~featureset.columns.isin(['assetCode','returnsOpenNextMktres10'])]]
xTrain = xTrain.fillna(xTrain.mean())
del market_train
del news_train
gc.collect()
###Model Define###
model = Sequential()
model.add(LSTM(512,return_sequences=True,activation='tanh',input_shape=(time_steps, 24)))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True,activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(128,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
batch_size = 128
class DataGenerator(Sequence):
    def __init__(self, xSet, ySet, batch_size):
        self.x = xSet
        self.y = ySet
        #This will select indices according to ySet which we have filtered for time-stepping.
        self.indices = ySet.index.values
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batchY = self.y.loc[batch_indices].values
        batchX = np.empty([time_steps,self.x.shape[1]])
        for i in batch_indices:
            # Converting indices to index locations to get rows according to time steps.
            ind_loc = self.x.index.get_loc(i)
            time_step = self.x.iloc[ind_loc:ind_loc+time_steps]
            batchX = np.dstack((batchX,time_step.values))
        batchX = batchX[:,:,1:]
        batchX = batchX.reshape(self.batch_size, time_steps, batchX.shape[1])
        return batchX, batchY
trainGenerator = DataGenerator(xTrain,yTrain,batch_size)
''' Fit generator generates 3d data on the fly and only for filtered indices to 
keep memory utilization under limit. '''
model.fit_generator(generator=trainGenerator,epochs=2,steps_per_epoch=int(len(yTrain)/batch_size),verbose=1,shuffle=False,use_multiprocessing=False)
days = env.get_prediction_days()
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    to_append = data_prep(market_obs_df,news_obs_df)
    featureset = featureset.append(to_append)
    featureset = featureset.sort_values(['assetCode'])
    asset_set = featureset.merge(predictions_template_df, on=['assetCode'], how='right')
    # Fetching last time frame for next prediction
    asset_set = asset_set.groupby('assetCode').tail(time_steps)
    # Filtering asset code which don't have enough past time steps.
    random_pred = asset_set.groupby('assetCode').filter(lambda x: len(x['assetCode']) != time_steps)
    asset_set = asset_set.groupby('assetCode').filter(lambda x: len(x['assetCode']) == time_steps)
    random_pred_assets = random_pred['assetCode'].unique().tolist()
    pred_assets = asset_set['assetCode'].unique().tolist()
    asset_set = asset_set[asset_set.columns[~asset_set.columns.isin(['assetCode','confidenceValue','returnsOpenNextMktres10'])]]
    for column in asset_set.columns:
        asset_set[column] = scaler_dict[column].transform(asset_set[[column]])
    asset_set = asset_set.fillna(asset_set.mean())
    asset_set = np.reshape(np.array(asset_set),(int(asset_set.shape[0]/time_steps), time_steps,asset_set.shape[1]))
    pred_val = model.predict(asset_set)
    pred_val = pred_val.flatten()
    print(pred_val.shape)
    pred_df = pd.DataFrame({'assetCode':pred_assets,'confidenceValue':pred_val})
    pred_df = pred_df.append(pd.DataFrame({'assetCode':random_pred_assets,'confidenceValue':np.full(len(random_pred_assets),fill_value=pred_df['confidenceValue'].mean())}),ignore_index=True)
    interm = pd.DataFrame(index=pred_df.index)
    # Using hampel tanh estimator for robustness in range[-1,1].
    interm['confidenceValue'] = np.tanh((pred_df['confidenceValue'].values-np.mean(pred_df['confidenceValue'].values))/np.std(pred_df['confidenceValue'].values))
    pred_df['confidenceValue'] = interm['confidenceValue']
    env.predict(pred_df)
env.write_submission_file()