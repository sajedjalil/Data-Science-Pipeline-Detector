# %% [markdown]
# 

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews


#***********************************import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.callbacks import Callback
from keras.applications.xception import Xception
from keras.layers import multiply

import keras
from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.regularizers import l2
from keras.layers.core import Dense, Lambda
from keras.layers.merge import concatenate, add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.optimizers import SGD

# %% [code] {"scrolled":true}
env = twosigmanews.make_env()
(market_train, newsdf) = env.get_training_data()

# %% [code]
cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10','sentimentNegative', 'wordCount','sentimentWordCount']

    
newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
#newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
#newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']

# combine multiple news reports for same assets on same day
newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
del newsdf



# %% [code]
print('merging news and data...')

market_train = pd.merge(market_train, newsgp, how='left', on=['time', 'assetCode'], copy=False)
del newsgp


# %% [code]
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=23)


# %% [markdown]
# # Handling categorical variables

# %% [code]
def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]

print(encoders)


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets

print(embed_sizes)


# %% [markdown]
# # Handling numerical variables

# %% [code]
from sklearn.preprocessing import StandardScaler
 
market_train[num_cols] = market_train[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()

#col_mean = market_train[col].mean()
#market_train[col].fillna(col_mean, inplace=True)
scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])


# %% [markdown]
# # Define NN Architecture

# %% [markdown]
# Todo: add explanaition of architecture

# %% [code]
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
#categorical_logits = Dense(32,activation='relu')(categorical_logits)
#categorical_logits =Dropout(0.5)(categorical_logits)
#categorical_logits =BatchNormalization()(categorical_logits)
categorical_logits = Dense(32,activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(len(num_cols),), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(128,activation='relu')(numerical_logits)

#numerical_logits=Dropout(0.3)(numerical_logits)
#numerical_logits = BatchNormalization()(numerical_logits)
#numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits=Dropout(0.3)(numerical_logits)
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128,activation='relu')(numerical_logits)


numerical_logits = Dense(64,activation='relu')(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)

# %% [code]
# Lets print our model
model.summary()

# %% [code]
def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)

# %% [code]
class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')

# %% [code]
class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint("model.hdf5",monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1),
            swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

# %% [markdown]
# # Train NN model

# %% [code]
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

"""
epochs = 10
snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)
batch_size = 32
swa = SWA('model_swa.hdf5',6)
history = model.fit(X_train,y_train.astype(int),
                    validation_data=(X_valid,y_valid.astype(int)),
                    epochs=epochs,
                    #batch_size=batch_size,
                    callbacks=snapshot.get_callbacks(),shuffle=True,verbose=2)
                    
early_stop = EarlyStopping( mode = 'max',patience=15, verbose=1)
check_point = ModelCheckpoint('model.hdf5', mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau( mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
#check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
#early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X_train,y_train.astype(int),
                    validation_data=(X_valid,y_valid.astype(int)), 
                    epochs=15,
                    callbacks=[check_point,reduce_lr,early_stop], 
                    verbose=2)

model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=10,
          verbose=True,
          callbacks=[early_stop,check_point]) 
"""


check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=10,
          verbose=True,
          callbacks=[early_stop,check_point]) 

# %% [code]
"""
try:
    print('using swa weight model')
    model.load_weights('model_swa.hdf5')
except:
    model.load_weights('model.hdf5')
"""

# %% [markdown]
# # Evaluation of Validation Set

# %% [code]
# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()

# %% [code]
# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)

# %% [markdown]
# # Prediction

# %% [code]
days = env.get_prediction_days()

# %% [code]
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    # combine multiple news reports for same assets on same day
    #market_obs_df['bartrend'] = market_obs_df['close'] / market_obs_df['open']
    #market_obs_df['average'] = (market_obs_df['close'] + market_obs_df['open'])/2
    #market_obs_df['pricevolume'] = market_obs_df['volume'] * market_obs_df['close']

    news_obs_df['assetCode'] = news_obs_df['assetCodes'].map(lambda x: list(eval(x))[0])
    #news_obs_df['position'] = news_obs_df['firstMentionSentence'] / news_obs_df['sentenceCount']
    #news_obs_df['coverage'] = news_obs_df['sentimentWordCount'] / news_obs_df['wordCount']
    newsgp = news_obs_df.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    market_obs_df = pd.merge(market_obs_df, newsgp, how='left', on=['time', 'assetCode'], copy=False)
    
    t = time.time()

    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))

    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num':X_num_test}
    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values
    
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = model.predict(X_test)[:,0]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')

# %% [code]
# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()