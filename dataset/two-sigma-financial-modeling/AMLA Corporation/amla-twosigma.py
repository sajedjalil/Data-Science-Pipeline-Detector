import kagglegym
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from sklearn.linear_model import LinearRegression
from time import gmtime, strftime

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

train_avg = observation.train.y.mean()
print('y average value in training set:', train_avg)

train_data = observation.train
feat_cols = [c for c in observation.train.columns if '_' in c]
feat_cols = ['id', 'technical_20', 'technical_30', 'technical_13']
feat_count = len(feat_cols)

fields = ['technical_20', 'technical_30', 'technical_13', 'y']

df = observation.train[fields]

#low_y_cut = -0.086093
low_y_cut = min(train_data.y)
#high_y_cut = 0.093497
high_y_cut = max(train_data.y)

y_values_within = ((train_data['y'] > low_y_cut) & (train_data['y'] <high_y_cut))

train_cut = train_data.loc[y_values_within,:]



def y_weighted(df, score_dic, verbose = False):
    s20 = score_dic['technical_20']
    s3 = score_dic['tech3']
    syp = score_dic['y_past']
    sums = s20  + s3 + syp
    df['sums'] = (s20 * df.technical_20 + s3 * df.tech3 + syp * df. y_past)/sums
    return df['sums']

# Fill missing values
mean_values = train_cut.mean()
train_cut.fillna(mean_values, inplace=True)

train_cut['tech3'] = train_cut['technical_20'] - train_cut['technical_30'] + train_cut['technical_13']
print('tech3 range', min(train_cut['tech3']), max(train_cut['tech3']))

train_past = train_cut.groupby('id').shift(1).fillna(mean_values, inplace = True)
train_cut['tech3_past'] = train_past['tech3']
print('tech3 past range', min(train_cut['tech3_past']), max(train_cut['tech3_past']))

train_cut['y_past'] = (train_cut['tech3'] - 0.92*train_cut['tech3_past'])/0.07
print('min ', min(train_cut.y_past), 'max', max(train_cut.y_past))

col_plus = feat_cols

x_train = train_cut[col_plus]
y_train = train_cut["y"]

print(x_train.shape)
print(y_train.shape)

#x_train = train_cut[feat_cols]
#y_train = train_cut["y"]
target_cols = ['y']

print('x_train: ', len(x_train))
print('y_train: ', len(y_train))
print(train_cut.head())
print(feat_count)


b_size = 128
def keras_model(batch_size, feat_count):
    model = Sequential()
    model.add(Dense(feat_count, init='normal', input_shape = (feat_count,)))
    #odel.add(Dropout(0.2, batch_input_shape=(batch_size,feat_count)))
    model.add(Dense(100, init='normal', activation='relu', W_constraint=maxnorm(1)))
    #model.add(Dense(30, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

#from random import randint
#print(randint(0,9))

def batch_generator(batch_size, test_size = 1000, train_input=True):
    while(True):
        if train_input:
            for i in range(0, len(x_train)-batch_size, batch_size):
#                print ('row', i)
                gen_x = np.array(x_train[i:i+batch_size])
                gen_y = np.array(y_train[i:i+batch_size])
#                print(gen_x)
                
                yield gen_x, gen_y
        else:
            for i in range(0, len(x_test)-batch_size, batch_size):
                print ('row', i)
                gen_x = np.array(x_test[i:i+batch_size].fillna(mean_values, inplace=True))   
                yield gen_x

train_size = b_size*100

model_dic = {}
score_dic = {}
for col_plus in train_cut.columns:
    y = np.array(train_cut.y.values.reshape((len(train_cut),1)))
    x = np.array(train_cut[col_plus].values.reshape((len(train_cut), 1)))
    model = LinearRegression()
    model.fit(x,y)
    model_dic[col_plus] = model
    score_dic[col_plus] = model.score(x,y)*100.
    model = keras_model(b_size, feat_count)
    fit = model.fit_generator(generator=batch_generator(b_size, train_input=True), nb_epoch=2, samples_per_epoch=train_size, verbose = False)

ymax = max(observation.train.y)
ymin = min(observation.train.y)
mean_values.y = observation.train.y.mean()
observation.target.y = train_avg
observation, reward, done, info = env.step(observation.target)
print('First reward:', reward)
REW = False
if REW:
    rews = np.array(range(8192))
    rews[:] = 0.
    rews = rews.astype(float)

while True:
    target = observation.target
    o_size = len(target.y)
#    observation.features.id = observation.features.id/10000.
#    print(observation.features.id)
    x_test = np.array(observation.features[feat_cols].fillna(mean_values, inplace=True))
    x_test[x_test>2.] = 2.
    x_test[x_test<-2.] = -2.
    timestamp = observation.features["timestamp"][0]
#    print("predicting...", test_size, len(x_test))
#    print(x_test)
    for it in range(0, o_size-b_size, b_size):
#        print(it, b_size, o_size)
        target.y[target.index[it:it+b_size]] = model.predict(x_test[it:it+b_size,:], batch_size = b_size, verbose=0)
    target.y[target.index[o_size-b_size:o_size]]=model.predict(x_test[o_size-b_size:o_size,:], batch_size = b_size, verbose=0)
#    print(target.y)
#    target.y = (target.y + train_avg)/2.
    target.y[target.y>ymax] = ymax
    target.y[target.y<ymin] = ymin
#    preds[preds.isnull()==True] = mean_values.y
#    print("Predictions : ", target.y.shape)
#    target.y.fillna(mean_values.y)
#    print('yrange:', min(target.y, axis=1), max(target.y, axis=1))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp), reward)
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print(len(x_test), x_test[0:5,:], min(x_test), max(x_test))
        print(len(target.y), target.y.head(), min(target.y), max(target.y))
    if REW:
        rews[timestamp]=reward
    if done:
        print('Done!')
        print("Public score: {}".format(info["public_score"]))
        if REW:
            print('Rewards:')
            for i in range(len(rews)):
                if rews[i]!=0. : print(rews[i], ',')
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        break