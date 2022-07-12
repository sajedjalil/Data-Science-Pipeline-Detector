import kagglegym
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input
from sklearn import linear_model

def train_test_split(data, test_size=0.2):  
    """
    This just splits data to training and validation parts
    """   
    #df = pd.DataFrame(data)    
    ntrn = round(len(data) * (1 - test_size))
    ntrn = int(ntrn)
    tt = data.iloc[0:ntrn]
    vv = data.iloc[ntrn:]
    ttrain = np.array(tt)
    vval = np.array(vv)
    return (ttrain, vval)

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
Y = train.pop('y')
mean_values = train.mean(axis=0)
standev = train.std(axis=0)
train.fillna(mean_values, inplace=True)
train = (train-mean_values)/(standev)
train = train.drop(['id','y', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)

del mean_values
del standev

(xtrain, xval) = train_test_split(train)
(ytrain, yval) = train_test_split(Y) 

del train

l = xtrain.shape[1]
encoding_dim = 12  

# this is our input placeholder
input_dat = Input(shape=(l,))
encoded = Dense(encoding_dim, activation='relu')(input_dat)
decoded = Dense(l, activation='relu')(encoded)

autoencoder = Model(input=input_dat, output=decoded)
encoder = Model(input=input_dat, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(xtrain, xtrain, nb_epoch=10, batch_size=2000, validation_data=(xval, xval),verbose=False)

reg = linear_model.Ridge(max_iter = 5000,normalize=True)
reg.fit(encoder.predict(xtrain),ytrain)

del xtrain
del ytrain
del xval
del yval


print('Running the test set')
while True:
    #print('Running for test.')
    target = observation.target
    test = observation.features
 
    mean_values = test.mean(axis=0)
    #standev = test.std(axis=0)
    test = observation.features.fillna(mean_values)
    #test = (test-mean_values)/(standev)       
    test = test.drop(['id','y', 'timestamp','fundamental_61','derived_1','fundamental_17','derived_4','fundamental_26','fundamental_23','fundamental_1'], axis=1)
    #xtest = encoder.predict(test.as_matrix())

    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    target.loc[:,'y'] = reg.predict(encoder.predict(test.as_matrix()))
    #observation.target.fillna(0, inplace=True)
    del test
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print (reward)

    observation, reward, done, info = env.step(target)
    if done:
        break
    
print(info)