import kagglegym
import numpy as np
import pandas as pd

# Additional includes
from sklearn import linear_model


################################################
# Initialize environment 
#
# The "environment" is our interface for code competitions
env = kagglegym.make()
# We get our initial observation by calling "reset"
observation = env.reset()

################################################
# Compute features
df = observation.train[['timestamp','id','y','technical_20','technical_30']].sort_values(by = ['id','timestamp'])
df['feat2030'] = df['technical_20']-df['technical_30'] # feature2030
df['y_hat1'] = (df['feat2030'] - (0.92 * df['feat2030'].shift(1))) / 0.07 # cc05 reconstructed past y

# Add lagged predictors
max_lag = 2
for lag in np.arange(2,max_lag+1):
    df['y_hat{}'.format(lag)] = df['y_hat1'].shift(lag-1)

all_feats = ['y_hat{}'.format(max_lag-lag) for lag in range(max_lag)]
print(all_feats)
# Handle NAN
fillnawith = df.mean(axis=0)
df.fillna(fillnawith, inplace = True)
df.fillna(0, inplace=True)

################################################
# Model training
print('Training!')
model = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0,100,1000,10000))
model = model.fit(df[all_feats], df.y)

# Model training
print('Training model t20!')
modelt20 = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0,100,1000,10000))
modelt20 = modelt20.fit(df['technical_20'].values.reshape(-1,1), df.y)
t20mean = df['technical_20'].mean()

################################################
# Prepare for predictions
temp = df[df['timestamp'] > df['timestamp'].max()-max_lag]
feat2030 = temp.pivot(index = 'id', columns = 'timestamp', values = 'feat2030') # past values of feat
y_hat = temp.pivot(index = 'id', columns = 'timestamp', values = 'y_hat1') # past values of y

print('Predicting!')

while True:
    # get features
    test  = observation.features[['timestamp','id','technical_20','technical_30']].set_index('id')
    # get timestamp
    t = test["timestamp"][0]
    
    preds = observation.target
    try:
        # compute autoregressive features and expand datasets (for new ids)
        feat2030 = pd.concat([feat2030,test['technical_20']-test['technical_30']], axis = 1, join = 'outer')
        feat2030.rename(columns={0: t}, inplace = True)
        y_hat = pd.concat([y_hat,(feat2030[t] - (0.92 * feat2030[t-1]))/ 0.07], axis = 1, join = 'outer')
        y_hat.rename(columns={0: t}, inplace = True)
    
        
        # create test features dataset
        Xtest = pd.concat([test,y_hat.loc[:,(t-max_lag+1):t]],axis = 1, join = 'inner').iloc[:,-max_lag:]
        Xtest = pd.DataFrame(index = Xtest.index, data = Xtest.values, columns = all_feats) 
        # handle NANs
        Xtest.fillna(fillnawith, inplace = True)
        Xtest.fillna(0, inplace=True) 
        
        # predict!
        # preds = observation.target.reset_index().set_index('id')
        preds['y'] = model.predict(Xtest).clip(-0.0475,0.0475)
    except:
        # jumping timestamps
        print('Using T20')
        Xtest = np.array(observation.features['technical_20'].fillna(t20mean).fillna(0).values).reshape(-1,1)
        preds['y'] = modelt20.predict(Xtest).clip(-0.0475,0.0475)

    
    
    if t % 100 == 0:
        print("Timestamp #{}".format(t))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    #print(preds.reset_index().set_index('index').head())
    #print(preds.reset_index().set_index('index').rename_axis(None).head())
    #observation, reward, done, info = env.step(preds.reset_index().set_index('index').rename_axis(None))
    observation, reward, done, info = env.step(preds)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break