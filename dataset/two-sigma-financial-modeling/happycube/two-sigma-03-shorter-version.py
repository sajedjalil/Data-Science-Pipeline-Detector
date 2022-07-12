from collections import deque
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
import kagglegym
import numpy as np
import pandas as pd
import random
import math

random.seed(a=52)

class Model:

    def __init__(self, df, model):

        self.cols = ['technical_20', 'technical_30']
        self.scale = MinMaxScaler()
        
        self.regr = model

        df_cols = df[self.cols]
        self.means = df_cols.mean()
        
        df_cols = df_cols.fillna(self.means)
        x = self.scale.fit_transform(df_cols.values)

        self.regr.fit(x, df.y.values)

    # ------------------------------

    def predict(self, df):
        df_cut = df[self.cols].fillna(self.means)
        
        x = self.scale.transform(df_cut.values)
        
        return self.regr.predict(x)

env = kagglegym.make()
observation = env.reset()

train = observation.train

# build dataframe with volatility for each day

df = train[['timestamp', 'y', 'technical_20', 'technical_30']].copy()

grouped = df.groupby('timestamp')

stdgroup = grouped[['technical_20', 'technical_30']].std()
stdgroup_mean = stdgroup.rolling(window=3, win_type='triang').mean().mean(axis=1)

df_vol = pd.DataFrame({'vol': stdgroup_mean}, index=stdgroup.index)

regressors = [(SGDRegressor(loss = 'epsilon_insensitive', fit_intercept = False, random_state = 52), [-math.inf, 33]),
              (SGDRegressor(loss = 'huber', fit_intercept = False, random_state = 52), [33, 66]),
              (Ridge(alpha = 200, random_state = 52), [66, math.inf])]

models = []

for reg, rawlimits in regressors:
    limits = [0, 0]
    limits[0] = np.nanpercentile(df_vol.vol, rawlimits[0]) if not (np.isinf(rawlimits[0])) else rawlimits[0]
    limits[1] = np.nanpercentile(df_vol.vol, rawlimits[1]) if not (np.isinf(rawlimits[1])) else rawlimits[1]
    
    subset = (df_vol.vol > limits[0]) & (df_vol.vol <= limits[1])

    df_subset = train.loc[train.timestamp.isin(df_vol[subset].index)]
    
    models.append((Model(df_subset, reg), limits.copy()))

vol = deque([0, 0, 0], 3)

while True:
    df = observation.features
    target = observation.target
    
    vol.append((df.technical_20.std() + df.technical_30.std()) / 2)
    curvol = sum(vol) / len(vol)
    
    for model, bounds in models:
        if curvol > bounds[0] and curvol <= bounds[1]:
            target.y = model.predict(df)
            break
            
    observation, reward, done, info = env.step(target)
    
    if done:
        break

print(info)