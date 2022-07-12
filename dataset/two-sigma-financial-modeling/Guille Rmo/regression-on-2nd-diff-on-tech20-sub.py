import kagglegym
import numpy as np
import pandas as pd
import sklearn.linear_model as lm

env = kagglegym.make()
observation = env.reset()
tr = observation.train
ids = tr.id.unique()

t20 = tr.pivot_table(columns="timestamp", index="id", values="technical_20")
t20d = t20.diff(axis=0).diff(axis=0).fillna(0)
tr["t20d"] = tr.apply(lambda x: t20d.loc[x.id][x.timestamp], axis=1)

modelo = lm.LinearRegression()
modelo.fit(tr[["t20d"]].fillna(0), tr["y"])

while True:
    timestamp = observation.features["timestamp"][0]
    t20 = pd.concat([t20.iloc[:, (len(t20.columns)-5):],
           observation.features.pivot_table(columns="timestamp", index="id", values="technical_20")],
          axis=1)
    t20d = t20.diff(axis=0).diff(axis=0).fillna(0)
    X = observation.features.apply(lambda x: t20d.loc[x.id][x.timestamp], axis=1)
    observation.target.y = \
        modelo.predict(X.fillna(0).values.reshape(-1,1))
    if timestamp % 50 == 0:
        print("Timestamp #{}".format(timestamp), reward)

        # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(observation.target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break