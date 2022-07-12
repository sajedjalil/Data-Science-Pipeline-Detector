import numpy as np
import pandas as pd


df = pd.read_csv('../input/cities.csv')

ixy  = df.values # CityId(or index), X, Y
R    = []        # Result list of CityId

for i in range(len(ixy)):
    if i % 1000 == 0:
        print(i) # log, initial_len(ixy) = 197769
    d    = (ixy[:, 1] - ixy[0, 1]) ** 2 + (ixy[:, 2] - ixy[0, 2]) ** 2 # the distance from last choiced city
    ixyd = np.concatenate([ixy, d.reshape(-1, 1)], axis=1)
    argi = np.argsort(ixyd[:, 3]) # Argsorted index by the distance
    ixyd = ixyd[argi]
    R.append(int(ixyd[0, 0]))
    ixy = ixyd[1:, :-1] # update cities data

s = pd.read_csv('../input/sample_submission.csv')
s['Path'] = np.array(R + [0]) # Return 0

s.to_csv('simple_nearest.csv', index=False)
