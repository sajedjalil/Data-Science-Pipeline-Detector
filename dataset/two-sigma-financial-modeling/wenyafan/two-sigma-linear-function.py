import kagglegym
import numpy as np
import pandas as pd
from numpy.linalg import inv

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))
 
arr = observation.train.fillna(0).groupby(['id'],as_index=False).mean().values
n,m = arr.shape
X = arr[:,0:(m-1)]
X = np.append(np.ones((n,1)), X,1)

y = arr[:,(m-1):m]

t1 = inv(np.dot(X.transpose(), X))
t2 = np.dot(t1, X.transpose())
theta = np.dot(t2, y)
y_predict = np.dot(X,theta)
 
id_vector = arr[:,0:1]
 
id_y_predict =np.append(id_vector, y_predict ,1)
df = pd.DataFrame(id_y_predict, columns = ['id', 'y']) 
diff = y_predict - y
J = 0.5 * np.dot(diff.transpose(), diff)/n 

for i,r in observation.target.iterrows():
    a = df.loc[df['id'] == r['id'], 'y']
    observation.target.ix[i,1] = a.values[0]

#print(observation.target)

# The "target" dataframe is a template for what we need to predict:
 
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break