import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

df = observation.train

missing = -1
train = df.fillna(value = missing,axis = 1)
train1 = train.drop(['id','y','timestamp'],axis =1)
y = train.y.values
X = np.array(train1)

feat = ["technical_43","fundamental_10","fundamental_42","technical_0","fundamental_0","fundamental_44","fundamental_29","fundamental_34","fundamental_48","fundamental_43","fundamental_41","fundamental_30","technical_5","technical_33","technical_36","technical_6","technical_35","technical_41","technical_7","technical_20","technical_27","fundamental_62","technical_2","fundamental_21","technical_19","technical_40","technical_11","technical_17","technical_30","technical_21"
]

train2 = train1.loc[:,feat]
y_train = train.y.values
X_train = np.array(train2)

model1 = AdaBoostRegressor(n_estimators=5,learning_rate= 0.1,random_state=123)
model2 = AdaBoostRegressor(n_estimators=10,learning_rate= 0.05,random_state=456)
model3 = AdaBoostRegressor(n_estimators=15,learning_rate= 0.2,random_state=789)

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)


while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    test = observation.features[feat]
    test = test.fillna(value = missing,axis = 1)
    
    pred1 = model1.predict(test)
    pred2 = model2.predict(test)
    pred3 = model3.predict(test)
    
    pred = .3*pred1 + .4*pred2 + .3*pred3
    
    observation.target.y = pred
    
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break