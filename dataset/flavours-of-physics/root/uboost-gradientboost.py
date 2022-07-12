
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


from hep_ml.gradientboosting import UGradientBoostingClassifier,LogLossFunction

print("Load the training/test data using pandas")


train = pd.read_csv("../input/training.csv")

#randomize the training sample
train=train.iloc[np.random.permutation(len(train))]

test = pd.read_csv('../input/test.csv')

print("Eliminate SPDhits, which makes the agreement check fail")
features= ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2',
       
       
       'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_IP',
       'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 
            'p0_eta', 'p1_eta',
       'p2_eta']


print("train a UBoost classifier")


loss_funct=LogLossFunction()
ub=UGradientBoostingClassifier(loss=loss_funct,n_estimators=100, random_state=3,learning_rate=0.245,subsample=0.7)
ub.fit(train[features],train["signal"])

print("train a Gradientboost classifier")
gb=GradientBoostingClassifier(n_estimators=120, random_state=3,learning_rate=0.256,subsample=0.7,max_features=34)
gb.fit(train[features],train["signal"])




test_probs= 0.5*ub.predict_proba(test[features])[:, 1]+0.5*gb.predict_proba(test[features])[:, 1]
result = pd.DataFrame({"id": test["id"], "prediction": test_probs})



result.to_csv('UBoost_classifier.csv', index=False, sep=',')