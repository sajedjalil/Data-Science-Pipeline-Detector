import pandas as pd
from sklearn import linear_model

train = pd.read_csv("../input/training.csv", index_col='id')
test = pd.read_csv("../input/test.csv", index_col='id')

variables = ['FlightDistance',
             'FlightDistanceError',
             'LifeTime',
             'IP',
             'IPSig',
             'VertexChi2',
             'dira',
             'pt',
             'DOCAone',
             'DOCAtwo',
             'DOCAthree',
             'IP_p0p2',
             'IP_p1p2',
             'isolationa',
             'isolationb',
             'isolationc',
             'isolationd',
             'isolatione',
             'isolationf',
             'iso',
             'CDF1',
             'CDF2',
             'CDF3',
             'ISO_SumBDT',
             'p0_IsoBDT',
             'p1_IsoBDT',
             'p2_IsoBDT',
             'p0_track_Chi2Dof',
             'p1_track_Chi2Dof',
             'p2_track_Chi2Dof',
             'p0_pt',
             'p0_p',
             'p0_eta',
             'p0_IP',
             'p0_IPSig',
             'p1_pt',
             'p1_p',
             'p1_eta',
             'p1_IP',
             'p1_IPSig',
             'p2_pt',
             'p2_p',
             'p2_eta',
             'p2_IP',
             'p2_IPSig']

classifier = linear_model.LogisticRegression()
classifier.fit(train[variables], train['signal'])

result = pd.DataFrame({'id': test.index})
result['prediction'] = classifier.predict_proba(test[variables])[:, 1]

result.to_csv('submission.csv', index=False, sep=',')
