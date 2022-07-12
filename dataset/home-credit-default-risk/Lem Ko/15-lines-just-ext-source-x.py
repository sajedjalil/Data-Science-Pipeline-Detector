import pandas as pd 
from lightgbm import LGBMClassifier

data = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")

clf = LGBMClassifier()
clf.fit(data.filter(regex=r'^EXT_SOURCE_.', axis=1), data['TARGET'])

probabilities = clf.predict_proba(test.filter(regex=r'^EXT_SOURCE_.', axis=1))
submission = pd.DataFrame({
    'SK_ID_CURR': test['SK_ID_CURR'],
    'TARGET':     [ row[1] for row in probabilities]
})
submission.to_csv("submission.csv", index=False)