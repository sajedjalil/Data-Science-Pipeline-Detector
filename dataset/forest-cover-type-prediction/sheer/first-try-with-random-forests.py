import pandas as pd
import math
import numpy as np
from sklearn import ensemble

def w(v):
  bin = math.floor(v/100)
  if bin == 18:
    return int(0.15086917960088692 ** 3)
  elif bin == 19:
    return int(0.1730135922330097 ** 3)
  elif bin == 20:
    return int(0.17692946058091286 ** 3)
  elif bin == 21:
    return int(0.18436363636363637 ** 3)
  elif bin == 22:
    return int(0.21559463487332337 ** 3)
  elif bin == 23:
    return int(0.2555172413793103 ** 3)
  elif bin == 24:
    return int(0.2628888888888889 ** 3)
  elif bin == 25:
    return int(0.3087597989949749 ** 3)
  elif bin == 26:
    return int(0.3528139534883721 ** 3)
  elif bin == 27:
    return int(0.4176566371681416 ** 3)
  elif bin == 28:
    return int(0.49127906976744184 ** 3)
  elif bin == 29:
    return int(0.5601818181818182 ** 3)
  elif bin == 30:
    return int(0.6325436681222707 ** 3)
  elif bin == 31:
    return int(0.6950592274678111 ** 3)
  elif bin == 32:
    return int(0.8885893648449039 ** 3)
  elif bin == 33:
    return int(0.9322424242424241 ** 3)
  elif bin == 34:
    return int(1.1813639788997738 ** 3)
  elif bin == 35:
    return int(1.7053053016453381 ** 3)
  elif bin == 36:
    return int(2.4655654822335022 ** 3)
  elif bin == 37:
    return int(2.6190562874251495 ** 3)
  elif bin == 38:
    return int(2.8439430693069307 ** 3)
  else:
    print('oooooops')
    return 1
  
if __name__ == "__main__":
  loc_train = "../input/train.csv"
  loc_test = "../input/test.csv"
  loc_submission = "kaggle.rf200.entropy.submission.csv"

  df_train = pd.read_csv(loc_train)
  df_test = pd.read_csv(loc_test)

  
  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

  weights= [100-w(v) for v in df_train['Elevation']]
  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]
  
  y = df_train['Cover_Type']
  test_ids = df_test['Id']
  
  del df_train
  del df_test
  
  clf = ensemble.RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=0)
  clf.fit(X_train, y, weights)
  del X_train
  
  with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))
