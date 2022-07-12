import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier

if __name__ == "__main__":
  loc_train = "../input/train.csv"
  loc_test = "../input/test.csv"
  loc_submission = "sriteja.submission.csv"

  df_train = pd.read_csv(loc_train)
  df_test = pd.read_csv(loc_test)

  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]
  y = df_train['Cover_Type']
  test_ids = df_test['Id']
  del df_train
  del df_test
  
  clf = ExtraTreesClassifier(n_estimators=200,max_depth=None, min_samples_split=2)
  clf.fit(X_train, y)
  del X_train
  
  with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))