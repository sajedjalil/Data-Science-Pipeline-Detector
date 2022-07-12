import pandas as pd
from sklearn import ensemble

if __name__ == "__main__":
  loc_train = "../input/train.csv"
  loc_test = "../input/test.csv"
  loc_submission = "kaggle.rf200.entropy.submission.csv"

  df_train = pd.read_csv(loc_train)
  df_test = pd.read_csv(loc_test)

  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]
  y = df_train['Cover_Type']
  test_ids = df_test['Id']
  del df_train
  del df_test
  
  clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
  clf.fit(X_train, y)
  del X_train
  
  with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))