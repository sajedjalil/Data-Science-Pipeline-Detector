import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

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
  
  #RandomForestClassifier(n_estimators=20,n_jobs=-1,random_state=0)
  #PCA(n_components=15)
  #GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
  pipe = RandomForestClassifier(n_estimators=20,n_jobs=-1,random_state=0)
  #make_pipeline(RandomForestClassifier(n_estimators=20,n_jobs=-1,random_state=0))
  pipe
  
  pipe.fit(X_train, y)
  
  names = X_train.columns
  x = sorted(zip(map(lambda x: round(x, 4), pipe.feature_importances_), names), reverse=True)
  print(x)
  
  
  with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(pipe.predict(X_test))):
      outfile.write("%s,%s\n"%(test_ids[e],val))