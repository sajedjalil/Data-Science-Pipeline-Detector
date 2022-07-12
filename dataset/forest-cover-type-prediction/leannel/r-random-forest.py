import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import math

def two_largest_indices(inlist):
  largest = 0
  second_largest = 0
  largest_index = 0
  second_largest_index = -1
  for i in range(len(inlist)):
    item = inlist[i]
    if item > largest:
      second_largest = largest
      second_largest_index = largest_index
      largest = item
      largest_index = i
    elif largest > item >= second_largest:
      second_largest = item
      second_largest_index = i
    # Return the results as a tuple
  return largest_index, second_largest_index
    
if __name__ == "__main__":
  loc_train = "../input/train.csv"
  loc_test = "../input/test.csv"
  loc_submission = "kaggle.rf200.entropy.submission.csv"

  df_train = pd.read_csv(loc_train)
  df_test = pd.read_csv(loc_test)
  
  cols_to_normalize = ['Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
  'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']

  df_train[cols_to_normalize] = normalize(df_train[cols_to_normalize])
  df_test[cols_to_normalize] = normalize(df_test[cols_to_normalize])


  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]
  feature_cols.append('binned_elevation')
  feature_cols.append('Horizontal_Distance_To_Roadways_Log')
  feature_cols.append('Soil_Type12_32')
  feature_cols.append('Soil_Type23_22_32_33')
  
  
  df_train['binned_elevation'] = [math.floor(v/50.0) for v in df_train['Elevation']]
  df_test['binned_elevation'] = [math.floor(v/50.0) for v in df_test['Elevation']]
  
  df_train['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df_train['Horizontal_Distance_To_Roadways']]
  df_test['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df_test['Horizontal_Distance_To_Roadways']]

  df_train['Soil_Type12_32'] = df_train['Soil_Type32'] + df_train['Soil_Type12']
  df_test['Soil_Type12_32'] = df_test['Soil_Type32'] + df_test['Soil_Type12']
  df_train['Soil_Type23_22_32_33'] = df_train['Soil_Type23'] + df_train['Soil_Type22'] + df_train['Soil_Type32'] + df_train['Soil_Type33']
  df_test['Soil_Type23_22_32_33'] = df_test['Soil_Type23'] + df_test['Soil_Type22'] + df_test['Soil_Type32'] + df_test['Soil_Type33']
  
  df_train_1_2 = df_train[(df_train['Cover_Type'] <= 2)]
  df_train_3_4_6 = df_train[(df_train['Cover_Type'].isin([3,4,6]))]
  
  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]
  
  X_train_1_2 = df_train_1_2[feature_cols]
  X_train_3_4_6 = df_train_3_4_6[feature_cols]
  
  y = df_train['Cover_Type']
  y_1_2 = df_train_1_2['Cover_Type']
  y_3_4_6 = df_train_3_4_6['Cover_Type']
  
  test_ids = df_test['Id']
  del df_train
  del df_test
  
  clf = ensemble.ExtraTreesClassifier(n_estimators=100,n_jobs=-1,random_state=0)
  clf.fit(X_train, y)
  
  clf_1_2 = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
  clf_1_2.fit(X_train_1_2, y_1_2)
  
  clf_3_4_6 = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
  clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)
  
  del X_train
  
  vals_1_2 = {}
  for e, val in enumerate(list(clf_1_2.predict_proba(X_test))):
    vals_1_2[e] = val
  print(clf_1_2.classes_) 
  
  vals_3_4_6 = {}
  for e, val in enumerate(list(clf_3_4_6.predict_proba(X_test))):
    vals_3_4_6[e] = val 
  print(clf_3_4_6.classes_)
  
  vals = {}
  for e, val in enumerate(list(clf.predict(X_test))):
    vals[e] = val 
  
    
  with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict_proba(X_test))):
      val[0] += vals_1_2[e][0]/1.3
      val[1] += vals_1_2[e][1]/1.1
      val[2] += vals_3_4_6[e][0]/3.4
      val[3] += vals_3_4_6[e][1]/4.0
      val[5] += vals_3_4_6[e][2]/3.6
      i,j = two_largest_indices(val)
      v = i  + 1
      outfile.write("%s,%s\n"%(test_ids[e],v))