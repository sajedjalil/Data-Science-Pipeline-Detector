# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

# Any results you write to the current directory are saved as output.

# loading data
events = pd.read_csv("../input/events.csv", dtype = {"device_id": np.str}, infer_datetime_format = True, parse_dates = ["timestamp"])
app_events = pd.read_csv("../input/app_events.csv", usecols = [0, 2, 3],
                            dtype = {"is_active": np.float16, "is_installed": np.float16})

# get hour and drop timestamp
events["hour"] = events["timestamp"].apply(lambda x: x.hour).astype(np.int8)
events.drop("timestamp", axis = 1, inplace = True)

# merge data w/o train or test
events = events.merge(app_events, how = "left", on = "event_id")
del app_events
events.drop("event_id", axis = 1, inplace = True)

# prep brands
phone = pd.read_csv("../input/phone_brand_device_model.csv", dtype={"device_id": np.str})

print (events.info())
print (phone.info())

# feature hasher
feat = FeatureHasher(n_features=12, input_type="string", dtype=np.float32)
print(feat)

feat1 = feat.transform(phone["phone_brand"] + " " + phone["device_model"])

print(feat1) 

events = events.merge(pd.concat([phone["device_id"], pd.DataFrame(phone["phone_brand"] + " " + phone["device_model"])], axis = 1), how = "left", on = "device_id")

print(events.head(5))

del phone, feat, feat1

print("pre-merging and hashing finished.")

# train steps
train = pd.read_csv("../input/gender_age_train.csv", dtype = {"device_id": np.str},\
                    usecols = [0, 3])
t2 = train.copy()
train.drop("group", axis = 1, inplace = True)
train = train.merge(events, how = "left", on = "device_id")
train.fillna(-1, inplace = True)
train = train.groupby("device_id").mean().reset_index()
train = train.merge(t2, how ="left", on = "device_id")

label = train["group"].copy()
train.drop(["group", "device_id"], axis = 1, inplace = True)
del t2

print("train data merged and prepared")
print("-----------------------------------")
print(train.info())
print(train.head(5))
print("-----------------------------------")


# splitting data for memory and validation
#train, valid, train_lab, valid_lab = train_test_split(train, label, train_size = 250000, test_size = 100,\
#                                                        random_state = 123, stratify = label)
#del valid, valid_lab
#print("train data splitted")

# building Adaboost model
dt = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=123)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=700, learning_rate=0.05, random_state=123)
#params = {"n_estimators": [20, 30, 100], "base_estimator__max_depth": [3, 4, None], "learning_rate": [0.01, 0.02]}
params = {"n_estimators": [800], "base_estimator__max_depth": [None], "learning_rate": [0.04]}
grid = GridSearchCV(ada, params, scoring = "log_loss", cv = 3, n_jobs = -1)

grid.fit(train, label)
del train, label

print("train finished")
#print("validation accuracy: " + str(ada.score(valid, valid_lab)))
#print("validation log loss: " + str(log_loss(valid_lab, ada.predict_proba(valid))))
#del valid, valid_lab

print(pd.DataFrame(grid.grid_scores_))

# load test data merge with events
test = pd.read_csv("../input/gender_age_test.csv", dtype = {"device_id": np.str})

test = test.merge(events, how = "left", on = "device_id")
del events
print("test loaded and merged")


# prep test data for prediction
#ids = test["device_id"].copy()
#test.drop("device_id", axis = 1, inplace = True)
test.fillna(-1, inplace = True)
test["hour"] = test["hour"].astype(np.float16)
test = test.groupby("device_id").mean().reset_index()
ids = test["device_id"].copy()
test.drop("device_id", axis = 1, inplace = True)

print("test prepared")
print("-----------------------------------")
print(test.info())
print("-----------------------------------")

pred = grid.predict_proba(test)

#ind = np.array_split(np.arange(test.shape[0]), 20)
#print("ind shape: " + str(len(ind)))
#pred = np.empty((test.shape[0], len(ada.classes_)), dtype = np.float32)
#print("pred empty shape: " + str(pred.shape))

#print("starting prediction")
#run = 1
#for i in ind:
#    pred[i, :] = ada.predict_proba(test.iloc[i, :]).astype(np.float32)
#    print("prediction round: " + str(run) + " of " + str(len(ind)))
#    run += 1

del test #, ind
print("prediction finished")

pred = pd.concat([ids, pd.DataFrame(pred, columns = grid.best_estimator_.classes_)], axis = 1)
del ada, dt, grid, ids
#pred.fillna(class_prob, inplace = True)
#del class_prob
print("concat finished")
print("-----------------------------------")

sub = pred.copy() #.groupby("device_id").mean().reset_index()
del pred
print("shape of submission: " + str(sub.shape))
sub.to_csv("submission.csv", index = False)
del sub
print("submission saving finished.")
