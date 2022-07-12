# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
y_train = train["interest_level"]
x_train = train.drop(["interest_level", "building_id", "display_address", "manager_id", "street_address"], axis=1)
#print(x_train.head())
#print(test.head())
#print(train.describe())

x_train["num_photos"] = x_train["photos"].apply(len)
x_train["num_features"] = x_train["features"].apply(len)
x_train["num_description_words"] = x_train["description"].apply(lambda x: len(x.split(" ")))
x_train["created"] = pd.to_datetime(x_train["created"])
x_train["created_year"] = x_train["created"].dt.year
x_train["created_month"] = x_train["created"].dt.month
x_train["created_day"] = x_train["created"].dt.day
x_train=x_train.drop(["photos", "created", "description"], axis=1)

#train_feats = train.copy()
x_train["dishwasher"] = 0
x_train["doorman"] = 0
x_train["pets"] = 0
x_train["concierge"] = 0
x_train["ac"] = 0
x_train["parking"] = 0
x_train["balcony"] = 0
x_train["bike"] = 0
x_train["storage"] = 0
x_train["outdoor"] = 0
x_train["roof"] = 0
x_train["garden"] = 0
x_train["fitness"] = 0
x_train["pool"] = 0
x_train["backyard"] = 0
x_train["dining_room"] = 0
x_train["in_unit_laundry"] = 0
x_train["laundry"] = 0
x_train["elevator"] = 0
x_train["garage"] = 0
x_train["prewar"] = 0
x_train["no_fee"] = 0
x_train["reduced_fee"] = 0
x_train["internet"] = 0

tot_feats = ''
for ind, row in x_train.iterrows():
    feats = ''
    for feature in row['features']:
        feats = " ".join([feats, "_".join(feature.strip().split(" "))])
        feats = feats.lower()
        feats = feats.replace('-', '_')
        tot_feats = " ".join([tot_feats, "_".join(feature.strip().split(" "))])
        
    if "dishwasher" in feats:
        x_train.loc[ind, "dishwasher"] = 1
        
    if "doorman" in feats:
        x_train.loc[ind, "doorman"] = 1
        
    if "pets" in feats or "pet" in feats or "dog" in feats or "dogs" in feats or "cat" in feats or "cats" in feats and "no_pets" not in feats:
        x_train.loc[ind, "pets"] = 1
        
    if "concierge" in feats:
        x_train.loc[ind, "concierge"] = 1
        
    if "air_conditioning" in feats or "central" in feats:
        x_train.loc[ind, "ac"] = 1
        
    if "parking" in feats:
        x_train.loc[ind, "parking"] = 1
        
    if "balcony" in feats or "deck" in feats or "terrace" in feats or "patio" in feats:
        x_train.loc[ind, "balcony"] = 1
        
    if "bike" in feats:
        x_train.loc[ind, "bike"] = 1
        
    if "storage" in feats:
        x_train.loc[ind, "storage"] = 1
        
    if "outdoor" in feats or "courtyard" in feats:
        x_train.loc[ind, "outdoor"] = 1
        
    if "roof" in feats:
        x_train.loc[ind, "roof"] = 1
        
    if "garden" in feats:
        x_train.loc[ind, "garden"] = 1

    if "fitness" in feats or "gym" in feats:
        x_train.loc[ind, "fitness"] = 1
        
    if "pool" in feats:
        x_train.loc[ind, "pool"] = 1
        
    if "backyard" in feats:
        x_train.loc[ind, "backyard"] = 1
        
    if "laundry_room" in feats or "laundry_in_unit" in feats or "washer" in feats or "dryer" in feats:
        x_train.loc[ind, "in_unit_laundry"] = 1

    if "laundry" in feats or "washer" in feats or "dryer" in feats:
        x_train.loc[ind, "laundry"] = 1
        
    if "elevator" in feats:
        x_train.loc[ind, "elevator"] = 1

    if "garage" in feats:
        x_train.loc[ind, "garage"] = 1

    if "prewar" in feats or "pre_war" in feats:
        x_train.loc[ind, "prewar"] = 1
    
    if "no_fee" in feats:
        x_train.loc[ind, "no_fee"] = 1
        
    if "reduced_fee" in feats or "low_fee" in feats:
        x_train.loc[ind, "reduced_fee"] = 1
        
    if "internet" in feats or "wifi" in feats or "wi_fi" in feats:
        x_train.loc[ind, "internet"] = 1

#print(train_feats.loc[train_feats["doorman"] == 1])
#print(train_feats["doorman"].value_counts())

x_train=x_train.drop("features", axis=1)
"""
tot_feats = tot_feats.lower()
tot_feats = tot_feats.replace('-', '_')

feat_counts = collections.Counter(tot_feats.split())
for word, count in sorted(feat_counts.items()):
    if "doorman" in word:
        print('"%s" is repeated %d time%s.' % (word, count, "s" if count > 1 else ""))
"""
print(x_train.head())
"""
lr = linear_model.LogisticRegression()
lr.fit(x_train, y_train)
preds = lr.predict(x_train)
print(preds)
print(y_train.head())
"""
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.33)
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
Y_val_pred = clf.predict_proba(X_val)
#print(Y_val_pred)
#print(y_train.head())
print(log_loss(Y_val, Y_val_pred))

####
#
#Test Data
#
####

x_test = test.drop(["building_id", "display_address", "manager_id", "street_address"], axis=1)

x_test["num_photos"] = x_test["photos"].apply(len)
x_test["num_features"] = x_test["features"].apply(len)
x_test["num_description_words"] = x_test["description"].apply(lambda x: len(x.split(" ")))
x_test["created"] = pd.to_datetime(x_test["created"])
x_test["created_year"] = x_test["created"].dt.year
x_test["created_month"] = x_test["created"].dt.month
x_test["created_day"] = x_test["created"].dt.day
x_test = x_test.drop(["photos", "created", "description"], axis=1)

#train_feats = train.copy()
x_test["dishwasher"] = 0
x_test["doorman"] = 0
x_test["pets"] = 0
x_test["concierge"] = 0
x_test["ac"] = 0
x_test["parking"] = 0
x_test["balcony"] = 0
x_test["bike"] = 0
x_test["storage"] = 0
x_test["outdoor"] = 0
x_test["roof"] = 0
x_test["garden"] = 0
x_test["fitness"] = 0
x_test["pool"] = 0
x_test["backyard"] = 0
x_test["dining_room"] = 0
x_test["in_unit_laundry"] = 0
x_test["laundry"] = 0
x_test["elevator"] = 0
x_test["garage"] = 0
x_test["prewar"] = 0
x_test["no_fee"] = 0
x_test["reduced_fee"] = 0
x_test["internet"] = 0

tot_feats = ''
for ind, row in x_test.iterrows():
    feats = ''
    for feature in row['features']:
        feats = " ".join([feats, "_".join(feature.strip().split(" "))])
        feats = feats.lower()
        feats = feats.replace('-', '_')
        tot_feats = " ".join([tot_feats, "_".join(feature.strip().split(" "))])
        
    if "dishwasher" in feats:
        x_test.loc[ind, "dishwasher"] = 1
        
    if "doorman" in feats:
        x_test.loc[ind, "doorman"] = 1
        
    if "pets" in feats or "pet" in feats or "dog" in feats or "dogs" in feats or "cat" in feats or "cats" in feats and "no_pets" not in feats:
        x_test.loc[ind, "pets"] = 1
        
    if "concierge" in feats:
        x_test.loc[ind, "concierge"] = 1
        
    if "air_conditioning" in feats or "central" in feats:
        x_test.loc[ind, "ac"] = 1
        
    if "parking" in feats:
        x_test.loc[ind, "parking"] = 1
        
    if "balcony" in feats or "deck" in feats or "terrace" in feats or "patio" in feats:
        x_test.loc[ind, "balcony"] = 1
        
    if "bike" in feats:
        x_test.loc[ind, "bike"] = 1
        
    if "storage" in feats:
        x_test.loc[ind, "storage"] = 1
        
    if "outdoor" in feats or "courtyard" in feats:
        x_test.loc[ind, "outdoor"] = 1
        
    if "roof" in feats:
        x_test.loc[ind, "roof"] = 1
        
    if "garden" in feats:
        x_test.loc[ind, "garden"] = 1

    if "fitness" in feats or "gym" in feats:
        x_test.loc[ind, "fitness"] = 1
        
    if "pool" in feats:
        x_test.loc[ind, "pool"] = 1
        
    if "backyard" in feats:
        x_test.loc[ind, "backyard"] = 1
        
    if "laundry_room" in feats or "laundry_in_unit" in feats or "washer" in feats or "dryer" in feats:
        x_test.loc[ind, "in_unit_laundry"] = 1

    if "laundry" in feats or "washer" in feats or "dryer" in feats:
        x_test.loc[ind, "laundry"] = 1
        
    if "elevator" in feats:
        x_test.loc[ind, "elevator"] = 1

    if "garage" in feats:
        x_test.loc[ind, "garage"] = 1

    if "prewar" in feats or "pre_war" in feats:
        x_test.loc[ind, "prewar"] = 1
    
    if "no_fee" in feats:
        x_test.loc[ind, "no_fee"] = 1
        
    if "reduced_fee" in feats or "low_fee" in feats:
        x_test.loc[ind, "reduced_fee"] = 1
        
    if "internet" in feats or "wifi" in feats or "wi_fi" in feats:
        x_test.loc[ind, "internet"] = 1

x_test = x_test.drop("features", axis=1)

y_test = clf.predict_proba(x_test)

labels2idx = {label: i for i, label in enumerate(clf.classes_)}
print(labels2idx)

out = pd.DataFrame()
out["listing_id"] = test["listing_id"]
for label in ["high", "medium", "low"]:
    out[label] = y_test[:, labels2idx[label]]
out.to_csv("submission.csv", index=False)
