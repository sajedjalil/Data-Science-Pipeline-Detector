# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import neural_network as nn
from sklearn.externals import joblib

import datetime



file_train_path = "../input/train_V2.csv"
file_test_path = "../input/test_V2.csv"

import re

file_train_path_new = "./train_new.csv"
file_test_path_new = "./test_new.csv"

match_categories = ((",solo,", ",1,"),
                    (",duo,", ",2,"),
                    (",squad,", ",3,"),
                    (",solo-fpp,", ",4,"),
                    (",duo-fpp,", ",5,"),
                    (",squad-fpp,", ",6,"),
                    (",normal-solo,", ",7,"),
                    (",normal-duo,", ",8,"),
                    (",normal-squad,", ",9,"),
                    (",normal-solo-fpp,", ",10,"),
                    (",normal-duo-fpp,", ",11,"),
                    (",normal-squad-fpp,", ",12,"),
                    (",crashfpp,", ",13,"),
                    (",flaretpp,", ",14,"),
                    (",flarefpp,", ",15,"),
                    (",crashtpp,", ",16,"))

print("starting train")
with open(file_train_path) as f_in:
    with open(file_train_path_new, "w") as f_out:
        for line in f_in:
            for pair in match_categories:
                line = line.replace(*pair)
            f_out.write(line)

f_in.close()
f_out.close()
print("done train")

print("starting test")
with open(file_test_path) as f_in:
    with open(file_test_path_new, "w") as f_out:
        for line in f_in:
            for pair in match_categories:
                line = line.replace(*pair)
            f_out.write(line)

f_in.close()
f_out.close()
print("done test")



file_train_path = "./train_new.csv"
file_test_path = "./test_new.csv"

linearSVR_model_path = "./models/linearSVR.sav"
MLPRegressor_model_path = "./models/MLPRegressor.sav"

time_start = datetime.datetime.now()
print("Reading training data...")
time_start_read_train = datetime.datetime.now()
train_original = pd.read_csv(file_train_path)
time_end_read_train = datetime.datetime.now()
print(f"Done reading training data. This took {time_end_read_train - time_start_read_train} seconds")

print("Reading testing data...")
time_start_read_test = datetime.datetime.now()
test_original = pd.read_csv(file_test_path)
time_end_read_test = datetime.datetime.now()
print(f"Done reading training data. This took {time_end_read_test - time_start_read_test} seconds")


# For the kids
train_original['winPlacePerc'].fillna((train_original['winPlacePerc'].mean()), inplace=True)
corr = train_original.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, fmt='.1f',
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# f, ax = plt.subplots(figsize=(15, 15))
# sns.heatmap(train_original.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
# plt.show()


droppedFeatures = ["Id", "groupId", "matchId", "killPoints", "maxPlace", "roadKills", "swimDistance", "teamKills",
                   "vehicleDestroys", "winPoints", "numGroups"]

# Setting up the training dataset

# Remove unwanted features from the training data
print("Cleaning train")
train_cleanedFeatures = train_original.drop(droppedFeatures, axis=1).copy()
# Separate the features from the label
print("getting label_train")
label_train = train_cleanedFeatures["winPlacePerc"].to_frame().fillna(0.0001)
print("getting features_train")
features_train = train_cleanedFeatures.drop("winPlacePerc", axis=1)

# Setting up the testing dataset

# Remove unwanted features from the testing data
print("Cleaning test")
test_cleanedFeatures = test_original.drop(droppedFeatures, axis=1).copy()
# Separate the features from the label
print("getting features_test")
features_test = test_cleanedFeatures


# See if linearSVR model already exists
if os.path.isfile(linearSVR_model_path):
    clf_linearSVR = joblib.load(linearSVR_model_path)
    print(f"Loaded classifier from {linearSVR_model_path}")
else:
    # Define the classifier
    clf_linearSVR = svm.LinearSVR(verbose=1)
    print(clf_linearSVR)

    # Build it yo!
    print("Starting fit...")
    time_start_fit = datetime.datetime.now()
    clf_linearSVR.fit(features_train, label_train)
    time_end_fit = datetime.datetime.now()
    print(f"Finished fitting. This took {time_end_fit - time_start_fit} seconds")
    print(f"Dumping classifier into {linearSVR_model_path}")
    #joblib.dump(clf_linearSVR, linearSVR_model_path)

# Get score
train_score = clf_linearSVR.score(features_train, label_train)
print(f"Classifier score on training data: {train_score}")

# Predict test
print("Starting test...")
time_start_test = datetime.datetime.now()
label_test_predicted = clf_linearSVR.predict(features_test)
time_end_test = datetime.datetime.now()
print(f"Done testing. This took {time_end_test - time_start_test} seconds")


# See if MLPRegressor model already exists
if os.path.isfile(MLPRegressor_model_path):
    clf_MLPRegressor = joblib.load(MLPRegressor_model_path)
    print(f"Loaded classifier from {MLPRegressor_model_path}")
else:
    # Define the classifier
    clf_MLPRegressor = nn.MLPRegressor(hidden_layer_sizes=(100, 100), activation="relu", solver="adam", verbose=True)
    print(clf_MLPRegressor)

    # Build it yo!
    print("Starting fit...")
    time_start_fit = datetime.datetime.now()
    clf_MLPRegressor.fit(features_train, label_train)
    time_end_fit = datetime.datetime.now()
    print(f"Finished fitting. This took {time_end_fit - time_start_fit} seconds")
    print(f"Dumping classifier into {MLPRegressor_model_path}")
    #joblib.dump(clf_MLPRegressor, MLPRegressor_model_path)


# Get score linearSVR
linearSVR_train_score = clf_linearSVR.score(features_train, label_train)
print(f"Classifier score on training data: {train_score}")

# Predict test
print("Starting test...")
time_start_test = datetime.datetime.now()
linearSVR_label_test_predicted = clf_linearSVR.predict(features_test)
time_end_test = datetime.datetime.now()
print(f"Done testing. This took {time_end_test - time_start_test} seconds")
linearSVR_output = pd.concat([test_original.Id, pd.Series(linearSVR_label_test_predicted)], axis=1)
linearSVR_output = linearSVR_output.rename(columns={"Id": "Id", 0: "winPlacePerc"})
linearSVR_output.to_csv("./linearSVR_submission.csv", index=False)


# Get score for MLPRegressor
MLPRegressor_train_score = clf_MLPRegressor.score(features_train, label_train)
print(f"Classifier score on training data: {train_score}")

# Predict test
print("Starting test...")
time_start_test = datetime.datetime.now()
MLPRegressor_label_test_predicted = clf_MLPRegressor.predict(features_test)
time_end_test = datetime.datetime.now()
print(f"Done testing. This took {time_end_test - time_start_test} seconds")
MLPRegressor_output = pd.concat([test_original.Id, pd.Series(MLPRegressor_label_test_predicted)], axis=1)
MLPRegressor_output = MLPRegressor_output.rename(columns={"Id": "Id", 0: "winPlacePerc"})
MLPRegressor_output.to_csv("./MLPRegressor_submission.csv", index=False)
