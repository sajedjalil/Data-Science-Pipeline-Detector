# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

#print(check_output(["head", "../input/data.csv"]).decode("utf8"))
### Read in data
data = pd.read_csv("../input/data.csv")

### Remove features
data = data.drop("team_id", 1)
data = data.drop("team_name", 1)

data.columns
data[["loc_x", "loc_y", "lat", "lon"]]
data["playoffs"].value_counts()
### Feature engineering
shottype = pd.get_dummies(data["combined_shot_type"], prefix = "Shottype")
period = pd.get_dummies(data["period"], prefix = "Period")
season = pd.get_dummies(data["season"], prefix = "Season")
opponent = pd.get_dummies(data["opponent"], prefix = "Opponent")

# data["playoffs"], data["shot_distance"]
mydata = pd.concat([data['shot_made_flag'], data['shot_id'], shottype, period, season, data["playoffs"], data["shot_distance"], data[["loc_x", "loc_y", "lat", "lon", "playoffs"]], opponent], axis = 1)

### Split mydata to train and test set 
train = mydata[mydata["shot_made_flag"].notnull()]
test = mydata[mydata["shot_made_flag"].isnull()]

target = train["shot_made_flag"]
train = train.drop(["shot_made_flag", "shot_id"], 1)
test_shot_id = test["shot_id"]
test = test.drop(["shot_made_flag", "shot_id"], 1)
### Logistic Regression
lr = LogisticRegression().fit(train, target)
result = lr.predict_proba(test)

prob = result[: ,1]

print(max(prob), min(prob))
output = pd.concat([test_shot_id.reset_index(drop = True), pd.Series(data = prob, name = 'shot_made_flag')], axis = 1)
#pd.Series(data = prob, name = "prob")
output.to_csv("my_third_submission.csv", index = False)
print(check_output(["cp", "../input/sample_submission.csv", "sample_submission.csv"]).decode("utf8"))