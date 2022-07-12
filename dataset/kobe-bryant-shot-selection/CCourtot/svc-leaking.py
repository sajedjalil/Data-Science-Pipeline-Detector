# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import print_function
from sklearn import datasets
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn.svm import SVC
#from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

from sklearn.preprocessing import LabelEncoder

# Any results you write to the current directory are saved as output.
enc = LabelEncoder()
data = pd.read_csv("../input/data.csv")
data["action_type"] = enc.fit_transform(data["action_type"])
Combined_shot_type = set(data.combined_shot_type)
data["combined_shot_type"] = enc.fit_transform(data["combined_shot_type"])
data["shot_type"] = enc.fit_transform(data["shot_type"])
data["shot_zone_area"] = enc.fit_transform(data["shot_zone_area"])
data["shot_zone_basic"] = enc.fit_transform(data["shot_zone_basic"])
data["shot_zone_range"] = enc.fit_transform(data["shot_zone_range"])
data["opponent"] = enc.fit_transform(data["opponent"])
data["game_date"]=enc.fit_transform(data["game_date"])
data["season"]=enc.fit_transform(data["season"])
data = data.drop("team_name",1)
data = data.drop("team_id",1)
data = data.drop("matchup",1)
#data = data.drop("shot_type",1)
#data = data.drop("playoffs",1)
data = data.drop("combined_shot_type",1)
data = data.drop("shot_zone_area",1)
data = data.drop("shot_zone_basic",1)
data = data.drop("shot_zone_range",1)
data = data.drop("loc_x",1)
data = data.drop("loc_y",1)

train = data.dropna(axis=0)
test = data[data.shot_made_flag.isnull()]
test = test.drop("shot_made_flag",1)
testID = test.shot_id
trainSMF = train.shot_made_flag
train = train.drop("shot_made_flag",1)

#forest = ExtraTreesClassifier(n_estimators=500)
#forest = SVC()
forest = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.1,subsample=0.5)

forest.fit(train,trainSMF)
dt = forest.predict(test)
prediction = pd.DataFrame()
prediction["shot_id"] = testID
prediction["shot_made_flag"]= dt
prediction.to_csv("ETC.csv",index=False)
