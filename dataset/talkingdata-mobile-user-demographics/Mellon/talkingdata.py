# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher

events = pd.read_csv("../input/events.csv", dtype = {"device_id": np.str}, infer_datetime_format = True, parse_dates = ["timestamp"])
app_events = pd.read_csv("../input/app_events.csv", usecols = [0, 2, 3],
                            dtype = {"is_active": np.float16, "is_installed": np.float16})

print(events.head(10))
print(events.columns)

print(events.info())

print(app_events.info())

# get hour and drop timestamp
events["hour"] = events["timestamp"].apply(lambda x: x.hour).astype(np.int8)
events.drop("timestamp", axis = 1, inplace = True)

# merge data w/o train or test
events = events.merge(app_events, how = "left", on = "event_id")
#del app_events
events.drop("event_id", axis = 1, inplace = True)

# prep brands
phone = pd.read_csv("../input/phone_brand_device_model.csv", dtype={"device_id": np.str},usecols = [0, 1, 2])

events = events.merge(pd.concat([phone["device_id"], phone["phone_brand"],phone["device_model"]], axis = 1), how = "left", on = "device_id")
#del phone, feat, feat1

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


print(train.head(5))

print(train.info())
