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
from sklearn.metrics import roc_auc_score

data_path  = "../input/"

people = pd.read_csv(data_path + "people.csv",
                     index_col=0, usecols=["people_id", "char_38"])
train = pd.read_csv(data_path + "act_train.csv",
                    index_col=0, usecols=["people_id", "outcome"])
test = pd.read_csv(data_path + "act_test.csv",
                   index_col=0, usecols=["people_id", "activity_id"])

train = train.join(people)
print("AUC in train:", roc_auc_score(train.outcome, train.char_38))

test.join(people).set_index("activity_id").\
    rename(columns={"char_38": "outcome"}).\
    to_csv("BB_char_38.csv", header=True)