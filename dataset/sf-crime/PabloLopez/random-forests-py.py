# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

train['Dates']=pd.to_datetime(train['Dates'])
train['month'] = train['Dates'].dt.month
train['hour'] = train['Dates'].dt.hour
#print(train['hour'])
y, X = dmatrices('Category ~ C(month) + C(hour) + C(DayOfWeek) + C(PdDistrict) + X + Y', train, return_type="dataframe")
model = RandomForestClassifier(n_jobs=2)
#model = LogisticRegression()
model = model.fit(X, y)
#print(model.score(X,y))