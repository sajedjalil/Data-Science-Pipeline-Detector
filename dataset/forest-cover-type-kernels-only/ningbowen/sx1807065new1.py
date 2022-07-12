# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestClassifier 
import numpy as np  



data_train = pd.read_csv("../input/train.csv")
X_train,y_train = data_train.iloc[:,1:55].values,data_train['Cover_Type']
data_test = pd.read_csv("../input/test.csv")
X_test = data_test.iloc[:,1:55].values

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2
        )
forest.fit(X_train,y_train)

pred = forest.predict(X_test)


X_id, y = data_test.iloc[:,0],pred
dataFrame = pd.DataFrame({"Id": X_id, "Cover_Type": pred}, columns=['Id', 'Cover_Type'])
dataFrame.to_csv("submission.csv", index=False)
# Any results you write to the current directory are saved as output.