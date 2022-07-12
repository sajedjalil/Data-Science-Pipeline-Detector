import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
data = train.drop('Id', axis=1)

y = data.Cover_Type
X = data.drop('Cover_Type', axis=1)
feature = [col for col in train.columns if col not in ['Cover_Type', 'Id']]
X_test = test[feature]

clf = RandomForestClassifier(n_estimators=1000, random_state=33)
clf.fit(X, y)

sub = pd.DataFrame({"Id": test['Id'], "Cover_Type": clf.predict(X_test)})
sub.to_csv("etc2.csv", index=False)