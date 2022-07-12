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
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC

x_train = pd.read_csv("../input/train.csv")
x_test = pd.read_csv("../input/test.csv").values

target = x_train.iloc[:, -1].values
X = x_train.iloc[:, 1:-1].values

pca = PCA(n_components=0.8)
X_new = pca.fit_transform(X)
X_positive = X_new[target == 1, :]
X_negative = X_new[target == 0, :]
pos_len = len(X_positive)
neg_len = len(X_negative)

part_len = neg_len / 24

clfs = [0] * 24
for i in range(24):
    X_part = X_negative[i * part_len : (i + 1) * part_len - 1]
    clfs[i] = SVC()
    clfs[i].fit(np.concatenate((X_part, X_positive)), np.concatenate((np.zeros((len(X_part), 1)), np.ones((pos_len, 1)))).ravel())

result = np.array([clfs[i].predict(pca.transform(x_test[:, 1:])) for i in range(24)])
result = np.sum(result, axis=0) / 24.

result = pd.DataFrame({"ID": x_test[:, 0].astype('int'), "TARGET": result})
result.to_csv('submission.csv', index=False)