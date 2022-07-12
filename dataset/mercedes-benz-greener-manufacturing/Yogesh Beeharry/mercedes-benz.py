# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

n = train_df.shape[1] - 2 # count of X columns in input data

X = pd.concat([train_df[train_df.columns[-n:]], test_df[test_df.columns[-n:]] ], ignore_index=True)
X = pd.get_dummies(X)

train_X = X.head(train_df.shape[0]).as_matrix()
test_X = X.tail(test_df.shape[0]).as_matrix()
train_Y = train_df["y"].as_matrix()
test_id = test_df["ID"].as_matrix()


pca = PCA(n_components=100)
pca.fit(train_X)

lr = Ridge().fit(pca.transform(train_X), train_Y)
print("Training set score: {:.2f}".format(lr.score(pca.transform(train_X), train_Y)))

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

predict_Y = lr.predict(pca.transform(test_X))


output = pd.DataFrame({'y': predict_Y})
output['ID'] = test_id
output = output.set_index('ID')
output.to_csv('submission.csv')

