import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

y_train = train["type"].map({"Ghoul": 1, "Ghost": 2, "Goblin": 3})


train.drop("type", inplace=True, axis=1)

train_test = pd.concat([train, test], axis=0)

# drop 'color'
train_test = train_test.drop('color', 1)

X_train = train_test.iloc[:len(y_train)].as_matrix()
X_test  = train_test.iloc[len(y_train):].as_matrix()
y_train = y_train.transpose().as_matrix()

# OK now for the guts of the classifier

kernel = 1.0 * RBF([1.0, 1.0, 1.0, 1.0])
gpc = GaussianProcessClassifier(kernel)
gpc.fit(X_train, y_train)
y_pred = gpc.predict(X_test)

# Now write out the predictions to a file

monster_map = {1:"Ghoul", 2:"Ghost", 3:"Goblin"}

with open('submission-GGGaussian.csv', 'w') as f:
    f.write("id,type\n")
    y_test_it = 0
    for row in test.iterrows():
        f.write("{},{}\n".format(row[0], monster_map[y_pred[y_test_it]]))
        y_test_it += 1