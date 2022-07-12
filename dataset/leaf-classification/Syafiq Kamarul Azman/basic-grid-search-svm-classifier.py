import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

le = LabelEncoder()
le.fit(train.species)

scaler = StandardScaler()

X_train = train.drop(["id", "species"], axis=1).as_matrix()
y_train = le.transform(train.species)
X_test = test.drop(["id"], axis=1).as_matrix()

scaler.fit(X_train)
X_train = scaler.transform(X_train)

print("Training model...")
params = {"kernel": ("rbf", "linear"), "C": [0.1, 0.3, 1, 3, 10, 30, 100]}
svm = SVC()
clf = GridSearchCV(svm, params, cv=5)
clf.fit(X_train, y_train)

print("Best parameters: " + str(clf.best_params_))
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
    print(scores)

print("Predicting test set...")
results = clf.predict(scaler.transform(X_test))
r = np.zeros([len(X_test), len(le.classes_)])

for i, v in enumerate(results):
    r[i, v] = 1

submit = pd.DataFrame(r, index=test.id, columns=le.classes_)
submit.to_csv('submit.csv')
