import pandas as pd
from sklearn import svm
from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier as xgb

sample = [[7, 7, 5, 1],
          [7, 3, 7, 1],
          [0, 2, 3, 0],
          [0, 6, 2, 0],
          [0, 0, 0, 0],
          [7, 0, 7, 1],
          [9, 6, 2, 1],
          [6, 5, 3, 1],
          [1, 4, 5, 0],
          [7, 6, 2, 1],
          [1, 6, 6, 0],
          [1, 7, 4, 0],
          [6, 2, 1, 0],
          [3, 6, 7, 1],
          [6, 5, 1, 0],
          [7, 3, 2, 0],
          [5, 0, 2, 0],
          [7, 7, 5, 1],
          [2, 3, 0, 0],
          [7, 5, 4, 1],
          [9, 6, 1, 1],
          [8, 0, 1, 0],
          [0, 1, 7, 0],
          [3, 4, 2, 0],
          [3, 4, 6, 0],
          [6, 8, 9, 1]]

data = pd.DataFrame(sample, columns=['x1', 'x2', 'x3', 'y'])
cols = [col for col in data.columns.values if 'x' in col]
X = data[cols].values
y = data.y.tolist()
num_of_data = [13, 20]
for n in num_of_data:
    train_data = X[:n]
    test_data = X[n:]
    t_y = y[n:]
    clf = xgb(objective='binary:logistic')
    clf.fit(train_data, y[:n])
    results = clf.predict(test_data)
    print("matthews corrcoef : " + str(matthews_corrcoef(t_y, results)))

