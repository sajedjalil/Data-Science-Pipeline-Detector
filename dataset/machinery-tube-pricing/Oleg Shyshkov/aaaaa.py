import numpy as np
import pandas as pd
import os

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score


train_csv = pd.read_csv("../input/train_set.csv")
test_csv = pd.read_csv("../input/test_set.csv")
tube_csv = pd.read_csv("../input/tube.csv")
bill_of_materials_csv = pd.read_csv("../input/bill_of_materials.csv")

comp_filenames = [f for f in os.listdir("../input/") if f.startswith("comp_")]
component_files = [pd.read_csv("../input/" + f) for f in comp_filenames]
components = pd.concat(component_files, axis=0)

train_tube = pd.merge(train_csv, tube_csv, how='left', on='tube_assembly_id')
test_tube = pd.merge(test_csv, tube_csv, how='left', on='tube_assembly_id')

train_tube['material_id'].fillna('SP-9999',inplace=True)
test_tube['material_id'].fillna('SP-9999',inplace=True)

train = train_tube
test  = test_tube
lables = np.log1p(train.cost.values)

train['id'] = np.arange(-1, -len(lables)-1, -1)

train = train.drop(['tube_assembly_id', 'cost', 'quote_date'], axis = 1)
test  = test. drop(['tube_assembly_id', 'quote_date'], axis = 1)

dataset = pd.concat([train, test])

categorial_features = [f for f in dataset.axes[1] if 'object' == dataset.ix[:, f].dtype]
print(categorial_features)

for feat in categorial_features:
    lbl = LabelEncoder()
    lbl.fit(dataset.ix[:, feat])
    dataset.ix[:, feat] = lbl.transform(dataset.ix[:, feat])

train = dataset[dataset['id'] < 0].drop(['id'], axis=1)
test = dataset[dataset['id'] >= 0]
idx = test['id']
test = test.drop(['id'], axis=1)


clf = RandomForestRegressor(n_estimators=1000, criterion="mse", n_jobs=-1)

clf.fit(train, lables)

pred = np.expm1(clf.predict(test))

pred = pd.DataFrame({"id": idx, "cost": pred})
pred.to_csv('submition.csv', index=False)