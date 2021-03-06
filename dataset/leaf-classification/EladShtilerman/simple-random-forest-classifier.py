
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# prepare data
le = LabelEncoder().fit(train.species)
labels = le.transform(train.species)  # species codes
classes = list(le.classes_)  # species names
test_ids = test.id

train = train.drop(['species', 'id'], axis=1)
test = test.drop(['id'], axis=1)

x_train = train.values
y_train = labels

# Predict test data
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
test_predictions = rf.predict_proba(test)

# Format submission
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export submission
submission.to_csv('submission.csv', index = False)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

