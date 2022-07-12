import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


batch_size = 100
features = 64

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

le = LabelEncoder().fit(train.species) 
y_train = le.transform(train.species)

x_train = train.drop(['id', 'species'], axis=1).values
x_test = test.drop(['id'], axis=1).values

scaler = StandardScaler().fit(x_train)    
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=features)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[100, 200, 100],
                                            n_classes=99)

classifier.fit(x=x_train, y=y_train, steps = 2000)
y_test = list(classifier.predict_proba(x_test, as_iterable=True))


test_ids = test.pop('id')
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')