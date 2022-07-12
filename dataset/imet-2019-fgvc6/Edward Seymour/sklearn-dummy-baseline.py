# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import csv
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.dummy import DummyClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
datadir = '../input'

with open(os.path.join(datadir, 'train.csv')) as f:
    next(f) # Eat the first line of input
    train_labels = [[int(label)-1 for label in line[1].split(' ')] for line in csv.reader(f, delimiter=',')]
    
y_train = np.zeros((len(train_labels), max(label for labels in train_labels for label in labels)+1))
for i, labels in enumerate(train_labels):
    y_train[i, labels] = 1.
    
baselineClassifier = DummyClassifier(strategy='most_frequent', random_state=0xdeadbeef)
baselineClassifier.fit(y_train, y_train)

images = [image.split('.')[0] for image in os.listdir(os.path.join(datadir, 'test'))]
predictions = baselineClassifier.predict(images)

with open('submission.csv', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(['id', 'attribute_ids'])
    for image, prediction in zip(images, predictions):
        labels = []
        for i, label in enumerate(prediction):
            if label:
                labels.append(str(i+1))
        csv_writer.writerow([image, ' '.join(labels)])

# Any results you write to the current directory are saved as output.
