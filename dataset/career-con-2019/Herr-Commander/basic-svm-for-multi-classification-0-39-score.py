import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import skimage
import sklearn as sk
import scipy
import random
import skimage
from skimage import transform
from skimage import util
import os

import progressbar

import matplotlib.pyplot as plt

y_train_csv = pd.read_csv('../input/y_train.csv').values

if os.path.isfile('X_train.npy'):
    X_train = np.load('X_train.npy')
else:
    with progressbar.ProgressBar(max_value=3810) as bar:
        X_train_csv = pd.read_csv('../input/X_train.csv').values
        X_train = np.array([])
        res = []
        prev = 0
        for i in X_train_csv:
            if i[1] != prev:
                if prev == 0:
                    X_train = np.array([res], dtype=np.float32)
                else:
                    X_train = np.append(X_train, [res], axis=0)
                prev = i[1]
                res = []
            res += [i[3:]]
            bar.update(prev)
        X_train = np.append(X_train, [res], axis=0)
        np.save('X_train', X_train)

if os.path.isfile('X_test.npy'):
    X_test = np.load('X_test.npy')
else:
    with progressbar.ProgressBar(max_value=3816) as bar:
        X_test_csv = pd.read_csv('../input/X_test.csv').values
        X_test = np.array([])
        res = []
        prev = 0
        for i in X_test_csv:
            if i[1] != prev:
                if prev == 0:
                    X_test = np.array([res], dtype=np.float32)
                else:
                    X_test = np.append(X_test, [res], axis=0)
                prev = i[1]
                res = []
            res += [i[3:]]
            bar.update(prev)
        X_test = np.append(X_test, [res], axis=0)
        np.save('X_test', X_test)

y_train = y_train_csv[:, 2]
classes = ['fine_concrete', 'concrete', 'soft_tiles', 'tiled', 'soft_pvc',
           'hard_tiles_large_space', 'carpet', 'hard_tiles', 'wood']


from sklearn.svm import SVC

X_train_flatten = [i.flatten() for i in X_train]

clf = SVC(gamma='auto')
X_train_flatten = np.array(X_train_flatten)
clf.fit(X_train_flatten, y_train)

clf.score(X_train_flatten, y_train)


X_test_flatten = [i.flatten() for i in X_test]
res = clf.predict(X_test_flatten)

df = pd.DataFrame({'series_id': np.arange(len(X_test_flatten)), 'surface': res})
f = open('submission.csv', 'w')
f.write(df.to_csv(index=False))
print('Answer Saved')
