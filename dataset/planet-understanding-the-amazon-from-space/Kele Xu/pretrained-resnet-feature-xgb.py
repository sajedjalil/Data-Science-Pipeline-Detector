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
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input

import scipy
from sklearn.metrics import fbeta_score

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

n_classes = 17

train_path = "../input/train-jpg"
test_path = "../input/train-jpg/test-jpg"
train = pd.read_csv("../input/train_v2.csv")
test = pd.read_csv("../input/sample_submission_v2.csv")

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# use ResNet50 model extract feature from fc1 layer
base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
input = Input(shape=(224,224,3),name = 'image_input')
x = base_model(input)
x = Flatten()(x)
model = Model(inputs=input, outputs=x)

X_train = []
y_train = []

for f, tags in tqdm(train.values[:], miniters=1000):
    # preprocess input image
    img_path = train_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features_reduce =  features.squeeze()
    X_train.append(features_reduce)

    # generate one hot vector for label
    targets = np.zeros(n_classes)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)

X = np.array(X_train)
y = np.array(y_train, np.uint8)

X_test = []

for f, tags in tqdm(test.values[:], miniters=1000):
    img_path = test_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # generate feature [4096]
    features = model.predict(x)
    features_reduce = features.squeeze()
    X_test.append(features_reduce)


print('Training and making predictions')
for class_i in tqdm(range(n_classes), miniters=1):
    model = xgb.XGBClassifier(max_depth=15, learning_rate=0.1, n_estimators=200, \
                              objective='binary:logistic', nthread=-1, \
                              subsample=0.7, colsample_bytree=0.7, seed=random_seed, missing=None)
    model.fit(X, y[:, class_i])
    y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]

preds = []
scores = []
for y_pred_row in y_pred:
    result = []
    full_result = []
    for i, value in enumerate(y_pred_row):
        full_result.append(str(i))
        full_result.append(str(value))
        if value > 0.2:
            result.append(labels[i])
    preds.append(" ".join(result))
    scores.append(" ".join(full_result))

orginin = pd.DataFrame()
orginin['image_name'] = test.image_name.values
orginin['tags'] = scores
orginin.to_csv('ResNet_XGB_result.csv', index=False)