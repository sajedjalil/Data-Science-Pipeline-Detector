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

from subprocess import check_output
print(check_output(["ls", "../input/train"]).decode("utf8"))

import glob
train= glob.glob("../input/train/**/*.jpg")+glob.glob("../input/additional/**/*.jpg")
train = pd.DataFrame([[p.split('/')[3],p.split('/')[4],p] for p in train], columns = ['type','image','path'])
test = glob.glob("../input/test/*.jpg")
test = pd.DataFrame([[p.split('/')[3],p] for p in test], columns = ['image','path'])

from PIL import ImageFilter, ImageStat, Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import cv2

types = train.groupby('type', as_index=False).count()
types.plot(kind='bar', x='type', y='path', figsize=(7,4))

types

from multiprocessing import Pool, cpu_count

def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df

train = im_stats(train)
sizes = train.groupby('size', as_index=False)['path'].count()
_ = sizes.plot(kind='bar', x='size', y='path', figsize=(7,4))
#../input/additional/Type_1/5893.jpg
#../input/additional/Type_2/2845.jpg
#../input/additional/Type_2/5892.jpg

sizes

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return [path, resized]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

#train = glob.glob('../input/train/**/*.jpg') + glob.glob('../input/additional/**/*.jpg')
print(len(train))
#train = pd.DataFrame([[p.split('/')[3],p.split('/')[4],p] for p in train], columns = ['type','image','path'])
train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
train_data = normalize_image_features(train['path'])
#np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)
#train_data = np.load('train.npy')

le = LabelEncoder()
train_target = le.fit_transform(train['type'].values)
#np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)
#train_target = np.load('train_target.npy')

def create_model(opt_):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64)))
    model.add(Convolution2D(8, 3, 3))
    model.add(Dropout(0.2))
    model.add(Dense(12))
    model.add(Dropout(0.2))
    model.add(Dense(6))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy']) #loss='binary_crossentropy' not working
    return model

model = KerasClassifier(build_fn=create_model, nb_epoch=9, batch_size=15, verbose=2)

opts_ = ['adamax'] #['adadelta','sgd','adagrad','adam','adamax']
epochs = np.array([10])
batches = np.array([15])
param_grid = dict(nb_epoch=epochs, batch_size=batches, opt_=opts_)
grid = GridSearchCV(estimator=model, cv=StratifiedKFold(n_splits=2), param_grid=param_grid, verbose=20)
grid_result = grid.fit(train_data, train_target)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
#print("Log Loss...", log_loss(train_target, grid_result.predict(train_data)))

test_data = normalize_image_features(test['path'])
#np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)
#test_data = np.load('test.npy')
test_id = test.image.values
#np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)
#test_id = np.load('test_id.npy')

pred = grid_result.predict_proba(test_data)
df = pd.DataFrame(pred, columns=le.classes_)
df['image_name'] = test_id
df.to_csv('submission 05-05 01.csv', index=False)