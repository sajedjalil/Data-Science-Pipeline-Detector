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
data_dir = '../input/'

def load_data(data_dir):
    train = pd.read_json(data_dir + 'train.json')
    test = pd.read_json(data_dir + 'test.json')
    #Fill 'na' angles with mode
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    test.inc_angle = test.inc_angle.replace('na', 0)
    test.inc_angle = test.inc_angle.astype(float).fillna(0.0)
    return train, test

train, test = load_data(data_dir)

def color_composite(data):
    import cv2
    w,h = 75,75
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        #Add in to resize for resnet50 use 197 x 197
        rgb = cv2.resize(rgb, (w,h)).astype(np.float32)
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)

def get_stats(train,label=1):
    train['max'+str(label)] = [np.max(np.array(x)) for x in train['band_'+str(label)] ]
    train['maxpos'+str(label)] = [np.argmax(np.array(x)) for x in train['band_'+str(label)] ]
    train['min'+str(label)] = [np.min(np.array(x)) for x in train['band_'+str(label)] ]
    train['minpos'+str(label)] = [np.argmin(np.array(x)) for x in train['band_'+str(label)] ]
    train['med'+str(label)] = [np.median(np.array(x)) for x in train['band_'+str(label)] ]
    train['std'+str(label)] = [np.std(np.array(x)) for x in train['band_'+str(label)] ]
    train['mean'+str(label)] = [np.mean(np.array(x)) for x in train['band_'+str(label)] ]
    train['p25_'+str(label)] = [np.sort(np.array(x))[int(0.25*75*75)] for x in train['band_'+str(label)] ]
    train['p75_'+str(label)] = [np.sort(np.array(x))[int(0.75*75*75)] for x in train['band_'+str(label)] ]
    train['mid50_'+str(label)] = train['p75_'+str(label)]-train['p25_'+str(label)]

    return train

train_feat = get_stats(train,1)
train_feat = get_stats(train,2)
test_feat = get_stats(test,1)
test_feat = get_stats(test,2)

x_train_feat = np.array(train_feat[['max1', 'maxpos1', 'min1', 'minpos1', 'med1', 'std1', 'mean1','p25_1', 'p75_1', 'mid50_1',
               'max2', 'maxpos2', 'min2', 'minpos2', 'med2', 'std2', 'mean2','p25_2', 'p75_2', 'mid50_2']])
x_test_feat = np.array(test_feat[['max1', 'maxpos1', 'min1', 'minpos1', 'med1', 'std1', 'mean1','p25_1', 'p75_1', 'mid50_1',
               'max2', 'maxpos2', 'min2', 'minpos2', 'med2', 'std2', 'mean2','p25_2', 'p75_2', 'mid50_2']])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train_feat)
x_train_feat = scaler.transform(x_train_feat)
x_test_feat = scaler.transform(x_test_feat)

del train_feat, test_feat

print(x_train_feat.shape)
print(x_test_feat.shape)


rgb_train = color_composite(train)
rgb_test = color_composite(test)

y_train = np.array(train['is_iceberg'])

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Concatenate, Merge, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

modelCNN = Sequential()
modelCNN.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(75,75,3)))
modelCNN.add(Conv2D(32, (5, 5), activation='relu'))
modelCNN.add(MaxPooling2D(pool_size=(2, 2), strides=2))

modelCNN.add(Conv2D(64, (5, 5), activation='relu'))
#modelCNN.add(MaxPooling2D(pool_size=(3, 3), strides=2))

#modelCNN.add(Conv2D(128, (3,3), activation='relu'))
#modelCNN.add(MaxPooling2D(pool_size=(3, 3), strides=2))
modelCNN.add(Flatten())
#modelCNN.add(Dense(64, activation='relu'))

modelFEAT = Sequential()
modelFEAT.add(Dense(32, input_dim=20, activation='relu'))
modelFEAT.add(Dropout(0.5))
modelFEAT.add(Dense(64, activation='relu'))
modelFEAT.add(Dropout(0.5))


merged = Merge([modelCNN, modelFEAT], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit([rgb_train, x_train_feat], y_train, batch_size=32, epochs=50)

test_predictions = model.predict([rgb_test, x_test_feat])
# Create .csv
pred_df = test[['id']].copy()
pred_df['is_iceberg'] = test_predictions
pred_df.to_csv('predictionsShallowNet.csv', index = False)


