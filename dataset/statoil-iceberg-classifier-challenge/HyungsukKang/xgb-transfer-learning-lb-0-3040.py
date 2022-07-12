import numpy as np # linear algebra
np.random.seed(42)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from sklearn.model_selection import train_test_split
print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

#Load data
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
print("done!")


# Preprocess data
def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))
        
        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)


X_train = color_composite(train)
X_angle_train = np.array(train.inc_angle)
y_train = np.array(train["is_iceberg"])

X_test = color_composite(test)
X_angle_test = np.array(test.inc_angle)


X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
                    , X_angle_train, y_train, random_state=123, train_size=0.7)


# Resize data
from tqdm import tqdm
width = 299


n = len(X_train)
train_img = np.zeros((n, width, width, 3), dtype=np.float32)


for i in tqdm(range(n)):
    x = X_train[i]
    train_img[i] = resize(x, (299,299), mode='reflect')

n = len(X_valid)
valid_img = np.zeros((n, width, width, 3), dtype=np.float32)


for i in tqdm(range(n)):
    x = X_valid[i]
    valid_img[i] = resize(x, (299,299), mode='reflect')

n = len(X_test)
test_img = np.zeros((n, width, width, 3), dtype=np.float32)


for i in tqdm(range(n)):
    x = X_test[i]
    test_img[i] = resize(x, (299,299), mode='reflect')


# Feature Extraction
def get_features(MODEL, data=None):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalMaxPooling2D()(x)
    cnn_model = Model(inputs, x)

    features = cnn_model.predict(data, batch_size=4, verbose=1)
    return features


inception_features = get_features(InceptionV3, train_img)
xception_features = get_features(Xception, train_img)
train_features = np.concatenate([inception_features, xception_features], axis=-1)

inception_features = get_features(InceptionV3, valid_img)
xception_features = get_features(Xception, valid_img)
valid_features = np.concatenate([inception_features, xception_features], axis=-1)

inception_features = get_features(InceptionV3, test_img)
xception_features = get_features(Xception, test_img)
test_features = np.concatenate([inception_features, xception_features], axis=-1)

# Save features
np.savez('train_features.npz' , X=train_features)
np.savez('valid_features.npz' , X=valid_features)
np.savez('test_features.npz' , X=test_features)

import xgboost as xgb
train_features_angles = np.concatenate([train_features, X_angle_train.reshape(len(y_train),1)], axis=-1)
valid_features_angles = np.concatenate([valid_features, X_angle_valid.reshape(len(y_valid),1)], axis=-1)
test_features_angles = np.concatenate([test_features, X_angle_test.reshape(8424,1)], axis=-1)
d_train =  xgb.DMatrix(train_features_angles,label=y_train)
d_valid =  xgb.DMatrix(valid_features_angles,label=y_valid)
d_test =  xgb.DMatrix(test_features_angles)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]


params = {
        'objective': 'binary:logistic',
        'n_estimators':1000,
        'max_depth': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.9 ,
        'eta': 0.01,
        'eval_metric': 'logloss'
        }

# train
clf =xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=100,  verbose_eval=100)

# validate

y_preds = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
print('Valid LogLoss {}'.format(log_loss(y_valid, y_preds)))
print('Validation Accuracy {}'.format(accuracy_score(y_valid, np.round(y_preds))))

# predict
xgb_preds = clf.predict(d_test)


#Submit
submission = pd.DataFrame({'id': test["id"], 'is_iceberg': xgb_preds})
submission.to_csv("./submission_xgb.csv", index=False)