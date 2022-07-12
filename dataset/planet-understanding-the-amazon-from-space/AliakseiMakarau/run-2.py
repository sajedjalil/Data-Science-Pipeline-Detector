import numpy
import os
import pandas
import random
from tqdm import tqdm
import xgboost as xgb
import scipy
import pdb
from sklearn.metrics import fbeta_score
from PIL import Image

import tifffile                                                                                                                                                                                                                                                                                                                
from tifffile import imread   

random_seed = 0
random.seed(random_seed)
numpy.random.seed(random_seed)


# Load data
#train_path = '../input/train-jpg/'
#test_path = '../input/test-jpg/'

#train_path = '../input/train-tif-sample/'
#test_path = '../input/test-tif-sample/'

train_path = '../input/train-tif-v2/'
test_path = '../input/test-tif-v2/'

train = pandas.read_csv('../input/train.csv')
test = pandas.read_csv('../input/sample_submission.csv')


def extract_features(df, data_path):
    im_features = df.copy()

    r_max = []
    g_max = []
    b_max = []

    r_min = []
    g_min = []
    b_min = []

    for image_name in tqdm(im_features.image_name.values, miniters=100):
        im = tifffile.imread(data_path + image_name + '.tif')
        im = numpy.array(im)
        print(image_name)
        print(im.shape)
        print(im[:, :, 0])
        print(im[:, :, 1])
        print(im[:, :, 2])
        print(im[:, :, 3])
        quit()

        im = numpy.array(im)[:, :, :3]

        # here change to tiff


        r_max.append(numpy.max(im[:,:,0].ravel()))
        g_max.append(numpy.max(im[:,:,1].ravel()))
        b_max.append(numpy.max(im[:,:,2].ravel()))

        r_min.append(numpy.min(im[:,:,0].ravel()))
        g_min.append(numpy.min(im[:,:,1].ravel()))
        b_min.append(numpy.min(im[:,:,2].ravel()))

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min

    return im_features


if __name__ == '__main__':
    # Extract features
    print('Extracting train features')
    train_features = extract_features(train, train_path)
    print('Extracting test features')
    test_features = extract_features(test, test_path)

    '''
    # Prepare data
    X = numpy.array(train_features.drop(['image_name', 'tags'], axis=1))
    y_train = []

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = numpy.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))

    label_map     = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    for tags in tqdm(train.tags.values, miniters=1000):
        targets = numpy.zeros(17)

        for t in tags.split(' '):
            targets[label_map[t]] = 1

        y_train.append(targets)

    y = numpy.array(y_train, numpy.uint8)

    print('X.shape = ' + str(X.shape))
    print('y.shape = ' + str(y.shape))

    n_classes = y.shape[1]
    X_test = numpy.array(test_features.drop(['image_name', 'tags'], axis=1))

    # Train and predict with one-vs-all strategy
    y_pred = numpy.zeros((X_test.shape[0], n_classes))

    print('Training and making predictions')

    for class_i in tqdm(range(n_classes), miniters=1): 
        model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, \
                                  silent=True, objective='binary:logistic', nthread=-1, \
                                  gamma=0, min_child_weight=1, max_delta_step=0, \
                                  subsample=1, colsample_bytree=1, colsample_bylevel=1, \
                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, \
                                  base_score=0.5, seed=random_seed, missing=None)

        model.fit(X, y[:, class_i])
        y_pred[:, class_i] = model.predict_proba(X_test)[:, 1]


    pp = [y_pred_row for y_pred_row in y_pred]
    preds = [' '.join(labels[ll > 0.2]) for ll in pp]

    subm = pandas.DataFrame()
    subm['image_name'] = test_features.image_name.values
    subm['tags'] = preds
    subm.to_csv('submission.csv', index=False)
    '''
