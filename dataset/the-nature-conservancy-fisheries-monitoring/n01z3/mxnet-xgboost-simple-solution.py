import mxnet as mx
import xgboost as xgb
import numpy as np
import cv2
from multiprocessing import Pool
import os
from sklearn import cross_validation
import joblib


def get_extractor():
    model = mx.model.FeedForward.load('InceptionBN/Inception', 9, ctx=mx.cpu(),
                                      numpy_batch_size=1)

    internals = model.symbol.get_internals()
    fea_symbol = internals["global_pool_output"]

    # if you have GPU, then change ctx=mx.gpu()
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol,
                                             numpy_batch_size=4,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((1, 3, 224, 224))
    img = img[:, :, [2, 1, 0]]

    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    img = img[yy: yy + short_egde, xx: xx + short_egde]

    img = cv2.resize(img, (224, 224))

    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2).astype(np.float32)

    img -= 117
    return img.reshape([1, 3, 224, 224])


def extract_features():
    os.mkdir('feats')
    folders = sorted(os.listdir('train'))
    
    print(folders)
    for n, dir in enumerate(folders):
        folder = os.path.join('train', dir)
        paths = sorted([os.path.join(folder, fn) for fn in os.listdir(folder)])

        pool = Pool(20)
        img_samples = pool.map(preprocess_image, paths)
        samples = np.vstack(img_samples)

        model = get_extractor()
        global_pooling_feature = model.predict(samples)
        np.save(os.path.join('feats', 'feats%s' % n), global_pooling_feature)


def train_xgb():
    x, y = [], []
    for i in range(8):
        feats = np.load('feats/feats%s.npy' % i)
        print(feats.shape)
        feats = feats.reshape((feats.shape[0], 1024))
        x.append(feats)
        y += [i] * feats.shape[0]

    x = np.vstack(x)
    y = np.array(y)

    print(x.shape, y.shape)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                         test_size=0.20)

    clf = xgb.XGBClassifier(max_depth=5,
                            n_estimators=500,
                            learning_rate=0.1,
                            nthread=-1,
                            objective='multi:softmax',
                            seed=42)

    clf.fit(x_train, y_train, early_stopping_rounds=30, eval_metric="mlogloss",
            eval_set=[(x_test, y_test)])

    joblib.dump(clf, "xgb_model")


def predict():
    feats = np.load('feats/feats%s.npy' % 8)
    feats = feats.reshape((feats.shape[0], 1024))

    clf = joblib.load("xgb_model")
    pred = clf.predict_proba(feats)

    print(pred.shape)

    np.save('preds', pred)


def make_submite():
    preds = np.load('preds.npy')
    paths = sorted([fn for fn in os.listdir('train/test')])

    fw_out = open('submission_0.csv', 'w')
    fw_out.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
    for i, img in enumerate(paths):
        pred = ['%.6f' % p for p in preds[i, :]]
        fw_out.write('%s,%s\n' % (img, ','.join(pred)))


if __name__ == '__main__':
    extract_features()
    train_xgb()
    predict()
    make_submite()
