__author__ = 'n01z3'

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import pandas as pd
import glob
import os
import gc

import tensorflow as tf
from multiprocessing import Pool

FOLDER = '/media/n01z3/DATA/dataset/yt8m/' #path to: train, val, test folders with *tfrecord files


def ap_at_n(data):
    # based on https://github.com/google/youtube-8m/blob/master/average_precision_calculator.py
    predictions, actuals = data
    n = 20
    total_num_positives = None

    if len(predictions) != len(actuals):
        raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)

    ap = 0.0

    sortidx = np.argsort(predictions)[::-1]

    if total_num_positives is None:
        numpos = np.size(np.where(actuals > 0))
    else:
        numpos = total_num_positives

    if numpos == 0:
        return 0

    if n is not None:
        numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap


def gap(pred, actual):
    lst = zip(list(pred), list(actual))

    with Pool() as pool:
        all = pool.map(ap_at_n, lst)

    return np.mean(all)


def tf_itr(tp='test', batch=1024):
    tfiles = sorted(glob.glob(os.path.join(FOLDER, tp, '*tfrecord')))
    print('total files in %s %d' % (tp, len(tfiles)))
    ids, aud, rgb, lbs = [], [], [], []
    for index_i, fn in enumerate(tfiles):
        print("\rLoading files: [{0:50s}] {1:.1f}%".format('#' * int((index_i+1)/len(tfiles) * 50), (index_i+1)/len(tfiles)*100), end="", flush=True)
        for example in tf.python_io.tf_record_iterator(fn):
            tf_example = tf.train.Example.FromString(example)
            ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
            rgb.append(np.array(tf_example.features.feature['mean_rgb'].float_list.value))
            aud.append(np.array(tf_example.features.feature['mean_audio'].float_list.value))

            yss = np.array(tf_example.features.feature['labels'].int64_list.value)
            out = np.zeros(4716).astype(np.int8)
            for y in yss:
                out[y] = 1
            lbs.append(out)
            if len(ids) >= batch:
                yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
                ids, aud, rgb, lbs = [], [], [], []
        if index_i+1==len(tfiles):
            yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
            ids, aud, rgb, lbs = [], [], [], []


def fc_block(x, n=1024, d=0.2):
    x = Dense(n, init='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x


def build_mod():
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1)

    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2)

    x = merge([x1, x2], mode='concat', concat_axis=1)
    x = fc_block(x)
    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # model.summary()
    return model


def train():
    if not os.path.exists('weights'): os.mkdir('weights')
    batch = 10 * 1024
    n_itr = 10
    n_eph = 100

    _, x1_val, x2_val, y_val = next(tf_itr('val', 10000))

    model = build_mod()
    cnt = 0
    for e in range(n_eph):
        for d in tf_itr('train', batch):
            _, x1_trn, x2_trn, y_trn = d
            model.train_on_batch({'x1': x1_trn, 'x2': x2_trn}, {'output': y_trn})
            cnt += 1
            if cnt % n_itr == 0:
                y_prd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=False, batch_size=100)
                g = gap(y_prd, y_val)
                print('val GAP %0.5f; epoch: %d; iters: %d' % (g, e, cnt))
                model.save_weights('weights/%0.5f_%d_%d.h5' % (g, e, cnt))

def conv_pred(el):
    t = 20
    idx = np.argsort(el)[::-1]
    return ' '.join(['{} {:0.5f}'.format(i, el[i]) for i in idx[:t]])


def predict():
    
    model = build_mod()
    
    batch = 100000
    
    wfn = sorted(glob.glob('weights/*.h5'))[-1]
    model.load_weights(wfn)
    print('loaded weight file: %s' % wfn)
    
    cnt = 0
    for d in tf_itr('test', batch):
        cnt += 1
        idx, x1_val, x2_val, _ = d
        print("\n")
        ypd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=1, batch_size=32)
        del x1_val, x2_val
    
        with Pool() as pool:
            out = pool.map(conv_pred, list(ypd))
    
        df = pd.DataFrame.from_dict({'VideoId': idx, 'LabelConfidencePairs': out})
        df.to_csv(FOLDER+'submissions/'+'subm'+str(cnt)+'.csv', header=True, index=False, columns=['VideoId', 'LabelConfidencePairs'])
        gc.collect()

    f_subs = glob.glob(os.path.join(FOLDER,'submissions',"subm*.csv"))
    df = pd.concat((pd.read_csv(f) for f in f_subs))
    df.to_csv(os.path.join(FOLDER,'submissions',"all_subs.csv"),index=None)

if __name__ == '__main__':
    train()
    predict()
