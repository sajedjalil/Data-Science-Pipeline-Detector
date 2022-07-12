#from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as ssignal
import glob
import os
import multiprocessing
import sys

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, StratifiedKFold


# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns

DATA_DIR = "../input"

DATA_LABELS_SAFE = "{}/train_and_test_data_labels_safe.csv".format(DATA_DIR)

# This function is taken from ZFTurbo code
def mat_to_pandas(path):
    mat = sio.loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']
    return pd.DataFrame(ndata['data'], columns=[int(_) for _ in ndata['channelIndices'][0]]).astype(float), sequence

def load_labels_safe():
    labels_safe = pd.read_csv(DATA_LABELS_SAFE)
    print('original files\t{}'.format(labels_safe.shape))
    labels_safe = labels_safe[labels_safe.safe == 1]
    print('safe files\t{}'.format(labels_safe.shape))
    
    splitted = labels_safe.image.str.strip('.mat').str.split('_', expand=True).ix[:, :1]
    splitted.columns = ['patient', 'segment']
    labels_safe = labels_safe.merge(splitted, left_index=True, right_index=True, how='left')
    
    return labels_safe[['patient', 'segment', 'class', 'image']]

def extract_features(path):
    print('Processing {}'.format(path), flush=True)
    features = {}
    try:
        df, sq = mat_to_pandas(path)

        features.update ({'sequence'  : sq})
        for i in range(1, 17):
            si = str(i)
            signal = df[i].values
            f, pxx = ssignal.periodogram(signal, 1, window=None)
            features.update(
                {
                    'mean_' + si: df[i].mean(),
                    'std_'  + si: df[i].std(),
                    'skew_' + si: df[i].skew(),
                    'kurtosis_' + si: df[i].kurtosis(),
                    'rms_' + si: (signal**2).mean()**0.5,
                    'psd_max_f_' + si: f[np.argmax(pxx)],
                    'psd_max_' + si: pxx.max(),
                    'psd_mean_' + si: pxx.mean(),
                    'psd_std_' + si: pxx.std(),            
                })
    except:
        print('could not process {}'.format(path), flush=True)
    
    return features

def process_single_train_file(i, path, progress, train_images):
    img = os.path.basename(path)
    image_features = {'image': img}
    if img in train_images:
        image_features.update(extract_features(path))
    
    if i % 50  == 0:
        print('{0:3.2f}%'.format(progress), end=" ", flush=True)

    return image_features

def process_single_test_file(i, path, progress):
    img = os.path.basename(path)
    patient, segment = img[4:-4].split('_')
    image_features = {'image': img, 'patient': patient, 'segment': segment}
    image_features.update(extract_features(path))
    
    if i % 50  == 0:
        print('{0:3.2f}%'.format(progress), end=" ", flush=True)

    return image_features

def process_matlab_files(file_sets, train=True, train_images=[]):
    pool = multiprocessing.Pool(5)
    results = []
    
    paths = []
    for file_set in file_sets:
        paths += sorted(glob.glob("{0}/{1}/*.mat".format(DATA_DIR, file_set)))
    
    if train:    
        print('processing MATLAB train files ...', end=" ", flush=True)
    else:
        print('processing MATLAB test files ...', end=" ", flush=True)
        
    for i, path in enumerate(paths):
        progress = (i + 1) * 100.0 / len(paths)
        if train:
            results.append(pool.apply_async(process_single_train_file, [i, path, progress, train_images]))
        else:
            results.append(pool.apply_async(process_single_test_file, [i, path, progress]))
            
    pool.close()
    pool.join()
    
    features = [_.get() for _ in results]
    dfx = pd.DataFrame(features)
    return dfx


def main():

    train = load_labels_safe()
    train.to_csv('train_1.csv', index=False)
    #dftrain1 = process_matlab_files(['test_1','test_1', 'test_3'], train=False, train_images=train.image.values)
    #dftrain1.to_csv('train_features1.csv', index=False)
    dftrain2 = process_matlab_files(['test_1','test_1', 'test_3'], train=True, train_images=train.image.values)
    dftrain2.to_csv('train_features2.csv', index=False)
    #dftrain = process_matlab_files(['test_1','train_1', 'train_2', 'train_3'], train=True, train_images=train.image.values)
    #dftrain.to_csv('train_features3.csv', index=False)
    train = train.merge(dftrain2, how='left', left_on='image', right_on='image')
    train.to_csv('train2.csv', index=False)
    dftest = process_matlab_files(['test_1_new', 'test_2_new', 'test_3_new'], train=False)
    dftest.to_csv('test_features3.csv', index=False)
    test = dftest
    
    return

    
    # def twoplot(df, col, xaxis=None, response="Response"):
    #     ''' scatter plot a feature split into response values as two subgraphs '''
    #     if col not in df.columns.values:
    #         print('ERROR: %s not a column' % col)
    #     ndf = pd.DataFrame(index = df.index)
    #     ndf[col] = df[col]
    #     ndf[xaxis] = df[xaxis] if xaxis else df.index
    #     ndf[response] = df[response]
    
    #     g = sns.FacetGrid(ndf, col=response, hue=response)
    #     g.map(plt.scatter, xaxis, col, alpha=.7, s=1)
    #     g.add_legend()
                          
    #     del ndf
    
    # for c in np.setdiff1d(train.columns, ['class', 'image'])[:20]:
    #     twoplot(train[train.patient=='1'], c, response='class')

    Xtrain = train[np.setdiff1d(train.columns, ['class', 'image', 'sequence'])]
    ytrain = train['class']
    groups = train['sequence']

    Xtrain = Xtrain.loc[~train.mean_1.isnull(), :]
    ytrain = ytrain.loc[~train.mean_1.isnull()].values.ravel()
    groups = groups.loc[~train.mean_1.isnull()].values.ravel()

    print(Xtrain.shape, ytrain.shape)

    Xtest = test[Xtrain.columns]
    print(Xtest.shape)


    gkf = GroupKFold(n_splits=5)
    skf = StratifiedKFold(n_splits=5)
    proba = []
    real = []
    scores = []
    lrs = []
    for train_inds, test_inds in gkf.split(Xtrain, ytrain, groups=groups):
    #for train_inds, test_inds in skf.split(Xtrain, ytrain):
        
        Xr = Xtrain.iloc[train_inds, :]
        yr = ytrain[train_inds]
        
        lr = LogisticRegression()
        lr.fit(Xr, yr)
        lrs.append(lr)
    
        Xt = Xtrain.iloc[train_inds, :]
        yt = ytrain[train_inds]
        
        yp = lr.predict_proba(Xt)[:,1]
        
        proba.extend(yp)
        real.extend(yt)
        score = roc_auc_score(y_true=yt, y_score=yp)
        print ('AUC score for this fold {}'.format(score))
        scores.append(score)
        
    print ('AUC score for all CV: {}'.format(roc_auc_score(y_true=real, y_score=proba)))
    print ('AUC score for all: {0}+{1}'.format(np.array(scores).mean(), np.array(scores).std()))
        
    
    yproba = np.zeros(len(ytrain))
    for lr in lrs:
        yproba += lr.predict_proba(Xtrain)[:,1]
    yproba = yproba/5
    print (roc_auc_score(y_true=ytrain, y_score=yproba))
    
    
        
    
    ytproba = np.zeros(Xtest.shape[0])
    for lr in lrs:
        ytproba += lr.predict_proba(Xtest)[:,1]
    ytproba = ytproba/5
    #print (roc_auc_score(y_true=ytrain, y_score=yproba))
    
    sub = pd.read_csv('{}/sample_submission.csv'.format(DATA_DIR))
    print(sub.head())
    sub['File'] = test['image']
    sub['Class'] = ytproba
    print(sub.head())
    sub.to_csv('sub1.csv', index=False)
    
    
if __name__ == "__main__":
    main()