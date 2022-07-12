import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict
from sklearn.utils.class_weight import compute_class_weight

import logging
import time
import sys
import gc

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
if not logger.hasHandlers():
  logger.addHandler(logging.StreamHandler(sys.stdout))
logger.handlers[0].setFormatter(formatter)

train_path = '../input/train.csv'
header = pd.read_csv(train_path, nrows=5)

for name in header.columns:
  if name != 'click_epoch' and str(header[name].dtype) == 'int64':
    if name == 'is_attributed':
      header[name] = header[name].astype(np.int8)
    else:
      header[name] = header[name].astype(np.int32)

header['click_epoch'] = pd.to_datetime(header['click_time']).astype(np.int64) // int(1e9)
header['hour'] = pd.to_datetime(header['click_time']).dt.hour  
dtypes = header.dtypes.to_dict()
del dtypes['attributed_time']
features_names = [n for n in header.columns if not n in
  ['click_time','attributed_time', 'is_attributed']]
features_idx = [list(header.columns).index(n) for n in features_names]
assert(features_names == [header.columns[i] for i in features_idx])

def compute_required_stat(train_path):
  total_data, n_attributed = 0, 0
  
  with open(train_path, 'rb') as fp:
    _ = fp.readline() # skip header
    for line in fp:
      n_attributed += (int(line.rsplit(b',', maxsplit=1)[-1]) == 1)
      total_data += 1

  fake_y = np.zeros((total_data,), dtype=np.int8)
  fake_y[:n_attributed] = 1
  class_weights = compute_class_weight('balanced', np.arange(2), fake_y)
  class_weights /= class_weights[0]
  class_weights[1] /= 3
  class_weights = dict([(i, v) for i, v in enumerate(class_weights)])
  return class_weights, total_data, n_attributed 
  
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
import sys

from joblib import Parallel, delayed
from sklearn.base import clone 
from sklearn.preprocessing import label_binarize
from functools import reduce
from operator import add
import scipy as sp

def parallel_fit(hasher, X):
  return hasher.fit_transform(X)
  
def expand_features(hasher, data_set):
  denses, col_spm = [], []
  for k, v in data_set.items():
    if str(v.dtype) == 'object': # string type using FeatureHasher
      col_spm.append(v)
    else:
      denses.append(v.astype(np.float32)[:, np.newaxis])
  sparses = Parallel(n_jobs=2)(delayed(parallel_fit)(clone(hasher), v) for v in col_spm)
  if sparses and denses:
    return sp.sparse.hstack([np.hstack(denses)] + [reduce(add, sparses)]).tocsr()
  elif sparses and not denses:
    return reduce(add, sparses)
  elif denses and not sparses:
    return np.hstack(denses)
    
np.random.seed(42)

def iterate_file(file_path, chunk_size, total_chunks, test_size=0., 
                 binarize_labels=None):
  data_source = pd.read_csv(file_path, iterator=True, dtype=dtypes,
                            usecols=lambda x: x is not "attributed_time")
  n_chunk = 0
  for n_chunk in range(total_chunks):
    try:
      if n_chunk == total_chunks - 1: 
        # read all the remaining data 
        chunk = data_source.get_chunk()
      else:
        chunk = data_source.get_chunk(chunk_size)
    except StopIteration:
      break
    else:
      chunk['click_time'] = pd.to_datetime(chunk['click_time'])
      chunk['click_epoch'] = chunk['click_time'].astype(np.int64) // int(1e9)
      chunk['hour'] = chunk['click_time'].dt.hour   
      if binarize_labels is not None:
        for name in binarize_labels:
          chunk[name] = chunk[name].apply(lambda x: "%s=%s" %(name, x))
      chunk = chunk.set_index('click_time')
      if test_size > 0:
        raise NotImplementedError('no validation')
      else:
        train_idx, test_idx = np.arange(len(chunk)), None
      train_set = chunk.iloc[train_idx,].to_records(index=False,
                                                     convert_datetime64=False)
      if test_idx is not None:
        test_set = chunk.iloc[test_idx,].to_records(
          index=False, convert_datetime64=False)
        yield (OrderedDict([(name, train_set[name]) for name in features_names]), 
               train_set['is_attributed']), \
              (OrderedDict([(name, test_set[name]) for name in features_names]), 
                 test_set['is_attributed'])
      else:
        if 'is_attributed' in chunk:
          yield (OrderedDict([(name, train_set[name]) for name in features_names]), 
                 train_set['is_attributed']), None
        else:
           yield (OrderedDict([(name, train_set[name]) for name in features_names]), 
                 ), None
                 
def training_loop(chunk_size, binarize_labels=None, 
                  use_regressor=False):
  class_weights, total_data, n_attributed = compute_required_stat(train_path)
  total_chunks = total_data // chunk_size
  params = {'n_estimators': 30,
            'verbose': 0, 
            'bootstrap': False, 
            'oob_score': False, 
            'min_samples_split': 2*(n_attributed/(1000*total_data)),
            'min_samples_leaf': (n_attributed/(1000*total_data)),
            'n_jobs':-1,
            'warm_start':True,
            }
  if use_regressor:
    embedder = ExtraTreesRegressor(criterion='mse', **params)
  else:
    embedder = ExtraTreesClassifier(criterion='gini',  # gini or entropy
                                     # to counter imbalance classification
                                    class_weight=class_weights, 
                                    **params)
  hasher = FeatureHasher(
    n_features=2**18,  # maybe change to increase efficiency
    input_type='string', 
    dtype=np.float32)
  
  valid_auc = []
  for i, (train_set, test_set) in enumerate(
    iterate_file(train_path, chunk_size, total_chunks,  
                 binarize_labels=binarize_labels)):
    if i == 0:
      logging.info("features used={}".format(
        ','.join(["%s:%s"%(k, np.array2string(v[:2])) 
                             for k, v in train_set[0].items()])))
    Xt = expand_features(hasher, train_set[0])
    if i > 0: # adding more n_estimators
      n_ests = embedder.n_estimators
      embedder.set_params(n_estimators=(n_ests + 5))
    if use_regressor:
      noises = np.random.normal(0.25, 0.1, size=(train_set[-1] == 0).sum())
      if np.any(noises < 0):
        noises[noises < 0] = 0
      yt = train_set[-1].astype(np.float32)
      yt[train_set[-1] == 0] = yt[train_set[-1] == 0] + noises
      logging.debug("generating %d noises for not_attributed %.3f" % (
        (noises!=0).sum(), yt[yt!=1].mean()))
    else:
      yt = train_set[-1]
    embedder = embedder.fit(Xt, yt)
    
    if use_regressor:
      y_pred = embedder.predict(Xt)
    else:
      y_pred = embedder.predict_proba(Xt)[:, 1]
      
    logging.info("chunk #{:2d}, read in {:3.1f}% data, "
                 "percent of attributed ={:3.2f}%, n_estimators = #{:2d}, "
                 "training AUC = {:1.4f}".format(
        i, 100*(i * chunk_size + Xt.shape[0]) / total_data, 
           100*(train_set[-1] == 1).mean(), 
                     embedder.n_estimators,
    roc_auc_score(train_set[-1], y_pred)))
    if test_set is not None:
      Xt = expand_features(hasher, test_set[0])
      if use_regressor:
        y_pred = np.clip(embedder.predict(Xt), 0, 1)
      else:
        y_pred = embedder.predict_proba(Xt)[:, 1]
      valid_auc.append(roc_auc_score(test_set[-1], y_pred))
  return embedder, hasher, valid_auc

chunk_size = int(1e7)
embedder, hasher, _ = training_loop(chunk_size, binarize_labels=None, 
                                     use_regressor=False)

with open('../input/test.csv') as fp:
    test_set, _ = next(iterate_file(fp, None, 1))
    Xt = expand_features(hasher, test_set[0])
    y_pred = embedder.predict_proba(Xt)[:, 1]

with open('../input/sample_submission.csv') as fp:
    submission = pd.read_csv(fp)


submission['is_attributed'] = y_pred
submission.to_csv('submission.csv', index=False)