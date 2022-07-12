import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict
from sklearn.utils.class_weight import compute_class_weight

import logging
import time
import sys

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
      header[name] = header[name].astype('category')
      header[name] = header[name].cat.set_categories([0, 1])
    else:
      header[name] = header[name].astype(np.int32)

header['click_epoch'] = pd.to_datetime(header['click_time']).astype(np.int64) // int(1e9)
header['hour'] = pd.to_datetime(header['click_time']).dt.hour                     
#header['count_per_hour'] = \
#    header['hour'].map(header['ip'].groupby(header['hour']).count())
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

np.random.seed(42)


def iterate_file(file_path, chunk_size, total_chunks, binarize_labels=None):
  dtypes = header.dtypes.to_dict()
  del dtypes['attributed_time']
  data_source = pd.read_csv(file_path, low_memory=True, iterator=True, 
                            dtype=dtypes, usecols=lambda x: x != 'attributed_time')
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
      yield chunk


import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
h2o.init(max_mem_size='13g')


def process_data(file_path=None, pydframe=None):
    if file_path is not None:
      h2o_frame = h2o.import_file(file_path)
    else:
      assert(pydframe is not None)
      h2o_frame = h2o.H2OFrame(pydframe)
    if 'is_attributed' in pydframe:
      h2o_frame[:, 'is_attributed'] = h2o_frame[:, 'is_attributed'].asfactor()
    return h2o_frame


def distributed_training_loop(chunk_size, binarize_labels=None):
  sample_factors = [0.05, 1.0]
  params = {
    'model_id': 'DRF_model_ooc',
    'histogram_type':'Random', # using extra random tree
    'score_tree_interval':5,  
    'balance_classes': True, # this must be true for any imbalancing setting to work
    'max_after_balance_size': 0.95,
    'class_sampling_factors': sample_factors,
    'sample_rate_per_class': sample_factors, # this works better
    'ignore_const_cols':False
  }
  clf = H2ORandomForestEstimator(ntrees=50, **params)
  class_weights, total_data, n_attributed = compute_required_stat(train_path)
  total_chunks = total_data // chunk_size

  for i, this_batch in enumerate(
    iterate_file(train_path, chunk_size, total_chunks,  
                 binarize_labels=binarize_labels)):
    train_frame = process_data(pydframe=this_batch)
    if i > 0:
      model_id = clf.model_id
      n_ests = clf.ntrees
      clf = H2ORandomForestEstimator(checkpoint=model_id, ntrees=(n_ests + 5),
                                     **params)
      assert(clf.model_id == model_id)
      # clf = clf.set_params(ntrees=(n_ests + 5)) # this will retrain everything
    clf.train(x=features_names, y='is_attributed', 
              training_frame=train_frame,
             #ignored_columns=['attributed_time', 'click_time'] # doesn't take effect
                                                                # only specify x explicity
                                                                # works
             ) 
    logging.info("chunk #{:2d} / #{:2d}, read in {:3.1f}% data, "
                 "percent of attributed ={:3.2f}%, n_estimators = #{:2d}, "
                 "training AUC = {:1.4f} LogLoss = {:1.4f}".format(i + 1, total_chunks,
            100*((i * chunk_size + len(this_batch)) / total_data), 
            100*(this_batch["is_attributed"] == 1).mean(), 
                   clf.ntrees, clf.auc(train=True), clf.logloss(train=True)))
  return clf


chunk_size = int(2**23)
clf = distributed_training_loop(chunk_size)


test_batch = next(iterate_file('../input/test.csv', None, 1))
test_frame = process_data(pydframe=test_batch)
y_pred = clf.predict(test_frame)
  
submission = pd.read_csv('../input/sample_submission.csv')
submission['is_attributed'] = y_pred['p1'].as_data_frame()
submission.to_csv('submission.csv', index=False)