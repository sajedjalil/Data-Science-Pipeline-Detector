import sys

sys.path.insert(0, '../input/wordbatch-133/wordbatch/')
sys.path.insert(0, '../input/randomstate/randomstate/')
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
from wordbatch.data_utils import *
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import gc
from contextlib import contextmanager
@contextmanager
def timer(name):
	t0 = time.time()
	yield
	print(f'[{name}] done in {time.time() - t0:.0f} s')

import os, psutil
def cpuStats():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0] / 2. ** 30
	print('memory GB:', memoryUse)

start_time = time.time()

mean_auc= 0

def fit_batch(clf, X, y, w):  clf.partial_fit(X, y, sample_weight=w)

def predict_batch(clf, X):  return clf.predict(X)

def evaluate_batch(clf, X, y, rcount):
	auc= roc_auc_score(y, predict_batch(clf, X))
	global mean_auc
	if mean_auc==0:
		mean_auc= auc
	else: mean_auc= 0.2*(mean_auc*4 + auc)
	print(rcount, "ROC AUC:", auc, "Running Mean:", mean_auc)
	return auc

def df_add_counts(df, cols, tag="_count"):
	arr_slice = df[cols].values
	unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
									 return_inverse=True, return_counts=True)
	df["_".join(cols)+tag] = counts[unqtags]
	return df

def df_add_uniques(df, cols, tag="_unique"):
	gp = df[cols].groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]].nunique().reset_index(). \
		rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols)+tag})
	df= df.merge(gp, on=cols[0:len(cols) - 1], how='left')
	return df

def df2csr(wb, df, pick_hours=None):
	df.reset_index(drop=True, inplace=True)
	with timer("Adding counts"):
		df['click_time']= pd.to_datetime(df['click_time'])
		dt= df['click_time'].dt
		df['day'] = dt.day.astype('uint8')
		df['hour'] = dt.hour.astype('uint8')
		del(dt)
		df= df_add_counts(df, ['ip', 'day', 'hour'])
		df= df_add_counts(df, ['ip', 'app'])
		df= df_add_counts(df, ['ip', 'app', 'os'])
		df= df_add_counts(df, ['ip', 'device'])
		df= df_add_counts(df, ['app', 'channel'])
		df= df_add_uniques(df, ['ip', 'channel'])

	with timer("Adding next click times"):
		D= 2**26
		df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
						 + "_" + df['os'].astype(str)).apply(hash) % D
		click_buffer= np.full(D, 3000000000, dtype=np.uint32)
		df['epochtime']= df['click_time'].astype(np.int64) // 10 ** 9
		next_clicks= []
		for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
			next_clicks.append(click_buffer[category]-time)
			click_buffer[category]= time
		del(click_buffer)
		df['next_click']= list(reversed(next_clicks))

	with timer("Log-binning features"):
		for fea in ['ip_day_hour_count','ip_app_count','ip_app_os_count','ip_device_count',
				'app_channel_count','next_click','ip_channel_unique']: 
				    df[fea]= np.log2(1 + df[fea].values).astype(int)

	with timer("Generating str_array"):
		str_array= ("I" + df['ip'].astype(str) \
			+ " A" + df['app'].astype(str) \
			+ " D" + df['device'].astype(str) \
			+ " O" + df['os'].astype(str) \
			+ " C" + df['channel'].astype(str) \
			+ " WD" + df['day'].astype(str) \
			+ " H" + df['hour'].astype(str) \
			+ " AXC" + df['app'].astype(str)+"_"+df['channel'].astype(str) \
			+ " OXC" + df['os'].astype(str)+"_"+df['channel'].astype(str) \
			+ " AXD" + df['app'].astype(str)+"_"+df['device'].astype(str) \
			+ " IXA" + df['ip'].astype(str)+"_"+df['app'].astype(str) \
			+ " AXO" + df['app'].astype(str)+"_"+df['os'].astype(str) \
			+ " IDHC" + df['ip_day_hour_count'].astype(str) \
			+ " IAC" + df['ip_app_count'].astype(str) \
			+ " AOC" + df['ip_app_os_count'].astype(str) \
			+ " IDC" + df['ip_device_count'].astype(str) \
			+ " AC" + df['app_channel_count'].astype(str) \
			+ " NC" + df['next_click'].astype(str) \
			+ " ICU" + df['ip_channel_unique'].astype(str)
		  ).values
	#cpuStats()
	if 'is_attributed' in df.columns:
		labels = df['is_attributed'].values
		weights = np.multiply([1.0 if x == 1 else 0.2 for x in df['is_attributed'].values],
							  df['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
	else:
		labels = []
		weights = []
	return str_array, labels, weights

class ThreadWithReturnValue(threading.Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
		threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
		self._return = None
	def run(self):
		if self._target is not None:
			self._return = self._target(*self._args, **self._kwargs)
	def join(self):
		threading.Thread.join(self)
		return self._return

batchsize = 10000000
D = 2 ** 20

wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
													 "lowercase": False, "n_features": D,
													 "norm": None, "binary": True})
						 , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)
clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
			  D_fm=8, e_noise=0.0, iters=2, inv_link="sigmoid", e_clip=1.0, threads=4, use_avx=1, verbose=0)

dtypes = {
		'ip'            : 'uint32',
		'app'           : 'uint16',
		'device'        : 'uint16',
		'os'            : 'uint16',
		'channel'       : 'uint16',
		'is_attributed' : 'uint8',
		}

p = None
rcount = 0
for df_c in pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv', engine='c', chunksize=batchsize,
#for df_c in pd.read_csv('../input/train.csv', engine='c', chunksize=batchsize, 
						skiprows= range(1,9308569), sep=",", dtype=dtypes):
	rcount += batchsize
	if rcount== 130000000:
		df_c['click_time'] = pd.to_datetime(df_c['click_time'])
		df_c['day'] = df_c['click_time'].dt.day.astype('uint8')
		df_c= df_c[df_c['day']==8]
	str_array, labels, weights= df2csr(wb, df_c, pick_hours={4, 5, 10, 13, 14})
	del(df_c)
	if p != None:
		p.join()
		del(X)
	gc.collect()
	X= wb.transform(str_array)
	del(str_array)
	if rcount % (2 * batchsize) == 0:
		if p != None:  p.join()
		p = threading.Thread(target=evaluate_batch, args=(clf, X, labels, rcount))
		p.start()
	print("Training", rcount, time.time() - start_time)
	cpuStats()
	if p != None:  p.join()
	p = threading.Thread(target=fit_batch, args=(clf, X, labels, weights))
	p.start()
	if rcount == 130000000:  break
if p != None:  p.join()

del(X)
p = None
click_ids= []
test_preds = []
rcount = 0
for df_c in pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv', engine='c', chunksize=batchsize,
#for df_c in pd.read_csv('../input/test.csv', engine='c', chunksize=batchsize,
						sep=",", dtype=dtypes):
	rcount += batchsize
	if rcount % (10 * batchsize) == 0:
		print(rcount)
	str_array, labels, weights = df2csr(wb, df_c)
	click_ids+= df_c['click_id'].tolist()
	del(df_c)
	if p != None:
		test_preds += list(p.join())
		del (X)
	gc.collect()
	X = wb.transform(str_array)
	del (str_array)
	p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
	p.start()
if p != None:  test_preds += list(p.join())

df_sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
df_sub.to_csv("wordbatch_fm_ftrl.csv", index=False)