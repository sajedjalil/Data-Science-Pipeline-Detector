import os
import time
import pickle
import numpy as np
import pandas as pd

batch_size = 1500
epochs = (4,4,5,4,4)
num_decompose = 24

time_fs = time.time()
np.random.seed(22)

print('read data:')

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_s = df_train.std(axis=0)
drop_cols = df_s[df_s==0].index
Y = np.log1p(df_train.target.values)
ID = df_test.ID.values
df_train = df_train.drop(['ID','target'], axis=1)
df_test = df_test.drop(['ID'], axis=1)
X = pd.concat([df_train,df_test], axis=0, sort=False, ignore_index=True)
X = X.div(X.max(), axis='columns')
y_min = np.min(Y)
y_max = np.max(Y)
Y = (Y - y_min) / (y_max - y_min)
del df_train, df_test, df_s

def get_manifold():
	from mxnet import nd
	from mxnet import ndarray as F
	from mxnet.gluon import Block, nn
	from mxnet.initializer import Uniform

	class Model(Block):
		def __init__(self, num_dim, **kwargs):
			super(Model, self).__init__(**kwargs)
			wi1 = Uniform(0.25)
			wi2 = Uniform(0.1)
			with self.name_scope():
				self.encoder1 = nn.Dense(num_dim//4, in_units=num_dim, weight_initializer=wi1)
				self.encoder2 = nn.Dense(num_dim//16, in_units=num_dim//4, weight_initializer=wi1)
				self.encoder3 = nn.Dense(num_dim//64, in_units=num_dim//16, weight_initializer=wi2)
				self.encoder4 = nn.Dense(num_dim//256, in_units=num_dim//64, weight_initializer=wi2)
				self.decoder4 = nn.Dense(num_dim//64, in_units=num_dim//256, weight_initializer=wi2)
				self.decoder3 = nn.Dense(num_dim//16, in_units=num_dim//64, weight_initializer=wi2)
				self.decoder2 = nn.Dense(num_dim//4, in_units=num_dim//16, weight_initializer=wi1)
				self.decoder1 = nn.Dense(num_dim, in_units=num_dim//4, weight_initializer=wi1)
			self.layers = [(self.encoder1,self.decoder1),
						(self.encoder2,self.decoder2),
						(self.encoder3,self.decoder3),
						(self.encoder4,self.decoder4)]
				
		def onelayer(self, x, layer):
			xx = F.tanh(layer[0](x))
			return layer[1](xx)
		
		def oneforward(self, x, layer):
			return F.tanh(layer[0](x))
		
		def forward(self, x):
			n_layer = len(self.layers)
			for i in range(n_layer):
				x = F.tanh(self.layers[i][0](x))
			for i in range(n_layer-1):
				x = F.tanh(self.layers[n_layer-i-1][1](x))
			return self.layers[0][1](x)
		
		def manifold(self, x):
			n_layer = len(self.layers)
			for i in range(n_layer-1):
				x = F.tanh(self.layers[i][0](x))
			return self.layers[n_layer-1][0](x)

	from mxnet import autograd
	from mxnet import cpu
	from mxnet.gluon import Trainer
	from mxnet.gluon.loss import L2Loss

	# Stacked AutoEncoder
	model = Model(X.shape[1])
	model.initialize(ctx=[cpu(0),cpu(1),cpu(2),cpu(3)])

	# Select Trainign Algorism
	trainer = Trainer(model.collect_params(),'adam')
	loss_func = L2Loss()

	# Start Pretraining
	print('start pretraining of StackedAE...')
	loss_n = [] # for log

	buffer = nd.array(X.values)
	for layer_id, layer in enumerate(model.layers):
		print('layer %d of %d...'%(layer_id+1,len(model.layers)))
		trainer.set_learning_rate(0.02)
		for epoch in range(1, epochs[layer_id] + 1):
			# random indexs for all datas
			indexs = np.random.permutation(buffer.shape[0])
			for bs in range(0,buffer.shape[0],batch_size):
				be = min(buffer.shape[0],bs+batch_size)
				data = buffer[indexs[bs:be]]
				# forward
				with autograd.record():
					output = model.onelayer(data, layer)
					# make loss
					loss = loss_func(output, data)
					# for log
					loss_n.append(np.mean(loss.asnumpy()))
					del output
				# backward
				loss.backward()
				# step training to one batch
				trainer.step(batch_size, ignore_stale_grad=True)
				del data, loss
			# show log
			print('%d/%d epoch loss=%f...'%(epoch,epochs[layer_id],np.mean(loss_n)))
			loss_n = []
			del bs, be, indexs
		buffer = model.oneforward(buffer, layer)
	del layer, loss_n, buffer

	print('start training of StackedAE...')
	loss_n = []
	buffer = nd.array(X.values)
	trainer.set_learning_rate(0.02)
	for epoch in range(1, epochs[-1] + 1):
		# random indexs for all datas
		indexs = np.random.permutation(buffer.shape[0])
		for bs in range(0,buffer.shape[0],batch_size):
			be = min(buffer.shape[0],bs+batch_size)
			data = buffer[indexs[bs:be]]
			# forward
			with autograd.record():
				output = model(data)
				# make loss
				loss = loss_func(output, data)
				# for log
				loss_n.append(np.mean(loss.asnumpy()))
				del output
			# backward
			loss.backward()
			# step training to one batch
			trainer.step(batch_size, ignore_stale_grad=True)
			del data, loss
		# show log
		print('%d/%d epoch loss=%f...'%(epoch,epochs[-1],np.mean(loss_n)))
		loss_n = []
		del bs, be, indexs
	del trainer, loss_func, loss_n, buffer

	print('making manifold...')
	manifold_X = pd.DataFrame()
	for bs in range(0,X.shape[0],batch_size):
		be = min(X.shape[0],bs + batch_size)
		nx = nd.array(X.iloc[bs:be].values)
		df = pd.DataFrame(model.manifold(nx).asnumpy())
		manifold_X = manifold_X.append(df, ignore_index=True, sort=False)
		del be, df, nx
	del model, bs
	return manifold_X

def get_meta():
	meta_X = pd.DataFrame({
		'soz':(X[X==0]).fillna(1).sum(axis=1),
		'nzm':X[X!=0].mean(axis=1),
		'nzs':X[X!=0].std(axis=1),
		'med':X[X!=0].median(axis=1),
		'max':X[X!=0].max(axis=1),
		'min':X[X!=0].min(axis=1),
		'var':X[X!=0].var(axis=1)})
	return meta_X

if not os.path.exists('manif.csv'):
	manifold_X = get_manifold()
	manifold_X.to_csv('manif.csv', index=False)
else:
	manifold_X = pd.read_csv('manif.csv')

if not os.path.exists('meta.csv'):
	meta_X = get_meta()
	meta_X.to_csv('meta.csv', index=False)
else:
	meta_X = pd.read_csv('meta.csv')

X[X==0.0] = -1.0
random_state = 17
submissions = []

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

print('PCA')
pca = PCA(n_components=num_decompose, random_state=random_state)
pca_X = pd.DataFrame(pca.fit_transform(X))
print('TruncatedSVD')
tsvd = TruncatedSVD(n_components=num_decompose, random_state=random_state)
tsvd_X = pd.DataFrame(tsvd.fit_transform(X))
print('GaussianRandomProjection')
grp = GaussianRandomProjection(n_components=num_decompose, eps=0.1, random_state=random_state)
grp_X = pd.DataFrame(grp.fit_transform(X))
print('SparseRandomProjection')
srp = SparseRandomProjection(n_components=num_decompose, dense_output=True, random_state=random_state)
srp_X = pd.DataFrame(srp.fit_transform(X))

X_drop = X.drop(drop_cols, axis=1)
X_all = pd.concat([X_drop, meta_X, manifold_X, pca_X, tsvd_X, grp_X, srp_X], axis=1, sort=False)

del X_drop, pca, tsvd, grp, srp, pca_X, tsvd_X, grp_X, srp_X

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

print('start training of CatBoost...')

for fold_id, (IDX_train, IDX_test) in enumerate(KFold(n_splits=5, random_state=random_state, shuffle=False).split(Y)):

	X_train = X_all.iloc[IDX_train].values
	X_test = X_all.iloc[IDX_test].values
	Y_train = Y[IDX_train]
	Y_test = Y[IDX_test]

	cb_clf = CatBoostRegressor(iterations=500,
								learning_rate=0.05,
								depth=10,
								eval_metric='RMSE',
								random_seed = fold_id,
								bagging_temperature = 0.2,
								od_type='Iter',
								metric_period = 50,
								od_wait=20)
	cb_clf.fit(X_train, Y_train, eval_set=(X_test, Y_test), cat_features=[], use_best_model=True, verbose=True)
	T = cb_clf.predict(X_all[Y.shape[0]:])
	T = T * (y_max - y_min) + y_min
	submissions.append(np.expm1(T))

del X_all

submissions = np.mean(submissions, axis=0)
result = pd.DataFrame({'ID':ID
					,'target':submissions})
result.to_csv('submission.csv', index=False)

print('end')
