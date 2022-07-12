# -*- coding: utf-8 -*-
import os
import time
import pickle
import numpy as np
import pandas as pd

batch_size = 150
epochs = 3
random_state = 41
n_nodes_hidden = 100
n_nodes_rnn = 64
n_nodes_cnn = 16
expand_terminal = 3

time_fs = time.time()
np.random.seed(random_state)

print('read data:')
df_depth = pd.read_csv('../input/depths.csv', index_col='id')

from mxnet import img as IM

# Image handling
def get_imagedata(fn):
	with open(fn, 'rb') as f:
		return f.read()
def extract_image(data):
	ximg = IM.imdecode(data, flag=0).astype('float32')
	ximg = IM.resize_short(ximg, 100)
	ximg, (x, y, width, height) = IM.center_crop(ximg, (100, 100))
	ximg = ximg.transpose()
	return ximg.asnumpy()

import numpy as np
import random
import glob

# Read Image Datas
X_imgs = {}
files = glob.glob('../input/train/images/*.png')
for file in files:
	dirs = file.split('/')
	fn = dirs[-1]
	id = fn.split('.')[0]
	X_imgs[id] = get_imagedata(file)
# Read Targets
df_train = pd.read_csv('../input/train.csv')
ID = df_train.id.values
target = df_train.rle_mask.fillna('').values
TG = np.array([t.split() for t in target])
df_train['mask_size'] = [len(TG[i]) for i in range(TG.shape[0])]
del target
# Max Word count
max_voc = np.max([np.max([int(s) for s in TG[i]]+[-1]) for i in range(TG.shape[0])])
max_out = np.max([len(TG[i]) for i in range(TG.shape[0])])
print('max_voc = %d, max_out = %d'%(max_voc,max_out))

from mxnet import nd
from mxnet import ndarray as F
from mxnet.gluon import Block, nn, rnn
from mxnet.initializer import Uniform

# Make Model : CNN+RNN encoder-decoder model
class Model(Block):
	def __init__(self, voc, **kwargs):
		super(Model, self).__init__(**kwargs)
		with self.name_scope():
			self.conv1 = nn.Conv2D(channels=n_nodes_cnn, kernel_size=5, strides=2)
			self.conv2 = nn.Conv2D(channels=n_nodes_cnn, kernel_size=5, strides=2)
			self.pool1 = nn.MaxPool2D(pool_size=2)
			self.pool2 = nn.MaxPool2D(pool_size=2)
			self.norm1 = nn.BatchNorm()
			self.norm2 = nn.BatchNorm()
			self.dense1 = nn.Dense(n_nodes_hidden)
			self.dense2 = nn.Dense(voc)
			self.encoder = rnn.SequentialRNNCell()
			with self.encoder.name_scope():
				self.encoder.add(rnn.LSTMCell(n_nodes_rnn))
			self.decoder = rnn.SequentialRNNCell()
			with self.decoder.name_scope():
				self.decoder.add(rnn.LSTMCell(n_nodes_rnn))
	# CNN+encoder
	def forward(self, x, z):
		bs = x.shape[0]
		x = F.relu(self.conv1(x))
		x = self.norm1(self.pool1(x))
		x = F.relu(self.conv2(x))
		x = self.norm2(self.pool2(x))
		x = F.relu(self.dense1(x))
		x = F.concat(x, z)
		status = self.encoder.begin_state(batch_size=bs)
		x, status = self.encoder(x, status)
		return x, status
	# decoder
	def one_word(self, x, status):
		x, status = self.decoder(x, status)
		return x, self.dense2(x), status

from mxnet import autograd
from mxnet import cpu
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

model = Model(max_voc+1)
model.initialize(ctx=[cpu(0),cpu(1),cpu(2),cpu(3)])

trainer = Trainer(model.collect_params(),'adam',{'learning_rate':0.05})
loss_func = SoftmaxCrossEntropyLoss()
loss_n = []

# Training
for epoch in range(1, epochs + 1):
	n_iter = 0
	t_index = []
	for df_g in df_train.groupby('mask_size'):
		irnd = np.random.permutation(len(df_g[1].index))
		t_index.append([df_g[1].index[i] for i in irnd])
		del irnd
	r_index = np.random.permutation(len(t_index))
	for i in range(len(r_index)):
		indexs = t_index[r_index[i]]
		# one minibatch
		for bs in range(0,len(indexs),batch_size):
			# get minibatch data
			be = min(len(indexs),bs+batch_size)
			ids = ID[indexs[bs:be]]
			tgt = TG[indexs[bs:be]]
			maxlen = np.max([len(tgt[b]) for b in range(be-bs)])
			# make input image datas
			buffer = np.zeros((be-bs,1,100,100))
			for b in range(be-bs):
				buffer[b] = extract_image(X_imgs[ids[b]])
			buffer = nd.array(buffer)
			# make depth
			dpt = df_depth.loc[ids].z.values.reshape((be-bs,1))
			dpt = nd.array(dpt)
			# make target sequencial datas
			batch = np.zeros((be-bs)*(maxlen+expand_terminal))
			for w in range(maxlen+expand_terminal):
				for b in range(be-bs):
					batch[(be-bs)*w+b] = max(0,int(tgt[b][w])-1) if w < len(tgt[b]) else max_voc
			batch = nd.array(batch)
			# forward
			with autograd.record():
				result = []
				# CNN+encoder forward
				output, status = model(buffer, dpt)
				# RNN decoder forward
				for w in range(maxlen+expand_terminal):
					output, word, status = model.one_word(output, status)
					result.append(word)
				# make RNN output to sequencial
				result = F.concat(*result, dim=0)
				# make loss
				loss = loss_func(result, batch)
				loss_n.append(np.mean(loss.asnumpy()))
				del output, result
			# backward
			loss.backward()
			trainer.step(be-bs, ignore_stale_grad=True)
			n_iter += be-bs
			del loss, ids, tgt, maxlen, buffer, dpt, batch
		del bs, be, indexs
	print('%d/%d epoch loss=%f...'%(epoch,epochs,np.mean(loss_n)))
	loss_n = []
	del n_iter, t_index, r_index
del trainer, loss_func, loss_n, ID, TG, X_imgs

print('make submisson.')

# Read Image Datas
ID = []
files = glob.glob('../input/test/images/*.png')
for file in files:
	dirs = file.split('/')
	fn = dirs[-1]
	id = fn.split('.')[0]
	ID.append(id)
ID = np.array(ID)
del files, dirs, fn, id

# make result
with open('submission.csv', 'w') as outfile:
	outfile.write('id,rle_mask\n')
	for bs in range(0,ID.shape[0],batch_size):
		# get minibatch data
		be = min(ID.shape[0],bs+batch_size)
		ids = ID[bs:be]
		# make input image datas
		buffer = np.zeros((be-bs,1,100,100))
		for b in range(be-bs):
			Y_img = get_imagedata('../input/test/images/%s.png'%ids[b])
			buffer[b] = extract_image(Y_img)
			del Y_img
		buffer = nd.array(buffer)
		# make depth
		dpt = df_depth.loc[ids].z.values.reshape((be-bs,1))
		dpt = nd.array(dpt)
		# result string
		result = [[]]*(be-bs)
		ends = [False]*(be-bs)
		# CNN+encoder forward
		output, status = model(buffer, dpt)
		# RNN decoder forward
		for w in range(max_out+1):
			if False not in ends:
				break
			output, word, status = model.one_word(output, status)
			for b in range(be-bs):
				if not ends[b]:
					idx = np.argmax(word[b].asnumpy())
					if idx < max_voc:
						result[b].append(str(idx+1))
					else:
						ends[b] = True
		for b in range(be-bs):
			outfile.write('%s,%s\n'%(ids[b],' '.join(result[b])))
		outfile.flush()
		del ids, buffer, result, ends, output, status, word


print('end')