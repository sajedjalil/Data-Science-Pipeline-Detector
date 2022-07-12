# -*- coding: utf-8 -*-
from collections import Counter
import pandas as pd
import numpy as np

# Read files
df = pd.read_csv("../input/train.csv")
df_val = pd.read_csv("../input/test.csv")

# Making label
from sklearn.preprocessing import LabelEncoder
space_encoder = LabelEncoder()
space_encoder.fit(np.hstack([df.spacegroup, df_val.spacegroup]))
space_labels = space_encoder.transform(df.spacegroup)
space_labels_val = space_encoder.transform(df_val.spacegroup)
atoms_encoder = LabelEncoder()
atoms_encoder.fit(np.hstack([df.number_of_total_atoms, df_val.number_of_total_atoms]))
atom_labels = atoms_encoder.transform(df.number_of_total_atoms)
atom_labels_val = atoms_encoder.transform(df_val.number_of_total_atoms)
df['space_labels'] = space_labels
df_val['space_labels'] = space_labels_val
df['atom_labels'] = atom_labels
df_val['atom_labels'] = atom_labels_val

# Make MXNet NDarray
X = df[['space_labels','atom_labels','percent_atom_al','percent_atom_ga','percent_atom_in','lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang','lattice_angle_alpha_degree','lattice_angle_beta_degree','lattice_angle_gamma_degree']].values
X_val = df_val[['space_labels','atom_labels','percent_atom_al','percent_atom_ga','percent_atom_in','lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang','lattice_angle_alpha_degree','lattice_angle_beta_degree','lattice_angle_gamma_degree']].values
Y = df[['formation_energy_ev_natom','bandgap_energy_ev']].values
from mxnet import nd
X = nd.array(X)
X_val = nd.array(X_val)
Y = nd.array(Y)

# Build MXNet Model
from mxnet import autograd
from mxnet import cpu
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L2Loss
from mxnet import ndarray as F
from mxnet.gluon import Block, nn

class Model(Block):
	def __init__(self, emb1len, emb2len, **kwargs):
		super(Model, self).__init__(**kwargs)
		with self.name_scope():
			self.embed1 = nn.Embedding(emb1len, emb1len)
			self.embed2 = nn.Embedding(emb2len, emb2len)
			self.dense1 = nn.Dense(32)
			self.dense2 = nn.Dense(32)
			self.dense3 = nn.Dense(16)
			self.dense4 = nn.Dense(16)
			self.dense5 = nn.Dense(2)

	def forward(self, x):
		bs = x.shape[0]
		xx = nd.concat(
				self.embed1(x[:,0]) ,
				self.embed2(x[:,1]) ,
				x[:,2].reshape((bs,1)) ,
				x[:,3].reshape((bs,1)) ,
				x[:,4].reshape((bs,1)) ,
				x[:,5].reshape((bs,1)) ,
				x[:,6].reshape((bs,1)) ,
				x[:,7].reshape((bs,1)) ,
				x[:,8].reshape((bs,1)) ,
				x[:,9].reshape((bs,1)) ,
				x[:,10].reshape((bs,1))
			)
		xx = F.relu(self.dense1(xx))
		xx = F.relu(self.dense2(xx))
		xx = F.relu(self.dense3(xx))
		xx = F.relu(self.dense4(xx))
		return self.dense5(xx)

model = Model(len(space_encoder.classes_), len(atoms_encoder.classes_))
model.initialize(ctx=[cpu(0),cpu(1),cpu(2),cpu(3)])

# Training
trainer = Trainer(model.collect_params(),'adam')
loss_func = L2Loss()
print('start training...')
batch_size = 15
epochs = 500
loss_n = [] # Log
for epoch in range(1, epochs + 1):
	# mini batch index
	indexs = np.random.permutation(X.shape[0])
	cur_start = 0
	while cur_start < X.shape[0]:
		# index window
		cur_end = (cur_start + batch_size) if (cur_start + batch_size) < X.shape[0] else X.shape[0]
		data = X[indexs[cur_start:cur_end]]
		label = Y[indexs[cur_start:cur_end]]
		# forward
		with autograd.record():
			output = model(data)
			loss = loss_func(output, label)
			loss_n.append(np.mean(loss.asnumpy()))
		# backward
		loss.backward()
		trainer.step(batch_size, ignore_stale_grad=True)
		cur_start = cur_end
	# show log
	ll = np.mean(loss_n)
	print('%d epoch loss=%f...'%(epoch,ll))
	loss_n = []

# Save result
print('make submission...')
result = F.relu(model(X_val)).asnumpy()
df = pd.DataFrame({'id': range(1,len(result)+1),'formation_energy_ev_natom': result[:,0],'bandgap_energy_ev': result[:,1]})
df.to_csv('submission.csv', index=False)