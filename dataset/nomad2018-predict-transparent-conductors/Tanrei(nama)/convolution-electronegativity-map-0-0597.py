# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os

nth_grid = 9
epochs = 1000
in_dir = "../input/nomad2018-predict-transparent-conductors/"
in_dir_k2 = "../input/make-electronegativity-map-from-xyz-files/"

import pandas as pd
import numpy as np
    
# Read files
df = pd.read_csv(in_dir+"/train.csv")
df_val = pd.read_csv(in_dir+"/test.csv")

df_mean = pd.read_csv(in_dir_k2+"/mean_train.csv")
df_mean_val = pd.read_csv(in_dir_k2+"/mean_test.csv")
df_metrix = pd.read_csv(in_dir_k2+"/metrix_train.csv")
df_metrix_val = pd.read_csv(in_dir_k2+"/metrix_test.csv")

mat_n = df_mean[['mass','min','max','std','mean']].values
mat_n_val = df_mean_val[['mass','min','max','std','mean']].values
dim = (nth_grid+1)*(nth_grid+1)*(nth_grid+1)
mat_m = df_metrix[list(map(str,list(range(1,1+dim))))].values
mat_m_val = df_metrix_val[list(map(str,list(range(1,1+dim))))].values

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
Xt = df[['space_labels','atom_labels','percent_atom_al','percent_atom_ga','percent_atom_in','lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang','lattice_angle_alpha_degree','lattice_angle_beta_degree','lattice_angle_gamma_degree']].values
Xt_val = df_val[['space_labels','atom_labels','percent_atom_al','percent_atom_ga','percent_atom_in','lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang','lattice_angle_alpha_degree','lattice_angle_beta_degree','lattice_angle_gamma_degree']].values

X = np.concatenate((Xt,mat_n,mat_m), axis=1)
X_val = np.concatenate((Xt_val,mat_n_val,mat_m_val), axis=1)
Y = df[['formation_energy_ev_natom','bandgap_energy_ev']].values

from mxnet import nd
X = nd.array(X)
X_val = nd.array(X_val)
Y = nd.array(Y)

# Build MXNet Model
from mxnet import autograd
from mxnet import cpu
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L1Loss, L2Loss
from mxnet import ndarray as F
from mxnet.gluon import Block, nn

class Model(Block):
    def __init__(self, emb1len, emb2len, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv3D(16, kernel_size=(2,2,2), strides=(2,2,2))
            self.pool1 = nn.MaxPool3D(pool_size=(2,2,2), strides=(1,1,1))
            self.conv2 = nn.Conv3D(32, kernel_size=(2,2,2), strides=(2,2,2))
            self.fc = nn.Dense(64)
            self.embed1 = nn.Embedding(emb1len, emb1len)
            self.embed2 = nn.Embedding(emb2len, emb2len)
            self.dense1 = nn.Dense(64)
            self.dense2 = nn.Dense(64)
            self.dense3 = nn.Dense(32)
            self.dense4 = nn.Dense(32)
            self.dense5 = nn.Dense(16)
            self.dense6 = nn.Dense(16)
        
    def forward(self, x):
        bs = x.shape[0]
        xc = x[:,16:].reshape((bs,1,nth_grid+1,nth_grid+1,nth_grid+1))
        xc = F.LeakyReLU(self.conv1(xc), act_type='leaky')
        xc = F.LeakyReLU(self.pool1(xc), act_type='leaky')
        xc = F.LeakyReLU(self.conv2(xc), act_type='leaky')
        xc = F.LeakyReLU(self.fc(xc), act_type='leaky')
        xx = nd.concat(
                self.embed1(x[:,0]) ,
                self.embed2(x[:,1]) ,
                x[:,2:16].reshape((bs,14)) ,
                xc
            )
        xx = F.LeakyReLU(self.dense1(xx), act_type='leaky')
        xx = F.LeakyReLU(self.dense2(xx), act_type='leaky')
        xx = F.LeakyReLU(self.dense3(xx), act_type='leaky')
        xx = F.LeakyReLU(self.dense4(xx), act_type='leaky')
        xx = F.LeakyReLU(self.dense5(xx), act_type='leaky')
        xx = F.LeakyReLU(self.dense6(xx), act_type='leaky')
        return xx

print('start training...')

from multiprocessing import Process

def get_train(tgti):
    loss_func = L2Loss()
    model = Model(len(space_encoder.classes_), len(atoms_encoder.classes_))
    model.initialize(ctx=[cpu(0),cpu(1),cpu(2),cpu(3)])
    trainer = Trainer(model.collect_params(),'adam')
    loss_n = 0 # Log
    batch_size = 15
    indexs = np.random.permutation(X.shape[0])
    for epoch in range(1,epochs+1):
        cur_start = 0
        while cur_start < X.shape[0]:
            # index window
            cur_end = (cur_start + batch_size) if (cur_start + batch_size) < X.shape[0] else X.shape[0]
            data = X[indexs[cur_start:cur_end]]
            label = Y[indexs[cur_start:cur_end]][:,tgti]
            # forward
            with autograd.record():
                output = F.sum(model(data), axis=1)
                loss = loss_func(output, label)
                loss_n = np.mean(loss.asnumpy())
            # backward
            loss.backward()
            trainer.step(batch_size, ignore_stale_grad=True)
            cur_start = cur_end
        # show log
        if epoch % 10 == 0:
            print('%d epoch loss%d=%f...'%(epoch,tgti,loss_n))
    model.save_params('model%d.params'%tgti)
    
jobs = [Process(target=get_train, args=(0,)),Process(target=get_train, args=(1,))]

for j in jobs:
    j.start() 
for j in jobs:
    j.join()

print("Jobs end")

models = [Model(len(space_encoder.classes_), len(atoms_encoder.classes_)),Model(len(space_encoder.classes_), len(atoms_encoder.classes_))]
for tgti in range(2):
    models[tgti].load_params('model%d.params'%tgti, ctx=[cpu(0),cpu(1),cpu(2),cpu(3)])
    os.remove('model%d.params'%tgti)

# Save result
print('make submission...')
with open('submission.csv', 'w') as file:
    file.write('id,formation_energy_ev_natom,bandgap_energy_ev\n')
    # forward
    result1 = F.sum(models[0](X_val), axis=1)
    result2 = F.sum(models[1](X_val), axis=1)
    for i in range(result1.shape[0]):
        pred1 = result1[i].asnumpy()
        pred2 = result2[i].asnumpy()
        pred1 = pred1 if pred1 > 0 else 0
        pred2 = pred2 if pred2 > 0 else 0
        file.write('%d,%f,%f\n' % (i+1,pred1,pred2))