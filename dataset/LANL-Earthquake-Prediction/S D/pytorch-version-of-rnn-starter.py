# BASED ON MICHAEL MAYER'S KERNEL
# I've been trying to convert his kernel to Pytorch + fastai, but can't quite make it work...



# BASIC IDEA OF THE KERNEL

# The data consists of a one dimensional time series x with 600 Mio data points. 
# At test time, we will see a time series of length 150'000 to predict the next earthquake.
# The idea of this kernel is to randomly sample chunks of length 150'000 from x, derive some
# features and use them to update weights of a recurrent neural net with 150'000 / 1000 = 150
# time steps. 

import numpy as np 
import pandas as pd
import os
from tqdm import tqdm

# Fix seeds
from numpy.random import seed
seed(639)
from tensorflow import set_random_seed
set_random_seed(5944)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


from fastai.basics import *
from fastai.callbacks import * 



torch.backends.cudnn.enabled = False



# Import
float_data = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values

# Helper function for the data generator. Extracts mean, standard deviation, and quantiles per time step.
# Can easily be extended. Expects a two dimensional array.
def extract_features(z):
     return np.c_[z.mean(axis=1), 
                  z.min(axis=1),
                  z.max(axis=1),
                  z.std(axis=1)]

# For a given ending position "last_index", we split the last 150'000 values 
# of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.
# From each piece, a set features are extracted. This results in a feature matrix 
# of dimension (150 time steps x features).  
def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_length // 10:]),
                 extract_features(temp[:, -step_length // 100:])]

# Query "create_X" to figure out the number of features
n_features = create_X(float_data[0:150000]).shape[1]
print("Our RNN is based on %i features"% n_features)
    
# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
         
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        for j, row in enumerate(rows):
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
            targets[j] = data[row - 1, 1]
        yield samples, targets
        
        


class EQdataset (Dataset):
    def __init__(self, data, min_index =0, max_index = None, n_steps=150, step_length=1000, is_test = False,dataset_size = 32000):
        
        
        
        if max_index is None:
            max_index = len(data) - 1
        
        
        self.data = data
        self.is_test = is_test
        self.n_steps = n_steps
        self.step_length = step_length 
        
        if is_test:
            self.dataset_size = 1
            self.rows = [150000]
        else:
            self.dataset_size = dataset_size
        
            #last indexes of samples
            self.rows = np.random.randint(min_index + n_steps * step_length, max_index , size=self.dataset_size)

    def __len__(self):
        return self.dataset_size
             
    def __getitem__(self, idx):
        
        row = self.rows[idx]
        sample = create_X(self.data[:, 0], last_index=row, n_steps=self.n_steps, step_length=self.step_length)
        

        if self.is_test:
            target =0
            
        else: 
            target = self.data[row - 1, 1]
        return torch.from_numpy(sample) , target
        
#batch_size = 32

# Position of second (of 16) earthquake. Used to have a clean split
# between train and validation
second_earthquake = 50085877
#float_data[second_earthquake, 1]




# Initialize generators
#train_gen = generator(float_data, batch_size=batch_size) # Use this for better score
#train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)
#valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)

val_ds = EQdataset(float_data,min_index =0, max_index = second_earthquake, dataset_size = 6400)
trn_ds = EQdataset(float_data,min_index =second_earthquake +1, dataset_size = 32000)


# needed until fastai gets updated
val_ds.loss_func = nn.L1Loss()
trn_ds.loss_func = nn.L1Loss()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


databunch = DataBunch.create(trn_ds,val_ds, device=device, bs=32)

# Define model


class EQRNN(nn.Module):
    def __init__(self, input_size = 12,  hidden_size=48,num_layers=1,bidirectional=False, dropout=0.5):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.bidirectional,self.num_layers = bidirectional,num_layers
        if bidirectional: self.num_directions = 2
        else: self.num_directions = 1
           
                 
        self.rnn = nn.GRU(input_size, hidden_size,bidirectional=self.bidirectional,batch_first=True)
        
       
        self.final_layers = nn.Sequential(
            
            nn.Linear(self.num_directions * hidden_size,10),    
            nn.ReLU(),            
            nn.Linear(10,1),   
          
            
        )
        
        
        
    def forward(self,input_seq):
    
        
        #output of shape ( batch_size, seq_len, num_directions * hidden_size)
        #h_n (not needed)
        output, h_n = self.rnn(input_seq)#,h_0)
        
        output = output[:,-1,:]
        
        
        output = self.final_layers(output)
        
        
        return output
        

# fit model

net = EQRNN()
criterion =  nn.L1Loss()
learn = Learner(databunch,net,callback_fns=[ShowGraph], loss_func = criterion)
lr=5e-4
learn.fit(30,lr)


# save the best model
learn.callbacks = [SaveModelCallback(learn,monitor='val_loss',mode='min')]




# Load submission file
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
    
    seg = pd.read_csv('../input/test/' + seg_id + '.csv',dtype={'acoustic_data': np.float32})
    test_ds  = EQdataset(np.array(seg),is_test = True)
    learn.data = DataBunch.create(trn_ds,val_ds,test_ds, device=device)  
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    submission['time_to_failure'][i] =preds

submission.head()

# Save
submission.to_csv('submission.csv')