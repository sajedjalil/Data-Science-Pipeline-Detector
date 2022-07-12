#######################################################
# Much of this comes from https://www.kaggle.com/pradeeppathak9/gamma-log-facies-type-prediction
# https://www.crowdanalytix.com/contests/gamma-log-facies-type-prediction
######################################################
import os
os.system('pip install pytorch_toolbelt')
import pandas as pd
import numpy as np
import json
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import time

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sklearn.model_selection import KFold
import gc

from tqdm import tqdm
from itertools import groupby, accumulate
from random import shuffle

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from pytorch_toolbelt import losses as L

ss = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})
train = pd.read_csv('../input/data-without-drift/train_clean.csv')
train['filter'] = 0
test = pd.read_csv('../input/data-without-drift/test_clean.csv')
test['filter'] = 2
ts1 = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

ts1['time2'] = pd.cut(ts1['time'], bins=np.linspace(0.0000, 700., num=14 + 1), labels=list(range(14)), include_lowest=True).astype(int)
ts1['time2'] = ts1.groupby('time2')['time'].rank( )/500000.

np.random.seed(321)
ts1['group'] = pd.cut(ts1['time'], bins=np.linspace(0.0000, 700., num=14*125 + 1), labels=list(range(14*125)), include_lowest=True).astype(int)
np.random.seed(321)

y = ts1.loc[ts1['filter']==0, 'open_channels']
group = ts1.loc[ts1['filter']==0, 'group']
X = ts1.loc[ts1['filter']==0, 'signal']

np.random.seed(321)
skf = GroupKFold(n_splits=5)
splits = [x for x in skf.split(X, y, group)]

use_cols = [col for col in ts1.columns if col not in ['index','filter','group', 'open_channels', 'time', 'time2']]  

# Create numpy array of inputs
for col in use_cols:
    col_mean = ts1[col].mean()
    ts1[col] = ts1[col].fillna(col_mean)
 
val_preds_all = np.zeros((ts1[ts1['filter']==0].shape[0], 11))
test_preds_all = np.zeros((ts1[ts1['filter']==2].shape[0], 11))

groups = ts1.loc[ts1['filter']==0, 'group']
times = ts1.loc[ts1['filter']==0, 'time']

new_splits = []
for sp in splits:
    new_split = []
    new_split.append(np.unique(groups[sp[0]]))
    new_split.append(np.unique(groups[sp[1]]))
    new_splits.append(new_split)
    
trainval = np.array(list(ts1[ts1['filter']==0].groupby('group').apply(lambda x: x[use_cols].values)))
test = np.array(list(ts1[ts1['filter']==2].groupby('group').apply(lambda x: x[use_cols].values)))
trainval_y = np.array(list(ts1[ts1['filter']==0].groupby('group').apply(lambda x: x[['open_channels']].values)))

gc.collect()
# transpose to B x C x L
trainval = trainval.transpose((0,2,1))
test = test.transpose((0,2,1))

trainval_y = trainval_y.reshape(trainval_y.shape[:2])
test_y = np.zeros((test.shape[0], trainval_y.shape[1]))

trainval = torch.Tensor(trainval)
test = torch.Tensor(test)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def __call__(self, score, model):
        if self.best_score is None or \
        (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    
class Seq2SeqRnn(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=.3,
            hidden_layers = [100, 200]):
        
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.output_size=output_size
        
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                           bidirectional=bidirectional, batch_first=True,dropout=0.3)
         # Input Layer
        if hidden_layers and len(hidden_layers):
            first_layer  = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_layers[0])

            # Hidden Layers
            self.hidden_layers = nn.ModuleList(
                [first_layer]+[nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers: nn.init.kaiming_normal_(layer.weight.data)   

            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            # output layers
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 
           
        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_siz, self.input_size)
            self.output_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 

        self.activation_fn = torch.relu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)

        outputs, hidden = self.rnn(x)        

        x = self.dropout(self.activation_fn(outputs))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
            
        x = self.output_layer(x)

        return x


class IonDataset(Dataset):
    """Car dataset."""

    def __init__(self, data, labels, training=True, transform=None, flip=0.5, noise_level=0, class_split=0.0):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.training = training
        self.flip = flip
        self.noise_level = noise_level
        self.class_split = class_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.data[idx]
        labels = self.labels[idx]
        if np.random.rand() < self.class_split:
            data, labels = class_split(data, labels)
        if  np.random.rand() < self.noise_level:
            data = data * torch.FloatTensor(10000).uniform_(1-self.noise_level, 1+self.noise_level)
        if np.random.rand() < self.flip:
            data = torch.flip(data, dims=[1])
            labels = np.flip(labels, axis=0).copy().astype(int)

        return [data, labels.astype(int)]

if not os.path.exists("./models"):
            os.makedirs("./models")
for index, (train_index, val_index ) in enumerate(new_splits[0:], start=0):
    print("Fold : {}".format(index))
    
    batchsize = 16
    train_dataset = IonDataset(trainval[train_index],  trainval_y[train_index], flip=False, noise_level=0.0, class_split=0.0)
    train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=8, pin_memory=True)

    valid_dataset = IonDataset(trainval[val_index],  trainval_y[val_index], flip=False)
    valid_dataloader = DataLoader(valid_dataset, batchsize, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = IonDataset(test,  test_y, flip=False, noise_level=0.0, class_split=0.0)
    test_dataloader = DataLoader(test_dataset, batchsize, shuffle=False, num_workers=8, pin_memory=True)
    test_preds_iter = np.zeros((2000000, 11))
    it = 0
    for it in range(1):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model=Seq2SeqRnn(input_size=trainval.shape[1], seq_len=4000, hidden_size=64, output_size=11, num_layers=2, hidden_layers=[64,64,64],
                         bidirectional=True).to(device)
    
        no_of_epochs = 150
        early_stopping = EarlyStopping(patience=20, is_maximize=True, checkpoint_path="./models/gru_clean_checkpoint_fold_{}_iter_{}.pt".format(index, it))
        criterion = L.FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, max_lr=0.001, epochs=no_of_epochs,
                                                steps_per_epoch=len(train_dataloader))
        avg_train_losses, avg_valid_losses = [], [] 
    
    
        for epoch in range(no_of_epochs):
            start_time = time.time()
    
            print("Epoch : {}".format(epoch))
            print( "learning_rate: {:0.9f}".format(schedular.get_lr()[0]))
            train_losses, valid_losses = [], []
    
            model.train() # prep model for training
            train_preds, train_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
    
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
    
                optimizer.zero_grad()
                predictions = model(x[:, :trainval.shape[1], :])
    
                predictions_ = predictions.view(-1, predictions.shape[-1]) 
                y_ = y.view(-1)
    
                loss = criterion(predictions_, y_)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                schedular.step()
                # record training lossa
                train_losses.append(loss.item())
    
                train_true = torch.cat([train_true, y_], 0)
                train_preds = torch.cat([train_preds, predictions_], 0)

            model.eval() # prep model for evaluation
            val_preds, val_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
            with torch.no_grad():
                for x, y in valid_dataloader:
                    x = x.to(device)
                    y = y.to(device)
    
                    predictions = model(x[:,:trainval.shape[1],:])
                    predictions_ = predictions.view(-1, predictions.shape[-1]) 
                    y_ = y.view(-1)
    
                    loss = criterion(predictions_, y_)
                    valid_losses.append(loss.item())
        
                    val_true = torch.cat([val_true, y_], 0)
                    val_preds = torch.cat([val_preds, predictions_], 0)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            print( "train_loss: {:0.6f}, valid_loss: {:0.6f}".format(train_loss, valid_loss))

            train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1), labels=list(range(11)), average='macro')
    
            val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1), labels=list(range(11)), average='macro')
            print( "train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))
    
            if early_stopping(val_score, model):
                print("Early Stopping...")
                print("Best Val Score: {:0.6f}".format(early_stopping.best_score))
                break
    
            print("--- %s seconds ---" % (time.time() - start_time))
        
        model.load_state_dict(torch.load("./models/gru_clean_checkpoint_fold_{}_iter_{}.pt".format(index, it)))
        with torch.no_grad():
            pred_list = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x[:,:trainval.shape[1],:])
                predictions_ = predictions.view(-1, predictions.shape[-1]) 

                pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())
            test_preds = np.vstack(pred_list)
       
        test_preds_iter += test_preds
        test_preds_all += test_preds
        if not os.path.exists("./predictions/test"):
            os.makedirs("./predictions/test")
        np.save('./predictions/test/gru_clean_fold_{}_iter_{}_raw.npy'.format(index, it), arr=test_preds_iter)
        np.save('./predictions/test/gru_clean_fold_{}_raw.npy'.format(index), arr=test_preds_all)

test_preds_all = test_preds_all/np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("./gru_preds.csv", index=False)