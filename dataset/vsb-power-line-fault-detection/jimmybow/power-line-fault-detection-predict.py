import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import math
import copy
import numpy as np 
import pandas as pd 
import pyarrow.parquet as pq
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import os

print(os.listdir("../input/"))
model = torch.load('../input/power-line-fault-detection-create-model/model', map_location='cpu')
time_step = model['time_step']
groupby_size = model['groupby_size']
seq_len = 800000
n_groupby = seq_len//groupby_size
n_features = n_groupby//time_step

meta_test = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_test.csv')
phase_test = pd.get_dummies(meta_test.phase).values
phase_test = phase_test.reshape(phase_test.shape[0], 1, phase_test.shape[1])
phase_test = np.repeat(phase_test, time_step, axis = 1)

class LSTM_seq2one(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2):
        super(LSTM_seq2one, self).__init__()
        
        self.cnn = nn.Sequential(   # output_len = (input_len - kernel_size)/stride + 1
            nn.Conv1d(input_size, 50, kernel_size = 4, stride = 4), # output_len = 4000
            nn.ReLU(),
            nn.Dropout(0.2),
            #nn.MaxPool1d(4),
            #nn.BatchNorm1d(40),
            nn.Conv1d(50, 50, kernel_size = 4, stride = 4), # output_len = 1000
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.BatchNorm1d(50),
            nn.Conv1d(50, 50, kernel_size = 4, stride = 2, padding = 1), # output_len = 500
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.MaxPool1d(2),
            nn.BatchNorm1d(50),
            nn.Conv1d(50, 50, kernel_size = 4, stride = 2, padding = 1), # output_len = 250
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(50),
            nn.Conv1d(50, 100, kernel_size = 4, stride = 2), # output_len = 124
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.MaxPool1d(2),
            nn.BatchNorm1d(100),
            nn.Conv1d(100, 150, kernel_size = 4, stride = 2), # output_len = 61
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(150),
        )
        
        #self.bn = nn.BatchNorm1d(61)
        self.lstm1 = nn.LSTM(150, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5, bidirectional=True)  
        self.out = nn.Sequential(
            #nn.Dropout(0),
            #nn.BatchNorm1d(2*hidden_size),
            nn.Linear(2*hidden_size, 2) 
        )
      
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x =  self.cnn(x)
        x = x.permute(0, 2, 1)
        #x = self.bn(x)
        x, (h_n, c_n) = self.lstm1(x) 
        # LSTM 輸入資料的格式 (mini_batch, seq_len, input_size)
        # LSTM 輸出的格式:
        # 1. 每個時間步最後一個循環層輸出  (mini_batch, seq_len, hidden_size * num_directions)
        # 2. 每一循環層最後一個時間步輸出 h_n 與 c_n (num_layers * num_directions, batch, hidden_size)
        y = h_n.permute(1,2,0)                     # (mini_batch, hidden_size, num_layers * num_directions)
        y = y[:, :, -2:]                           # (mini_batch, hidden_size, 2)
        y = y.reshape(y.shape[0], -1)              # (mini_batch, 2*hidden_size)
        y = self.out(y)                          # (mini_batch, 2)
        return y

model_list = model['model_list']
net = LSTM_seq2one(input_size = model['input_size'])
net.eval()

if torch.cuda.is_available():  
    net = net.cuda()

threshold = model['threshold']
LogSoftmax = nn.LogSoftmax(1)

iter_size = 100
n_iter = len(meta_test)//iter_size + 1
target = []
for k in range(0, n_iter):
    print('k =', k)
    start = iter_size*k
    end = iter_size*k + iter_size
    pred_x = pq.read_pandas('../input/vsb-power-line-fault-detection/test.parquet', columns = [str(i) for i in  meta_test.signal_id[start:end]]).to_pandas()
    pred_x = pred_x.values.reshape(n_groupby, groupby_size, pred_x.shape[-1])
    pred_x = pred_x.std(axis = 1).reshape(time_step, n_features, pred_x.shape[-1])
    pred_x = np.moveaxis(pred_x, [0, 1, 2], [1, 2, 0])

    pred_x = torch.tensor(np.concatenate((pred_x, phase_test[start:end]), 2)).float()
    if torch.cuda.is_available(): pred_x = pred_x.cuda()
    score_list = []
    for model_index in range(len(model_list)):
        net.load_state_dict(model_list[model_index]['state_dict'])
        with torch.no_grad():
            score = LogSoftmax(net(pred_x).cpu()).data.numpy()[:,1]
        score_list.append(score)
    target += np.where(np.mean(score_list, axis=0) < threshold, 0, 1).tolist()

submission = meta_test[['signal_id']]
submission['target'] = target
submission.to_csv('submission.csv', index=False)