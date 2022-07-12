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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os

print(os.listdir("../input"))
torch.manual_seed(2)
torch.cuda.manual_seed(2)

time_step = 16000
groupby_size = 50
seq_len = 800000
n_groupby = seq_len//groupby_size
n_features = n_groupby//time_step

meta_train = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_train.csv')
phase_train = pd.get_dummies(meta_train.phase).values
phase_train = phase_train.reshape(phase_train.shape[0], 1, phase_train.shape[1])
phase_train = np.repeat(phase_train, time_step, axis = 1)

data_x = torch.load('../input/data-set-output/data_{}t_{}g'.format(time_step, groupby_size))
data_y = torch.tensor(meta_train['target'].values)

class torch_Dataset(Data.Dataset): # 需要继承 data.Dataset
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, index):
        data = (self.x[index], self.y[index])
        return data
    def __len__(self):
        return len(self.y)


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

###############################################################################################################
###  k-fold split
###############################################################################################################
n_splits = 5
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019).split(data_x, data_y))

if torch.cuda.is_available(): 
    data_x = data_x.cuda()
    data_y = data_y.cuda()

###############################################################################################################
###  model train
###############################################################################################################
weight = torch.tensor([1,15]).float()
if torch.cuda.is_available(): 
    weight = weight.cuda()
loss_func = nn.CrossEntropyLoss(weight)

early_stopping_patience = 100
epochs = 3000
model_list = []
for model_index, (train_idx, validate_idx) in enumerate(splits):
    print("Beginning fold {}".format(model_index))
    train_dataset = torch_Dataset(data_x[train_idx], data_y[train_idx])
    train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 512, shuffle = True)
    validate_x = data_x[validate_idx]
    validate_y = data_y[validate_idx]
    
    net = LSTM_seq2one(input_size = validate_x.shape[2])
    if torch.cuda.is_available():  
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

    train_loss_list = []
    validate_loss_list = []
    for epoch in range(epochs):
        # training mode
        net.train()     
        for step, (x, y) in enumerate(train_loader, 1): 
            if torch.cuda.is_available(): 
                x = x.cuda()
                y = y.cuda()
            # 前向传播
            out = net(x)  # (mini_batch, 2)
            loss = loss_func(out, y)  
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluation mode
        net.eval()
        with torch.no_grad():
            validate_loss = loss_func(net(validate_x), validate_y).item()
        validate_loss_list.append(validate_loss)
        train_loss_list.append(loss.item())
        if epoch == 0 or validate_loss < best_validate_loss:            
            best_validate_loss = validate_loss
            best_state_dict = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            print('Best Epoch:', epoch, 'Train_loss:', '%.10f' % loss.item(), 'Validate_loss:', '%.10f' % validate_loss)
        elif epoch - best_epoch > early_stopping_patience: 
            print("Validate_RMSE don't imporved for {} epoch, training stop !".format(early_stopping_patience))
            break
        else:
            print('---- Epoch:', epoch, 'Train_loss:', '%.10f' % loss.item(), 'Validate_loss:', '%.10f' % validate_loss)

    model_list.append({
        'state_dict': best_state_dict,
        'best_epoch': best_epoch,
        'best_validate_loss': best_validate_loss,
        'train_loss_list': train_loss_list,
        'validate_loss_list': validate_loss_list
    })
######################################################################################################
###  model evaluation
######################################################################################################
for model_index in range(len(model_list)):
    train_loss_list = model_list[model_index]['train_loss_list']
    validate_loss_list = model_list[model_index]['validate_loss_list']

    plt.figure(figsize=(15,8))
    plt.plot(train_loss_list, 'g', lw=3, label='train loss') 
    plt.plot(validate_loss_list, 'r', lw=3, label='validate loss')  
    plt.ylabel('loss', fontsize = 20)
    plt.xlabel('Epoch', fontsize = 20)
    plt.legend(loc = 'best', prop={'size': 20})
    plt.title('model {} loss curve'.format(model_index))
    plt.savefig('model {} loss curve.jpeg'.format(model_index))
    plt.show()

LogSoftmax = nn.LogSoftmax(1)
true_label = []
score = []
for model_index, (train_idx, validate_idx) in enumerate(splits):
    print('model {}:'.format(model_index))
    print('-- Best validate loss = {}'.format(model_list[model_index]['best_validate_loss']))
    validate_x = data_x[validate_idx]
    validate_y = data_y[validate_idx]
    net.load_state_dict(model_list[model_index]['state_dict'])
    net.eval()
    true_label += validate_y.cpu().data.numpy().tolist()
    with torch.no_grad():
        score += LogSoftmax(net(validate_x)).cpu().data.numpy()[:,1].tolist()

true_label = np.array(true_label)
score = np.array(score)

fpr, tpr, thresholds = roc_curve(true_label, score, pos_label=1) 
AUC = auc(fpr, tpr)
plt.figure(figsize=(15,8))
plt.plot(fpr, tpr, 'r', lw=3, label='ROC ( AUC = %0.4f )' % AUC)  
plt.ylabel('Sensitivity', fontsize = 20)
plt.xlabel('1 - Specificity', fontsize = 20)
plt.legend(loc = 'lower right', prop={'size': 40})
plt.savefig('ROC.jpeg')
plt.show()

MCC = []
for i in thresholds:
    pred = np.where(score < i, 0, 1)
    MCC.append(matthews_corrcoef(true_label, pred))

max_index = np.argmax(MCC)
threshold = thresholds[max_index]
pred = np.where(score < threshold, 0, 1)
tb = pd.crosstab(index = true_label, columns = pred, rownames = ['實際值'], colnames = ['預測值'])

plt.figure(figsize=(15,8))
loc = np.arange(len(score))
ww_true = true_label == 1
plt.plot(loc, score)
plt.plot(loc[ww_true], score[ww_true], 'rp')
plt.hlines(y = threshold, xmin = loc[0], xmax = loc[-1], color='r', linewidth=2)
plt.savefig('curve.jpeg')
plt.show()

print(tb)
print('thresholds =', threshold)
print('AUC =', AUC)
print('MCC =', MCC[max_index])

output_model = {'model_list': model_list,
                'threshold': threshold,
                'groupby_size': groupby_size,
                'time_step': time_step,
                'input_size': validate_x.shape[2]}
torch.save(output_model, 'model')

