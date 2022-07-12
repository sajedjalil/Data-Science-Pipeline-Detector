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

print(os.listdir("../input"))
torch.manual_seed(1)
torch.cuda.manual_seed(1)

time_step = 100
groupby_size = 100
seq_len = 800000
n_groupby = seq_len//groupby_size
n_features = n_groupby//time_step

meta_train = pd.read_csv('../input/metadata_train.csv')
phase_train = pd.get_dummies(meta_train.phase).values
phase_train = phase_train.reshape(phase_train.shape[0], 1, phase_train.shape[1])
phase_train = np.repeat(phase_train, time_step, axis = 1)

#meta_test = pd.read_csv('../input/metadata_test.csv')
#phase_test = pd.get_dummies(meta_test.phase).values
#phase_test = phase_test.reshape(phase_test.shape[0], 1, phase_test.shape[1])
#phase_test = np.repeat(phase_test, time_step, axis = 1)

iter_size = 50
n_iter = len(meta_train)//iter_size + 1
data_x = np.array([]).reshape(-1, time_step, n_features)
for k in range(0, n_iter):
    print('k =', k)
    start = iter_size*k
    end = iter_size*k + iter_size      
    data_xt = pq.read_pandas('../input/train.parquet', columns = [str(i) for i in meta_train.signal_id[start:end]]).to_pandas()
    data_xt = data_xt.values.reshape(n_groupby, groupby_size, data_xt.shape[-1])
    data_xt = data_xt.std(axis = 1).reshape(n_features, time_step, data_xt.shape[-1])
    data_xt = np.moveaxis(data_xt, [0, -1], [-1, 0])
    data_x = np.concatenate((data_x, data_xt), 0)

#data_x = torch.tensor(np.concatenate((data_x, phase_train), 2)).float()
data_x = data_x.reshape(data_x.shape[0], -1)

ww_normal = np.where(meta_train['target'] == 0)[0]
ww_abnormal = np.where(meta_train['target'] == 1)[0]
data_abnomal = torch.tensor(data_x[ww_abnormal]).float()
data_x = torch.tensor(data_x[ww_normal]).float()

class torch_Dataset(Data.Dataset): # 需要继承 data.Dataset
    def __init__(self, x):
        self.x = x
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=15, num_layers=3):
        super(AutoEncoder, self).__init__()
        
        self.L1 = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.BatchNorm1d(300),
            nn.ReLU()
        )
        
        self.L2 = nn.Sequential(
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        
        self.L3 = nn.Sequential(
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU()
        )
        
        self.L4 = nn.Sequential(
            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )        
        
        self.L5 = nn.Sequential(
            nn.Linear(100, 300),
            nn.BatchNorm1d(300),
            nn.ReLU()
        )
        
        self.L6 = nn.Sequential(
            nn.Linear(300, input_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU()
        )
        #self.lstm1 = nn.LSTM(6, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5, bidirectional=True)  
        #self.line = nn.Linear(2*hidden_size, 2) 
    
    def forward(self, x):
        x =  self.L1(x)
        x =  self.L2(x)
        x =  self.L3(x)
        x =  self.L4(x)
        x =  self.L5(x)
        x =  self.L6(x)
        return x

###############################################################################################################
###  split data to train / validate / test set
###############################################################################################################
train_set_percent = 0.1
train_size = int(len(data_x) * train_set_percent)
validate_size = (len(data_x) - train_size)//2
test_size = len(data_x) - train_size - validate_size

dataset = torch_Dataset(data_x)
train_dataset, validate_dataset, test_dataset = Data.random_split(dataset, [train_size, validate_size, test_size])
train_x = train_dataset[:]
validate_x = validate_dataset[:]
test_x = test_dataset[:]

if torch.cuda.is_available(): 
    train_x = train_x.cuda()
    validate_x = validate_x.cuda()
    test_x = test_x.cuda()
###############################################################################################################
###  model train
###############################################################################################################
net = AutoEncoder(input_size = validate_x.shape[1])
#net = MLP(input_size = validate_x.shape[2])
#weight = torch.tensor([1,15]).float()
if torch.cuda.is_available():  
    net = net.cuda()
    weight = weight.cuda()

train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.001)

early_stopping_patience = 100
epochs = 5000
for epoch in range(epochs):
    # training mode
    net.train()     
    for step, x in enumerate(train_loader, 1): 
        if torch.cuda.is_available(): 
            x = x.cuda()
        # 前向传播
        out = net(x)  # (mini_batch, 2)
        loss = loss_func(out, x)  
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # evaluation mode
    net.eval()        
    validate_loss = loss_func(net(validate_x), validate_x).item()
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

######################################################################################################
###  model evaluation
######################################################################################################
net.load_state_dict(best_state_dict)
net.eval()        
train_loss = loss_func(net(train_x), train_x).item()
validate_loss = loss_func(net(validate_x), validate_x).item()
test_loss = loss_func(net(test_x), test_x).item()
print('training complete ! Select Best Epoch:', best_epoch)
print('train set  loss =', train_loss)
print('validate set  loss =', validate_loss)
print('test set      loss =', test_loss)

n_validate_abnomal = data_abnomal.shape[0]//2
dataset_abnomal = torch_Dataset(data_abnomal)
validate_dataset_abnomal, test_dataset_abnomal = Data.random_split(dataset_abnomal, [n_validate_abnomal, data_abnomal.shape[0] - n_validate_abnomal])

###  validate set
pred_value = net(validate_x)
pred_value_abnormal = net(validate_dataset_abnomal[:])
true_value = [0 for i in range(pred_value.shape[0])] + [1 for i in range(pred_value_abnormal.shape[0])]
score = [loss_func(pred_value[[i]], validate_x[[i]]).item() for i in range(pred_value.shape[0])] 
score += [loss_func(pred_value_abnormal[[i]], validate_dataset_abnomal[:][[i]]).item() for i in range(pred_value_abnormal.shape[0])] 
fpr, tpr, thresholds = roc_curve(true_value, score, pos_label=1) 
AUC = auc(fpr, tpr)
print('validate AUC =', AUC)

plt.figure(figsize=(15,8))
plt.plot(fpr, tpr, 'r', lw=3, label='ROC ( AUC = %0.4f )' % AUC)  
plt.ylabel('Sensitivity', fontsize = 20)
plt.xlabel('1 - Specificity', fontsize = 20)
plt.legend(loc = 'lower right', prop={'size': 40})
plt.savefig('validate_ROC.jpeg')
plt.show()

MCC = []
for i in thresholds:
    pred = np.where(score < i, 0, 1)
    MCC.append(matthews_corrcoef(true_value, pred))

max_index = np.array(MCC).argmax()
threshold = thresholds[max_index]
print('閾值 thresholds 取', threshold)
print('validate MCC 可達到', MCC[max_index])

plt.figure(figsize=(15,8))
loc = np.arange(len(score))
ww_true = np.where(np.array(true_value) == 1)[0]
plt.plot(loc, score)
plt.plot(loc[ww_true], np.array(score)[ww_true], 'rp')
plt.hlines(y = threshold, xmin = loc[0], xmax = loc[-1], color='r', linewidth=2)
plt.savefig('validate_curve.jpeg')
plt.show()

### test set
pred_value = net(test_x)
pred_value_abnormal = net(test_dataset_abnomal[:])
true_value_test = [0 for i in range(pred_value.shape[0])] + [1 for i in range(pred_value_abnormal.shape[0])]
score_test = [loss_func(pred_value[[i]], test_x[[i]]).item() for i in range(pred_value.shape[0])] 
score_test += [loss_func(pred_value_abnormal[[i]], test_dataset_abnomal[:][[i]]).item() for i in range(pred_value_abnormal.shape[0])] 
fpr, tpr, thresholds = roc_curve(true_value_test, score_test, pos_label=1) 
AUC_tset = auc(fpr, tpr)
print('test AUC =', AUC_tset)

plt.figure(figsize=(15,8))
plt.plot(fpr, tpr, 'r', lw=3, label='ROC ( AUC = %0.4f )' % AUC_tset)  
plt.ylabel('Sensitivity', fontsize = 20)
plt.xlabel('1 - Specificity', fontsize = 20)
plt.legend(loc = 'lower right', prop={'size': 40})
plt.savefig('test_ROC.jpeg')
plt.show()

pred_test = np.where(score_test < threshold, 0, 1)
MCC_test = matthews_corrcoef(true_value_test, pred_test)
tb = pd.crosstab(index = np.array(true_value_test), columns = pred_test, rownames = ['實際值'], colnames = ['預測值'])
print(tb)
print('MCC_test =', MCC_test)

plt.figure(figsize=(15,8))
loc = np.arange(len(score_test))
ww_true = np.where(np.array(true_value_test) == 1)[0]
plt.plot(loc, score_test)
plt.plot(loc[ww_true], np.array(score_test)[ww_true], 'rp')
plt.hlines(y = threshold, xmin = loc[0], xmax = loc[-1], color='r', linewidth=2)
plt.savefig('test_curve.jpeg')
plt.show()

output_model = {'state_dict': best_state_dict,
                'threshold': threshold,
                'groupby_size': groupby_size,
                'time_step': time_step,
                'input_size': validate_x.shape[1]}
torch.save(output_model, 'model')