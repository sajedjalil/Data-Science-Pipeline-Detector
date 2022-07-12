import sys
import os
import math
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

#Initialise the random seeds
def random_init(**kwargs):
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True
    
def load_data(df,cv=False,target=False,**kwargs):
    num_samples = len(df)
    sample_size = sum([len(args['categories'][c]) for c in args['categories']]) + len(args['num_feats'])
    dataset = torch.zeros((num_samples,sample_size),dtype=torch.float)
    idx = 0
    for c in args['cat_feats']:
        for i in range(len(args['categories'][c])):
            dataset[np.array(df[c])==args['categories'][c][i],idx] = 1.0
            idx += 1
    for n in args['num_feats']:
        dataset[:,idx] = torch.from_numpy(np.array(df[n]))
        idx += 1
    if target:
        targets = torch.from_numpy(np.array(df['target']))
    else:
        targets = None
    
    if cv == False:
        return dataset, targets

    idx = [i for i in range(num_samples)]
    random.shuffle(idx)
    trainset = dataset[idx[0:int(num_samples*(1-kwargs['cv_percentage']))]]
    traintargets = targets[idx[0:int(num_samples*(1-kwargs['cv_percentage']))]]
    validset = dataset[idx[int(num_samples*(1-kwargs['cv_percentage'])):]]
    validtargets = targets[idx[int(num_samples*(1-kwargs['cv_percentage'])):]]
    return trainset, validset, traintargets, validtargets  

def get_stats(trainset):
    mean = torch.mean(trainset,dim=0)
    std = torch.std(trainset,dim=0)
    for i in range(trainset.shape[1]):
        if ((trainset[:,i]==0) | (trainset[:,i]==1)).all():
            mean[i] = 0.5
            std[i] = 1.0
    return mean, std
    
class Predictor(nn.Module):
    def __init__(self, in_dim, **kwargs):
        super(Predictor, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']
        self.activation = kwargs['activation']
        self.out_dim = 1
        self.mean = kwargs['mean']
        self.std = kwargs['std']

        tmp_feat = self.in_dim
        self.inputnorm = InputNorm(self.mean,self.std)
        for i in range(self.n_layers):
            setattr(self,'hidden'+str(i+1),nn.Linear(tmp_feat,self.hid_dim))
            setattr(self,'activation'+str(i+1),getattr(nn,self.activation)())
            tmp_feat = self.hid_dim
        self.output = nn.Linear(self.hid_dim,self.out_dim)
        
    def forward(self,inputs):
        output = self.inputnorm(inputs)
        for i in range(self.n_layers):
            output = getattr(self,'hidden'+str(i+1))(output)
            output = getattr(self,'activation'+str(i+1))(output)
        output = self.output(output)
        return output.squeeze()

class InputNorm(nn.Module):
    def __init__(self, mean, std):
        super(InputNorm, self).__init__()
        self.register_buffer('mean',mean)
        self.register_buffer('std',std)
    def forward(self,x):
        #out = torch.mul(torch.add(x,-self.mean),1/self.std)
        #return out
        return x
    
def train_model(trainset,traintargets,predictor,optimizer,criterion,**kwargs):
    trainlen = trainset.shape[0]
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    total_loss = 0
    total_backs = 0
    for b in range(nbatches):
        #Data batch
        X = trainset[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
        Y = traintargets[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
        predictions = predictor(X)
        loss = criterion(predictions,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if total_backs == 100:
            total_loss = total_loss*0.99+loss.detach().cpu().numpy()
        else:
            total_loss += loss.detach().cpu().numpy()
            total_backs += 1
    return total_loss/(total_backs+1)

def evaluate_model(testset,predictor,**kwargs):
    testlen = testset.shape[0]
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    predictions = np.zeros((testlen,))
    with torch.no_grad():
        for b in range(nbatches):
            #Data batch
            X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
            estimated = predictor(X)
            predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])] = estimated.cpu().numpy()
    return predictions

class MLPRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,args):
        super(MLPRegressor, self).__init__()
        self.args = args
        
    def fit(self,X,y):
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        self.predictor = Predictor(X.shape[1],**args).to(self.args['device'])
        self.optimizer = torch.optim.Adam(self.predictor.parameters(),lr=self.args['learning_rate'])
        self.criterion = nn.MSELoss(reduction='mean').to(self.args['device'])

        for ep in range(1,self.args['epochs']+1):
            _ = train_model(X,y,self.predictor,self.optimizer,self.criterion,**self.args)
        
    def predict(self,X):
        X = torch.from_numpy(X)
        y = evaluate_model(X,self.predictor,**self.args)
        return y
    
#Global arguments
args = {
    'cv_percentage': 0.1,
    'seed': 0,
    }

# Load data
random_init(**args)

train_data = pd.read_csv('../input/tabular-playground-series-feb-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-feb-2021/test.csv')
args['cat_feats'] = [c for c in np.sort(train_data.columns) if 'cat' in c]
args['num_feats'] = [c for c in np.sort(train_data.columns) if 'cont' in c]
args['categories'] = {c: np.unique(train_data[c]) for c in args['cat_feats']}
testset, _ = load_data(test_data,cv=False,target=False,**args)
trainset, validset, traintargets, validtargets = load_data(train_data,cv=True,target=True,**args)
args['mean'],args['std'] = get_stats(trainset)

val_pred = {}
test_pred = {}

# Bagged SVRs

print('Bagged SVRs...')

# SVR arguments
args = {**args,**{
    'kernel': 'linear',
    'n_estimators': 3,
    'max_samples': 20000,
    'max_features':50
    }}

random_init(**args)

svrs = BaggingRegressor(SVR(kernel=args['kernel'],C=1.0, epsilon=0.2),
                        n_estimators=args['n_estimators'],max_samples=args['max_samples'],max_features=args['max_features'])
svrs.fit(((trainset-args['mean'])/args['std']).numpy(),traintargets.numpy())
val_pred['svrs'] = svrs.predict(((validset-args['mean'])/args['std']).numpy())
test_pred['svrs'] = svrs.predict(((testset-args['mean'])/args['std']).numpy())
rmse = math.sqrt(np.sum(np.square((val_pred['svrs'] - validtargets.numpy())))/len(validtargets))
print('SVRs -> Cross-validation RMSE: {0:.3f}'.format(rmse))

# Bagged decision trees

print('Bagged trees...')

args = {**args,**{
    'max_depth': 50,
    'min_samples_leaf': 100,
    'n_estimators': 100,
    'max_samples': 20000,
    'max_features':50
    }}

random_init(**args)
trees = BaggingRegressor(DecisionTreeRegressor(max_depth=args['max_depth'],min_samples_leaf=args['min_samples_leaf']),
                         n_estimators=args['n_estimators'],max_samples=args['max_samples'],max_features=args['max_features'])
trees.fit(((trainset-args['mean'])/args['std']).numpy(),traintargets.numpy())
val_pred['trees'] = trees.predict(((validset-args['mean'])/args['std']).numpy())
test_pred['trees'] = trees.predict(((testset-args['mean'])/args['std']).numpy())
rmse = math.sqrt(np.sum(np.square((val_pred['trees'] - validtargets.numpy())))/len(validtargets))
print('Trees -> Cross-validation RMSE: {0:.3f}'.format(rmse))

# MLP

print('Bagged MLPs...')

# MLP arguments
args = {**args,**{
    'epochs': 5,
    'batch_size': 32,
    'hidden_size': 128,
    'num_layers': 1,
    'activation': 'Sigmoid',
    'learning_rate': 0.0001,
    'device': torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    }}

random_init(**args)

args = {**args,**{
    'n_estimators': 5,
    'max_samples': 100000,
    'max_features': 50
    }}
mlps = BaggingRegressor(MLPRegressor(args),
                         n_estimators=args['n_estimators'],max_samples=args['max_samples'],max_features=args['max_features'])
_ = mlps.fit(((trainset-args['mean'])/args['std']).numpy(),traintargets.numpy())
val_pred['mlps'] = mlps.predict(((validset-args['mean'])/args['std']).numpy())
test_pred['mlps'] = mlps.predict(((testset-args['mean'])/args['std']).numpy())
rmse = math.sqrt(np.sum(np.square((val_pred['mlps'] - validtargets.numpy())))/len(validtargets))
print('MLPs -> Cross-validation RMSE: {0:.3f}'.format(rmse))

#System combination
val_pred = (val_pred['svrs'] + val_pred['mlps'] + val_pred['trees']) / 3
test_pred = (test_pred['svrs'] + test_pred['mlps'] + test_pred['trees']) / 3
rmse = math.sqrt(np.sum(np.square((val_pred - validtargets.numpy())))/len(validtargets))
print('\nFinal combination -> Cross-validation RMSE: {0:.3f}'.format(rmse))

out_df = pd.DataFrame(data={'id':test_data['id'],'target':test_pred}).set_index('id',drop=True)
out_df.to_csv('submission.csv')
