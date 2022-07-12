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

#Initialise the random seeds
def random_init(**kwargs):
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True
    
def load_data(df,cv=False,target=False,**kwargs):
    num_samples = len(df)
    sample_size = len(args['cat_feats']) + len(args['num_feats'])
    dataset = torch.zeros((num_samples,sample_size),dtype=torch.float)
    idx = 0
    for c in args['cat_feats']:
        dataset[:,idx] = torch.from_numpy(np.array([args['categories'][c][v] if v in args['categories'][c] else len(args['categories'][c]) for v in df[c]]))
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
    
class Predictor(nn.Module):
    def __init__(self, **kwargs):
        super(Predictor, self).__init__()
        self.cat_feats = kwargs['cat_feats']
        self.num_feats = kwargs['num_feats']
        self.categories = kwargs['categories']
        self.hid_dim = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']
        self.activation = kwargs['activation']
        self.out_dim = 2

        self.in_dim = 0
        for c in self.cat_feats:
            setattr(self,'encoder_'+c,nn.Embedding(len(self.categories[c])+1,int(math.ceil(math.log(len(self.categories[c])+1)/math.log(2.0)))))
            self.in_dim += int(math.ceil(math.log(len(self.categories[c])+1)/math.log(2.0)))
        self.in_dim += len(self.num_feats)
        tmp_feat = self.in_dim
        for i in range(self.n_layers):
            setattr(self,'hidden'+str(i+1),nn.Linear(tmp_feat,self.hid_dim))
            setattr(self,'activation'+str(i+1),getattr(nn,self.activation)())
            tmp_feat = self.hid_dim
        self.output = nn.Linear(self.hid_dim,self.out_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,inputs):
        output = inputs.new_zeros((inputs.shape[0],self.in_dim))
        idx = 0
        for i,c in enumerate(self.cat_feats):
            output[:,idx:idx+int(math.ceil(math.log(len(self.categories[c])+1)/math.log(2.0)))] = getattr(self,'encoder_'+c)(inputs[:,i].long())
            idx += int(math.ceil(math.log(len(self.categories[c])+1)/math.log(2.0)))
        output[:,idx:] = inputs[:,-len(self.num_feats):]
        for i in range(self.n_layers):
            output = getattr(self,'hidden'+str(i+1))(output)
            output = getattr(self,'activation'+str(i+1))(output)
        output = self.output(output)
        output = self.softmax(output)
        return output
    
def train_model(trainset,traintargets,predictor,optimizer,criterion,**kwargs):
    trainlen = trainset.shape[0]
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    total_loss = 0
    total_backs = 0
    for b in range(nbatches):
        #Data batch
        X = trainset[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
        Y = traintargets[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
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
    predictions = np.zeros((testlen,2))
    with torch.no_grad():
        for b in range(nbatches):
            #Data batch
            X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
            estimated = predictor(X)
            predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])] = estimated.cpu().numpy()
    return predictions

#Compute the class-based ROC metric
def compute_roc_auc(scores,labels):
    if scores.ndim == 2:
        scores = scores[:,1]
    pos_scores = scores[np.where(labels==1)]
    neg_scores = scores[np.where(labels==0)]
    thresholds = np.arange(np.max(scores),np.min(scores)-0.01,-0.001)
    fpr = []
    tpr = []
    for th in thresholds:
        fpr.append(len(np.where(neg_scores>=th)[0])/len(neg_scores))
        tpr.append(len(np.where(pos_scores>=th)[0])/len(pos_scores))
    roc = pd.DataFrame({'thresholds':thresholds,'tp':tpr,'fp':fpr})
    auc = np.trapz(roc['tp'],x=roc['fp'])
    return auc
    
args = {
    'cv_percentage': 0.1,
    'epochs': 5,
    'batch_size': 32,
    'hidden_size': 64,
    'num_layers': 8,
    'activation': 'ReLU',
    'learning_rate': 0.0001,
    'seed': 0,
    'device': torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    }

random_init(**args)

train_data = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
args['cat_feats'] = [c for c in np.sort(train_data.columns) if 'cat' in c]
args['num_feats'] = [c for c in np.sort(train_data.columns) if 'cont' in c]
args['categories'] = {c: {a:i for i,a in enumerate(np.unique(train_data[c]))} for c in args['cat_feats']}
trainset, validset, traintargets, validtargets = load_data(train_data,cv=True,target=True,**args)
args['num_mean'] = torch.mean(trainset[:,-len(args['num_feats']):],dim=0)
args['num_std'] = torch.std(trainset[:,-len(args['num_feats']):],dim=0)
testset, _ = load_data(test_data,cv=False,target=False,**args)

trainset[:,-len(args['num_feats']):] -= args['num_mean']
trainset[:,-len(args['num_feats']):] /= args['num_std']
validset[:,-len(args['num_feats']):] -= args['num_mean']
validset[:,-len(args['num_feats']):] /= args['num_std']
testset[:,-len(args['num_feats']):] -= args['num_mean']
testset[:,-len(args['num_feats']):] /= args['num_std']

predictor = Predictor(**args).to(args['device'])
optimizer = torch.optim.Adam(predictor.parameters(),lr=args['learning_rate'])
criterion = nn.NLLLoss(reduction='mean').to(args['device'])

for ep in range(1,args['epochs']+1):
    loss = train_model(trainset,traintargets,predictor,optimizer,criterion,**args)
    tmp_val_pred = evaluate_model(validset,predictor,**args)
    auc = compute_roc_auc(tmp_val_pred,validtargets.numpy())
    print('Epoch {0:d} of {1:d}. Training loss: {2:.3f}. Cross-validation ROC: {3:.3f}'.format(ep,args['epochs'],loss,auc))
    
test_pred = np.exp(evaluate_model(testset,predictor,**args))[:,1]

out_df = pd.DataFrame(data={'id':test_data['id'],'target':test_pred}).set_index('id',drop=True)
out_df.to_csv('submission.csv')