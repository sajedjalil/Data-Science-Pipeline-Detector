#Train a model from scratch in PyTorch and run evaluation
import tensorflow as tf
import sys
import io
import os
import subprocess
import math
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from collections import OrderedDict
import random
import torch
import torch.nn as nn
from PIL import Image

#Initialise all random numbers for reproducibility
def init_random(**kwargs):
    random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True

#Read the images and preprocess
def read_tfrecords(filepaths,train=False,**kwargs):
    sh = np.sum([1 for path in filepaths for record in tf.compat.v1.io.tf_record_iterator(path)])
    images = torch.zeros((sh,1,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    classes = {cl:torch.zeros((sh,),dtype=int) for cl in kwargs['vocab']}
    ids = list()
    idx = 0
    for ii,path in enumerate(filepaths):
        print('File {0:d} of {1:d}: {2:s}'.format(ii+1,len(filepaths),path))
        for record in tf.compat.v1.io.tf_record_iterator(path):
            example = tf.train.Example()
            example.ParseFromString(record)

            img = example.features.feature['image'].bytes_list.value[0]
            img = Image.open(io.BytesIO(img))
            img = img.resize((kwargs['ysize'],kwargs['xsize']))
            img = np.expand_dims(np.asarray(img),axis=0)
            images[idx,:,:,:] = torch.from_numpy(img)
            for cl in kwargs['vocab']:
                if cl in example.features.feature:
                    label = example.features.feature[cl].int64_list.value[0]
                    classes[cl][idx] = label
            idx += 1
            iid = example.features.feature['StudyInstanceUID'].bytes_list.value[0].decode('ascii')
            ids.append(iid)
    if train:
        idx = [i for i in range(images.shape[0])]
        random.shuffle(idx)
        images = images[idx]
        classes = {cl:classes[cl][idx] for cl in kwargs['vocab']}
    return images, classes, ids

class SimpleCNN(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleCNN, self).__init__()
        #Arguments
        self.xsize = kwargs['xsize']
        self.ysize = kwargs['ysize']
        self.num_blocks = kwargs['num_blocks']
        self.channels = kwargs['channels']
        self.input_channels = (kwargs['input_channels'] if 'input_channels' in kwargs else 1)
        self.reduce_size = (kwargs['reduce_size'] if 'reduce_size' in kwargs else False)
        self.dropout = kwargs['dropout']
        self.embedding_size = kwargs['embedding_size']
        self.vocab = kwargs['vocab']
        self.num_classes = {v:len(self.vocab[v]) for v in self.vocab}
        self.mean = kwargs['mean']
        self.std = kwargs['std']

        #Gaussian normalise the input
        self.inputnorm = InputNorm(self.mean,self.std)
        tmp_xsize = self.xsize
        tmp_ysize = self.ysize
        #Convolutional blocks
        for i in range(1,self.num_blocks+1):
            if self.reduce_size or i ==1:
                setattr(self,'convblock'+str(i),ConvBlock((self.channels if i>1 else self.input_channels), self.channels, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, dropout=self.dropout, residual=False))
                tmp_xsize = int(tmp_xsize/2)
                tmp_ysize = int(tmp_ysize/2)
            setattr(self,'convblock'+str(i)+'residual',ConvBlock(self.channels, self.channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=int(self.channels/2), dropout=self.dropout, residual=True))
            
        #Flatten the output
        self.flatten = nn.Flatten()
        #Reduce to embedding layer
        self.linear = nn.Linear(int(tmp_xsize*tmp_ysize*self.channels), self.embedding_size, bias=False)
        #Batch normalise
        self.batchnorm = nn.BatchNorm1d(self.embedding_size, momentum=0.9)
        #L2 normalise
        self.l2norm = L2Norm()
        #Classification layer and softmax
        self.output = MultiLinear(self.embedding_size, self.num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, return_embedding = False):
        out = self.inputnorm(x)
        for i in range(1,self.num_blocks+1):
            if self.reduce_size or i==1:
                conv = getattr(self,'convblock'+str(i))
                out = conv(out)
            conv = getattr(self,'convblock'+str(i)+'residual')
            out = conv(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.batchnorm(out)
        out = self.l2norm(out)
        if not return_embedding:
            out = self.output(out)
            out = {v:self.softmax(out[v]) for v in out}
        return out
    
class MultiLinear(nn.Module):
    def __init__(self, input_size, output_sizes):
        super(MultiLinear, self).__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        for v in self.output_sizes:
            setattr(self,'layer'+v,nn.Linear(self.input_size, self.output_sizes[v]))
        
    def forward(self, inputs):
        outputs = {v:getattr(self,'layer'+v)(inputs) for v in self.output_sizes}
        return outputs
    
#Performs gaussian normalisation of an input with mean and standard deviation
class InputNorm(nn.Module):
    def __init__(self, mean, std):
        super(InputNorm, self).__init__()
        self.mean = mean
        self.std = std
    def forward(self,x):
        out = torch.mul(torch.add(x,-self.mean),1/self.std)
        return out

#Residual convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, dropout=0.2, residual=False):
        super(ConvBlock, self).__init__()
        #2D convolution
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        #Batch normalisation
        self.bn = nn.BatchNorm2d(out_c, momentum=0.9)
        #Activation
        self.prelu = nn.PReLU(out_c)
        #Dropout
        self.dropout = nn.Dropout3d(p=dropout)
        self.residual = residual
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.dropout(out)
        #Residual connection
        if self.residual:
            out = out + x
        return out

#Do L2 normalisation of embedding vectors
class L2Norm(nn.Module):
    def __init__(self, axis=1):
        super(L2Norm, self).__init__()
        self.axis = axis
    def forward(self,x):
        norm = torch.norm(x, 2, self.axis, True)
        output = torch.div(x, norm)
        return output
    
def train_model(trainset,trainlabels,model,optimizer,criterion,**kwargs):
    trainlen = trainset.shape[0]
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
        model = model.train()
        for b in range(nbatches):
            #Obtain batch
            X = trainset[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().float()
            X = X.to(kwargs['device'])
            Y = {v:trainlabels[v][b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device']) for v in trainlabels}
            #Propagate
            posteriors = model(X)
            #Backpropagate
            loss = np.sum([criterion[v](posteriors[v],Y[v])*kwargs['weights'][v] for v in Y])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Track loss
            if total_backs == 100:
                total_loss = total_loss*0.99+loss.detach().cpu().numpy()
            else:
                total_loss += loss.detach().cpu().numpy()
                total_backs += 1
            pbar.set_description(f'Training epoch. Loss {total_loss/(total_backs+1):.2f}')
            pbar.update()
    return total_loss/(total_backs+1)

#Get posteriors for a test set
def evaluate_model(testset,model,**kwargs):
    testlen = testset.shape[0]
    predictions = {v:np.zeros((testlen,len(kwargs['vocab'][v]))) for v in kwargs['vocab']}
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    with torch.no_grad():
        model = model.eval()
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            for b in range(nbatches):
                #Obtain batch
                X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
                #Propagate
                posteriors = model(X)
                for cl in kwargs['vocab']:
                    predictions[cl][b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size']),:] = posteriors[cl].detach().cpu().numpy()
                pbar.set_description('Testing')
                pbar.update()
    return predictions

#Compute the class-based ROC metric
def compute_roc_auc(scores,labels):
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

#Arguments
args = {
    'cv_percentage': 0.1,
    'xsize': 256,
    'ysize': 256,
    'num_blocks': 5,
    'reduce_size': True,
    'channels': 48,
    'input_channels': 1,
    'dropout': 0.0,
    'embedding_size': 256,
    'epochs': 10,
    'batch_size': 96,
    'learning_rate': 0.001,
    'seed': 0,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
}

#Initialise RNGs
init_random(**args)

print('Loading data...')
df = pd.read_csv('../input/ranzcr-clip-catheter-line-classification/train.csv')
classes = [col for col in df.columns if ('ETT' in col or 'NGT' in col or 'CVC' in col or 'Swan Ganz' in col)]
args['vocab'] = OrderedDict({cl:OrderedDict({0:0,1:1}) for cl in classes})
train_files = np.sort(glob.glob("../input/ranzcr-clip-catheter-line-classification/train_tfrecords/*.tfrec"))
val_files = np.sort(train_files[math.ceil(len(train_files)*(1-args['cv_percentage'])):])
train_files = np.sort(train_files[:math.ceil(len(train_files)*(1-args['cv_percentage']))])
test_files = glob.glob("../input/ranzcr-clip-catheter-line-classification/test_tfrecords/*.tfrec")

#Read TF data
train_data, train_targets, _ = read_tfrecords(train_files,True,**args)
val_data, val_targets, _ = read_tfrecords(val_files,False,**args)
test_data, _, test_ids = read_tfrecords(test_files,False,**args)

#Mean and standard deviation for input normalisation
args['mean'] = torch.mean(train_data.float())
args['std'] = torch.std(train_data.float())

#Build model, optimizer and weighted criterion
init_random(**args)
model = SimpleCNN(**args).to(args['device'])
optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
priors = {cl:torch.Tensor([len(np.where(train_targets[cl].numpy()==t)[0])/train_targets[cl].shape[0] for t in np.unique(train_targets[cl])]) for cl in args['vocab']}
criterion = {cl:nn.NLLLoss(weight = 1 / (priors[cl] / torch.max(priors[cl])),reduction='mean').to(args['device']) for cl in args['vocab']}

args['weights'] = {cl:1.0 for cl in args['vocab']}

print('Training...')
best_auc = 0.0
for ep in range(1,args['epochs']+1):
    #Train an epoch
    loss = train_model(train_data,train_targets,model,optimizer,criterion,**args)
    #Get the posteriors for the validation set
    val_preds = evaluate_model(val_data,model,**args)
    val_preds = {cl:np.exp(val_preds[cl])[:,1] for cl in args['vocab']}
    #Compute AUC
    auc = {cl:compute_roc_auc(val_preds[cl],val_targets[cl]) for cl in args['vocab']}
    global_auc = np.mean([auc[cl] for cl in auc])
    print('Epoch {0:d}, loss: {1:.2f}, AUC: {2:.2f}'.format(ep,loss,global_auc))
    if global_auc >= best_auc:
        #Get the posteriors for the test set
        test_preds = evaluate_model(test_data,model,**args)
        predictions = {cl:np.exp(test_preds[cl])[:,1] for cl in args['vocab']}

#Write output
df_out = pd.DataFrame({**{'StudyInstanceUID': test_ids}, **predictions})
df_out.to_csv('/kaggle/working/submission.csv',index=False)