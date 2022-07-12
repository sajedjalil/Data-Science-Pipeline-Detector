import sys
import os
import subprocess
import math
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import soundfile
from librosa import resample
import scipy.signal
from tqdm import tqdm
import librosa
from PIL import Image

#Clone and import CNNWordReco repository
if ~os.path.isdir('CNNWordReco'):
    subprocess.call(['git', 'clone', 'https://github.com/saztorralba/CNNWordReco'])
if 'CNNWordReco' not in sys.path:
    sys.path.append('CNNWordReco')
from utils.cnn_func import train_model, validate_model
from models.SimpleCNN import SimpleCNN

def init_random(**kwargs):
    random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True
    
def gen_logfft(file,fs=8000,n_fft=10,normalise=False,hop_length=10):
    epsilon=1e-20
    n_fft=int(n_fft*fs/1000)
    hop_length=int(hop_length*fs/1000)
    #Read file
    audio,f=soundfile.read(file)
    if f!=fs:
        audio=resample(audio,f,fs)
    #Normalise input energy
    if normalise:
        audio=0.5*audio/np.max(np.absolute(audio))
    #High-pass filter
    audio=scipy.signal.convolve(audio,np.array([1,-0.98]),mode='same',method='fft')
    duration = (len(audio)-1)/fs
    n_samples = int((duration)/0.5)+((duration%0.5)>0.25)
    #Compute STFT
    DATA = []
    stft=np.abs(librosa.core.stft(y=audio,n_fft=n_fft,hop_length=hop_length))
    stft=np.transpose(np.log(stft+epsilon))
    for i in range(n_samples):
        start = i * 0.5 * fs / hop_length
        if i == n_samples-1:
            finish = stft.shape[0]
        else:
            finish = (i+1) * 0.5 * fs / hop_length
        DATA.append(stft[int(start):int(finish),:])
    return DATA

def feat2img(DATA,xsize=40,ysize=40):
    out = np.zeros((len(DATA),ysize,xsize))
    for i in range(len(DATA)):
        #Reorg dimensions
        DATA[i] = np.flipud(np.transpose(DATA[i]))
        #Clamp and normalise to [0,1]
        DATA[i] = (np.maximum(-15,np.minimum(0,DATA[i]))+15)/15
        #Convert to PIL
        im = Image.fromarray(np.uint8(DATA[i]*255))
        #Resize
        im = im.resize((ysize,xsize))
        #Back to numpy
        out[i,:,:] = np.array(im)
    return out
    
def load_data(files, targets=None, randomize=False, **kwargs):
    n_samples = 0
    indexes = []
    idx=0
    for i,file in enumerate(files):
        audio,fs = soundfile.read(file)
        if len(audio) == 0:
            indexes.append([])
            continue
        duration = (len(audio)-1)/kwargs['sampling']
        duration = int(duration/0.5)+((duration%0.5)>0.25)
        n_samples += duration
        indexes.append(list(range(idx,idx+duration)))
        idx = idx + duration
    dataset = torch.zeros((n_samples,1,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    labels = torch.zeros((n_samples),dtype=torch.uint8)
    idx = 0
    for i in tqdm(range(len(files)),disable=(kwargs['verbose']<2)):
        if len(indexes[i]) == 0:
            continue
        feats = feat2img(gen_logfft(files[i],fs=kwargs['sampling'],normalise=True),kwargs['ysize'],kwargs['xsize'])
        dataset[idx:idx+feats.shape[0],0,:,:] = torch.from_numpy(feats)
        if targets is not None:
            labels[idx:idx+feats.shape[0]] = kwargs['vocab'][targets[i]]
        idx = idx + feats.shape[0]
    assert n_samples == idx
        
    if randomize:
        idx = [i for i in range(n_samples)]
        random.shuffle(idx)
        dataset = dataset[idx]
        labels = labels[idx]
        indexes = []
    return dataset, labels, indexes

def evaluate_model(testset,model,**kwargs):
    testlen = testset.shape[0]
    predictions = np.zeros((testlen,len(kwargs['vocab'])))
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    with torch.no_grad():
        model = model.eval()
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            for b in range(nbatches):
                #Obtain batch
                X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
                #Propagate
                posteriors = model(X)
                predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size']),:] = posteriors.detach().cpu().numpy()
                pbar.set_description('Testing')
                pbar.update()
    return predictions

def compute_mapk(labels,predictions,k=3):
    estimated = np.argsort(predictions,axis=1)[:,::-1]
    apk = sum((estimated[:,0] == labels))
    for i in range(1,k):
        apk += sum((estimated[:,i] == labels)) / (i+1)
    mapk = apk/len(labels)
    return mapk
    
#Arguments
args = {
    'cv_percentage': 0.1,
    'xsize': 40,
    'ysize': 40,
    'num_blocks': 5,
    'channels': 32,
    'dropout': 0.0,
    'embedding_size': 128,
    'epochs': 10,
    'batch_size': 512,
    'learning_rate': 0.01,
    'seed': 0,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
    'augment': False,
    'sampling': 44100
}

#Initialise the random seeds
init_random(**args)

#Read data and store in dataframe
train_data = pd.read_csv('/kaggle/input/freesound-audio-tagging/train.csv')
train_files = np.array(['/kaggle/input/freesound-audio-tagging/audio_train/'+v for v in train_data.fname.values])
train_targets = np.array(train_data.label.values)
args['vocab'] = OrderedDict({t:i for i,t in enumerate(np.unique(train_targets))})
args['inv_vocab'] = {i:v for i,v in enumerate(args['vocab'])}
test_files = glob.glob('/kaggle/input/freesound-audio-tagging/audio_test/*.wav')

probs = np.array([random.random() for i in range(len(train_files))])
valid_files = train_files[probs>(1-args['cv_percentage'])]
valid_targets = train_targets[probs>(1-args['cv_percentage'])]
train_files = train_files[probs<=(1-args['cv_percentage'])]
train_targets = train_targets[probs<=(1-args['cv_percentage'])]

print('Loading training data: {0:d} files'.format(len(train_files)))
trainset, trainlabels, _ = load_data(train_files, train_targets, randomize=True,**args)
print('Loading validation data: {0:d} files'.format(len(valid_files)))
validset, validlabels, validindexes = load_data(valid_files, valid_targets, randomize=False,**args)
print('Loading evaluation data: {0:d} files'.format(len(test_files)))
testset, _, testindexes = load_data(test_files, None, randomize=False, **args)
args['mean'] = torch.mean(trainset.float())
args['std'] = torch.std(trainset.float())
#Model, optimiser and criterion
model = SimpleCNN(**args).to(args['device'])
optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
criterion = nn.NLLLoss(reduction='mean').to(args['device'])
best_mapk = 0
predictions = []
for ep in range(1,args['epochs']+1):
    #Do backpropgation and validation epochs
    loss = train_model(trainset,trainlabels,model,optimizer,criterion,**args)
    val_preds = evaluate_model(validset,model,**args)
    val_preds = np.array([np.sum(val_preds[i,:],axis=0)/len(i) for i in validindexes])
    mapk = compute_mapk(np.array([args['vocab'][t] for t in valid_targets]),val_preds)
    print('Epoch {0:d} of {1:d}. Training loss: {2:.2f}, validation map@3: {3:.4f}'.format(ep,args['epochs'],loss,mapk))
    if mapk >= best_mapk:
        test_preds = evaluate_model(testset,model,**args)
        test_preds = np.array([np.sum(test_preds[i,:],axis=0)/len(i) if len(i)>0 else np.sum(test_preds[i,:],axis=0) for i in testindexes])
        test_preds = np.argsort(test_preds,axis=1)[:,::-1]
        predictions = [' '.join([args['inv_vocab'][test_preds[i,k]] for k in range(3)]) for i in range(test_preds.shape[0])]
        best_mapk = mapk

out_df = pd.DataFrame({'fname': [f.split('/')[-1] for f in test_files], 'label': predictions})
out_df.to_csv('/kaggle/working/submission.csv',index=False)