#PyTorch CNN classifier from 1-second audio chunks
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
from sklearn.metrics import label_ranking_average_precision_score
from tqdm import tqdm
import librosa
from itertools import combinations
from PIL import Image

#Clone and import personal CNNWordReco repository
if ~os.path.isdir('CNNWordReco'):
    subprocess.call(['git', 'clone', 'https://github.com/saztorralba/CNNWordReco'])
if 'CNNWordReco' not in sys.path:
    sys.path.append('CNNWordReco')
from utils.cnn_func import train_model
from models.SimpleCNN import SimpleCNN

#Simple function to initialise all the random seeds to a given value for reproducibility
def init_random(**kwargs):
    random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True

#Read an audio file, resample if needed and trim if required
def read_resample(file,st,en,window,**kwargs):
    audio,f=soundfile.read(file)
    if f!=kwargs['sampling']:
        audio=resample(audio,f,kwargs['sampling'])
    if st == -1:
        return audio, 0
    st = int(max(0,(st - window*kwargs['windowing']) * kwargs['sampling']))
    en = int(min(len(audio), (en + window*kwargs['windowing']) * kwargs['sampling']))
    audio = audio[st:en]
    return audio, st/kwargs['sampling']
    
#Extract the log spectrogram of an audio signal in chunks of predetermined size
def gen_logfft(audio,normalise=False,**kwargs):
    #Parameters
    epsilon=1e-20
    win_length=int(kwargs['nfft']*kwargs['sampling']/1000)
    n_fft=int(math.pow(2,math.ceil(math.log(win_length)/math.log(2.0))))
    hop_length=int(kwargs['nhop']*kwargs['sampling']/1000)
    #Normalise input energy
    if normalise:
        audio=0.5*audio/np.max(np.absolute(audio))
    #High-pass filter
    audio=scipy.signal.convolve(audio,np.array([1,-0.98]),mode='same',method='fft')
    #Compute mel spectrogram
    stft=librosa.feature.melspectrogram(y=audio,sr=kwargs['sampling'],win_length=win_length,n_fft=n_fft,hop_length=hop_length,n_mels=kwargs['n_mels'],fmin=20,fmax=kwargs['mel_freq'],norm=1)
    stft=np.transpose(np.log(stft+epsilon))
    #Divide in chunks of length defined by the argument 'windowing'
    duration = (len(audio)-1)/kwargs['sampling']
    n_samples = int((duration)/kwargs['windowing'])+((duration%kwargs['windowing'])>(kwargs['windowing']/2))
    DATA = []
    for i in range(n_samples):
        start = i * kwargs['windowing'] * kwargs['sampling'] / hop_length
        if i == n_samples-1:
            finish = stft.shape[0]
        else:
            finish = (i+1) * kwargs['windowing'] * kwargs['sampling'] / hop_length
        DATA.append(stft[int(start):int(finish),:])
    return DATA

#Convert to image, normalise, resize, then return to numpy
def feat2img(DATA,**kwargs):
    out = np.zeros((len(DATA),kwargs['ysize'],kwargs['xsize']))
    for i in range(len(DATA)):
        #Reorg dimensions
        DATA[i] = np.flipud(np.transpose(DATA[i]))
        #Clamp and normalise to [0,1]
        DATA[i] = (np.maximum(-15,np.minimum(0,DATA[i]))+15)/15
        #Convert to PIL
        im = Image.fromarray(np.uint8(DATA[i]*255))
        #Resize
        im = im.resize((kwargs['xsize'],kwargs['ysize']))
        #Back to numpy
        out[i,:,:] = np.array(im)
    return out
    
#Read data as Torch tensors from a data source
def load_data(files, targets=None, starts=None, ends=None, randomize=False, **kwargs):
    n_samples = 0
    indexes = []
    idx=0
    #Pre-estimate total number of dataset samples
    for i,file in enumerate(files):
        if targets is None:
            audio, _ = read_resample(files[i],-1,-1,-1,**kwargs)
        else:
            audio, _ = read_resample(files[i],starts[i],ends[i],5,**kwargs)
        if len(audio) == 0:
            indexes.append([])
            continue
        duration = (len(audio)-1)/kwargs['sampling']
        duration = int(duration/kwargs['windowing'])+((duration%kwargs['windowing'])>(kwargs['windowing']/2))
        n_samples += duration
        indexes.append(list(range(idx,idx+duration)))
        idx = idx + duration
    #Allocate, read and store
    dataset = torch.zeros((n_samples,1,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    labels = torch.zeros((n_samples),dtype=torch.uint8)
    idx = 0
    for i in tqdm(range(len(files)),disable=(kwargs['verbose']<2)):
        if len(indexes[i]) == 0:
            continue
        if targets is None:
            audio, _ = read_resample(files[i],-1,-1,-1,**kwargs)
        else:
            audio, st = read_resample(files[i],starts[i],ends[i],5,**kwargs)
        feats = feat2img(gen_logfft(audio,normalise=True,**kwargs),**kwargs)
        dataset[idx:idx+feats.shape[0],0,:,:] = torch.from_numpy(feats)
        if targets is not None:
            en = math.ceil((ends[i] - st)/kwargs['windowing'])+1
            st = math.floor((starts[i] - st)/kwargs['windowing'])
            labels[idx:idx+feats.shape[0]] = kwargs['vocab'][-1]
            labels[idx+st:idx+en] = kwargs['vocab'][targets[i]]
        idx = idx + feats.shape[0]
    assert n_samples == idx
      
    #Randomize, only for training sets
    if randomize:
        idx = [i for i in range(n_samples)]
        random.shuffle(idx)
        dataset = dataset[idx]
        labels = labels[idx]
        indexes = []
    return dataset, labels, indexes

#Return the posteriors for a given dataset
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

#Return the average probability across target classes for non-silence audio chunks
def compute_fileprobs(preds,indexes,mini=1,maxi=np.inf):
    output = []
    for i in range(len(indexes)):
        tmp = np.exp(preds[indexes[i]])
        l = 0
        th = 0.5
        #For each file, find which chunks are more likely to contain a species,
        #then sum the posteriors across the selected chunks
        while (l < mini or l > maxi) and th < 1.0 and th > 0.0:
            tmp = tmp[np.where(tmp[:,0]<th)[0],:]
            l = tmp.shape[0]
            if l < mini:
                th = th + 0.05
            if l > maxi:
                th = th - 0.05
        tmp = np.sum(tmp,axis=0)[1:]
        if np.sum(tmp) > 0:
            output.append(tmp/np.sum(tmp))
        else:
            output.append(tmp)
    return np.array(output)

#Arguments
args = {
    'cv_percentage': 0.1,
    'xsize': 40,
    'ysize': 40,
    'n_mels': 40,
    'mel_freq': 16000,
    'num_blocks': 3,
    'channels': 32,
    'dropout': 0.4,
    'embedding_size': 128,
    'epochs': 60,
    'batch_size': 256,
    'learning_rate': 0.001,
    'seed': 0,
    'n_seeds': 10,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
    'augment': False,
    'sampling': 48000, #Resampling frequency in Hertzs
    'windowing': 1.0, #Size of chunks in seconds
    'nfft': 20, #FFT size for STFT
    'nhop': 20, #Hop length for STFT
}

init_random(**args)

#Read data from dataframes and folders
train_data = pd.read_csv('/kaggle/input/rfcx-species-audio-detection/train_tp.csv')
train_files = np.array(['/kaggle/input/rfcx-species-audio-detection/train/'+v+'.flac' for v in train_data.recording_id.values])
train_targets = np.array(list(train_data.species_id.values))
train_starttime = np.array(list(train_data.t_min.values))
train_endtime = np.array(list(train_data.t_max.values))
test_files = glob.glob('/kaggle/input/rfcx-species-audio-detection/test/*.flac')

#Mapping of outputs
args['vocab'] = OrderedDict({t:i for i,t in enumerate(np.unique([-1]+list(train_targets)))})
args['inv_vocab'] = {i:v for i,v in enumerate(args['vocab'])}

#Get a training/validation split
probs = np.array([random.random() for i in range(len(train_files))])
valid_files = train_files[probs>(1-args['cv_percentage'])]
valid_targets = train_targets[probs>(1-args['cv_percentage'])]
valid_starttime = train_starttime[probs>(1-args['cv_percentage'])]
valid_endtime = train_endtime[probs>(1-args['cv_percentage'])]
train_files = train_files[probs<=(1-args['cv_percentage'])]
train_targets = train_targets[probs<=(1-args['cv_percentage'])]
train_starttime = train_starttime[probs<=(1-args['cv_percentage'])]
train_endtime = train_endtime[probs<=(1-args['cv_percentage'])]

#Load training, validation and testing data
#For training and validation only the actual sounds are selected, plus 5 seconds before and after
print('Loading training data: {0:d} files'.format(len(train_files)))
trainset, trainlabels, _ = load_data(train_files, train_targets, train_starttime, train_endtime, randomize=True,**args)
print('Loading validation data: {0:d} files'.format(len(valid_files)))
validset, validlabels, validindexes = load_data(valid_files, valid_targets, valid_starttime, valid_endtime, randomize=False,**args)
print('Loading evaluation data: {0:d} files'.format(len(test_files)))
testset, _, testindexes = load_data(test_files, None, None, None, randomize=False, **args)

#Mean and standard deviation for input normalisation
args['mean'] = torch.mean(trainset.float())
args['std'] = torch.std(trainset.float())

#Ground truth matrix for LARP
ground_truth = np.zeros((len(valid_targets),len(np.unique(valid_targets))))
ground_truth[[i for i in range(len(valid_targets))],valid_targets] = 1

train_priors = torch.Tensor([len(np.where(trainlabels.numpy()==t)[0])/trainlabels.shape[0] for t in np.unique(trainlabels)])
#Weights for loss, they are normalised, but weight of no-sound class is kept at one
prior_weights = 1 / (train_priors / torch.max(train_priors[1:]))
prior_weights[0] = 1.0

best_val_preds = dict()
best_test_preds = dict()
#Run multiple seeds
for seed in range(args['n_seeds']):
    args['seed'] = seed
    init_random(**args)
    #Reshuffle train data
    idx = [i for i in range(trainset.shape[0])]
    random.shuffle(idx)
    trainset = trainset[idx]
    trainlabels = trainlabels[idx]
    #Model, optimiser and criterion
    model = SimpleCNN(**args).to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
    criterion = nn.NLLLoss(weight = prior_weights,reduction='mean').to(args['device'])
    best_larp = 0
    #Iterate through number of epochs
    for ep in range(1,args['epochs']+1):
        #Train an epoch
        loss = train_model(trainset,trainlabels,model,optimizer,criterion,**args)
        #Get the posteriors for the validation set
        val_preds = evaluate_model(validset,model,**args)
        #Collapse the posteriors at file level
        val_preds_file = compute_fileprobs(val_preds,validindexes,mini=1)
        #Compute LARP
        larp = label_ranking_average_precision_score(ground_truth,val_preds_file)
        #Find the epoch with the best LARP
        if larp >= best_larp:
            best_val_preds[seed] = val_preds
            #Posteriors for the evaluation set
            best_test_preds[seed] = evaluate_model(testset,model,**args)
            best_epoch = ep
            best_larp = larp
            best_loss = loss
    print('Model {0:d}. Best epoch: {1:d}. Training loss: {2:.2f}, larp: {3:.3f}'.format(seed,best_epoch,best_loss,best_larp))

#Merge the model outputs
val_preds_file = np.mean(([compute_fileprobs(best_val_preds[i],validindexes,mini=1) for i in best_val_preds]),axis=0)
larp = label_ranking_average_precision_score(ground_truth,val_preds_file)
print('Model combination larp: {0:.3f}'.format(larp))

#Compute the final file-level posteriors for the test set
test_results = np.mean([compute_fileprobs(best_test_preds[i],testindexes,mini=5,maxi=20) for i in best_test_preds],axis=0)
tmp = {**{'recording_id': [f.split('/')[-1].split('.')[0] for f in test_files]}, **{'s'+str(i):test_results[:,i] for i in range(24)}}
out_df = pd.DataFrame(tmp)
out_df.to_csv('/kaggle/working/submission.csv',index=False)