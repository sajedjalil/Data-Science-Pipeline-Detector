#PyTorch CNN bird call - no call classifier from 5-second audio chunks
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
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True

#Read an audio file, resample if needed and trim if required
def read_resample(file,**kwargs):
    audio,f=soundfile.read(file)
    if f!=kwargs['sampling']:
        audio=resample(audio,f,kwargs['sampling'])
    nsec = len(audio)/kwargs['sampling']
    samples = int(nsec/kwargs['sliding'])
    whole_samples = samples - (kwargs['windowing'] - kwargs['sliding'])
    return audio, int(whole_samples)
    
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
    nsec = len(audio)/kwargs['sampling']
    samples = int(nsec/kwargs['sliding'])
    whole_samples = int(samples - (kwargs['windowing'] - kwargs['sliding']))
    DATA = []
    for i in range(whole_samples):
        start = i * kwargs['sliding'] * kwargs['sampling'] / hop_length
        finish = start + ( kwargs['windowing'] * kwargs['sampling'] / hop_length )
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
def load_data(files, df, randomize=False, **kwargs):
    n_samples = 0
    #Pre-estimate total number of dataset samples
    for i,file in enumerate(files):
        audio, whole_samples = read_resample(files[i],**kwargs)
        n_samples += whole_samples
    #Allocate, read and store
    dataset = torch.zeros((n_samples,1,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    labels = torch.zeros((n_samples),dtype=torch.uint8)
    idx = 0
    for i in tqdm(range(len(files)),disable=(kwargs['verbose']<2)):
        audio, whole_samples = read_resample(files[i],**kwargs)
        feats = feat2img(gen_logfft(audio,normalise=True,**kwargs),**kwargs)
        dataset[idx:idx+feats.shape[0],0,:,:] = torch.from_numpy(feats)
        site = files[i].split('/')[-1].split('_')[1]
        audio_id = int(files[i].split('/')[-1].split('_')[0])
        file_df = df.loc[(df['site'] == site) & (df['audio_id'] == audio_id)].reset_index(drop=True)
        for j in range(feats.shape[0]):
            start = j * kwargs['sliding']
            low_idx = file_df.loc[(file_df['seconds'])>start].index[0]
            finish = start + kwargs['windowing']
            top_idx = file_df.loc[(file_df['seconds']>=finish)].index[0]
            lab = ('call' if 'call' in file_df.iloc[low_idx:top_idx+1]['birds'].values else 'nocall')
            labels[idx+j] = args['vocab'][lab]
        idx = idx + feats.shape[0]
    assert n_samples == idx
      
    #Randomize, only for training sets
    if randomize:
        idx = [i for i in range(n_samples)]
        random.shuffle(idx)
        dataset = dataset[idx]
        labels = labels[idx]
    return dataset, labels

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
    return np.exp(predictions)

def compute_results(predictions,labels):
    thresholds = np.arange(1.0,-0.000001,-0.001)
    fpr = []
    tpr = []
    acc = []
    prec = []
    rec = []
    f1 = []
    for th in thresholds:
        tp = np.sum((predictions >= th) * labels)
        tn = np.sum((predictions < th) * (1-labels))
        fp = np.sum((predictions >= th) * (1-labels))
        fn = np.sum((predictions < th) * labels)
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(tn+fp))
        acc.append((tp+tn)/(tp+tn+fp+fn))
        prec.append((tp / (tp + fp)) if (tp + fp) > 0 else 0.0)
        rec.append(tp / (tp + fn))
        f1.append((2*tp) / (2*tp + fp + fn))
    results = pd.DataFrame({'thresholds':thresholds,'tpr':tpr,'fpr':fpr,'acc':acc,'prec':prec,'rec':rec,'f1':f1})
    return results

#Arguments
args = {
    'cv_percentage': 0.1,
    'xsize': 200,
    'ysize': 40,
    'n_mels': 40,
    'mel_freq': 16000,
    'num_blocks': 3,
    'channels': 32,
    'dropout': 0.2,
    'reduce_size': True,
    'embedding_size': 128,
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.0001,
    'seed': 0,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
    'augment': True,
    'augmentation_prob': 1.0,
    'augmentation_noise': 10,
    'sampling': 32000, #Resampling frequency in Hertzs
    'windowing': 5.0, #Size of chunks in seconds
    'sliding': 1.0, #Window slide in seconds
    'nfft': 25, #FFT size for STFT (msec)
    'nhop': 25, #Hop length for STFT (msec)
}

init_random(**args)
train_data = pd.read_csv('../input/birdclef-2021/train_soundscape_labels.csv')
train_data.loc[train_data['birds']!='nocall','birds'] = 'call'
train_files = np.sort(glob.glob('../input/birdclef-2021/train_soundscapes/*.ogg'))
random.shuffle(train_files)
valid_files = train_files[-int(len(train_files)*args['cv_percentage']):]
train_files = train_files[:-int(len(train_files)*args['cv_percentage'])]

#Mapping of outputs
args['vocab'] = OrderedDict({'nocall':0,'call':1})
args['inv_vocab'] = {i:v for i,v in enumerate(args['vocab'])}

#Load training, validation data
print('Loading training data: {0:d} files'.format(len(train_files)))
trainset, trainlabels = load_data(train_files, train_data, randomize=True,**args)
print('Loading validation data: {0:d} files'.format(len(valid_files)))
validset, validlabels = load_data(valid_files, train_data, randomize=False,**args)

#Mean and standard deviation for input normalisation
args['mean'] = torch.mean(trainset.float())
args['std'] = torch.std(trainset.float())

train_priors = torch.Tensor([len(np.where(trainlabels.numpy()==t)[0])/trainlabels.shape[0] for t in np.unique(trainlabels)])
prior_weights = (1 / train_priors)
prior_weights = prior_weights / torch.min(prior_weights)

#Model, optimiser and criterion
model = SimpleCNN(**args).to(args['device'])
optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
criterion = nn.NLLLoss(weight = prior_weights,reduction='mean').to(args['device'])

#Iterate through number of epochs
best_f1 = 0.0
for ep in range(1,args['epochs']+1):
    #Train an epoch
    loss = train_model(trainset,trainlabels,model,optimizer,criterion,**args)
    #Get the posteriors for the validation set
    predictions = evaluate_model(validset,model,**args)
    results = compute_results(predictions[:,1],validlabels.numpy())
    th = results.loc[results['acc']==np.max(results['acc'])]['thresholds'].values[0]
    acc = 100*results.loc[results['acc']==np.max(results['acc'])]['acc'].values[0]
    auc = np.trapz(results['tpr'],x=results['fpr'])
    th2 = results.loc[results['f1']==np.max(results['f1'])]['thresholds'].values[0]
    f1 = results.loc[results['f1']==np.max(results['f1'])]['f1'].values[0]
    print('Epoch: {0:d} of {1:d}. Training loss: {2:.2f}, validation AUC: {3:.3f}, validation accuracy: {4:.2f}%@{5:.3f}, validation F1: {6:.3f}@{7:.3f}'.format(ep,args['epochs'],loss,auc,acc,th,f1,th2))
    if f1 > best_f1:
        best_f1 = f1
        model.optimal_threshold = min(th,th2)
        torch.save(model,'birdcall_activity_detector.pytorch')