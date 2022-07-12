#PyTorch CNN bird call inference
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
from itertools import combinations
from PIL import Image

#Clone and import personal CNNWordReco repository
if '../input/birdcall-classificationlogmelcnn/CNNWordReco' not in sys.path:
    sys.path.append('../input/birdcall-classificationlogmelcnn/CNNWordReco')
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
def load_data(files, **kwargs):
    n_samples = 0
    #Pre-estimate total number of dataset samples
    for i,file in enumerate(files):
        audio, whole_samples = read_resample(files[i],**kwargs)
        n_samples += whole_samples
    #Allocate, read and store
    dataset = torch.zeros((n_samples,1,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    row_id = []
    idx = 0
    for i in tqdm(range(len(files)),disable=(kwargs['verbose']<2)):
        audio, whole_samples = read_resample(files[i],**kwargs)
        feats = feat2img(gen_logfft(audio,normalise=True,**kwargs),**kwargs)
        dataset[idx:idx+feats.shape[0],0,:,:] = torch.from_numpy(feats)
        audio_id = files[i].split('/')[-1].split('_')[0]
        site = files[i].split('/')[-1].split('_')[1]
        for j in range(feats.shape[0]):
            row_id.append('{0:s}_{1:s}_{2:d}'.format(audio_id,site,int(j*kwargs['sliding']+kwargs['sliding'])))
        idx = idx + feats.shape[0]
    assert n_samples == idx

    return dataset, row_id

#Return the posteriors for a given dataset
def evaluate_model(testset,act_model,cls_model,**kwargs):
    testlen = testset.shape[0]
    act_predictions = np.zeros((testlen,len(act_model.vocab)))
    cls_predictions = np.zeros((testlen,len(cls_model.vocab)))
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    with torch.no_grad():
        act_model = act_model.eval()
        cls_model = cls_model.eval()
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            for b in range(nbatches):
                #Obtain batch
                X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
                #Propagate
                act_posteriors = act_model(X)
                act_predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size']),:] = act_posteriors.detach().cpu().numpy()
                cls_posteriors = cls_model(X)
                cls_predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size']),:] = cls_posteriors.detach().cpu().numpy()
                pbar.set_description('Testing')
                pbar.update()
    return np.exp(act_predictions), np.exp(cls_predictions)

def predict_birds(act_preds, cls_preds, act_th, cls_vocab):
    inv_vocab = {cls_vocab[v]:v for v in cls_vocab}
    birds = []
    for i in range(act_preds.shape[0]):
        if act_preds[i,1] >= act_th:
            best = np.argsort(cls_preds[i,:])[::-1]
            best = best[0:5]
            birds.append(' '.join([inv_vocab[b] for b in best]))
        else:
            birds.append('nocall')
    return birds

def compute_micro_average_f1_score(df_pred,df_true):
    f1 = []
    for i in df_true['row_id'].values:
        labels = df_true.loc[df_true['row_id']==i]['birds'].values[0]
        labels = labels.strip().split()
        preds = df_pred.loc[df_pred['row_id']==i]['birds'].values[0]
        preds = preds.strip().split()
        tp = 0
        fp = 0
        fn = 0
        for l in labels:
            if l in preds:
                tp += 1
            else:
                fn += 1
        for p in preds:
            if p not in labels:
                fp += 1
        f1.append((2*tp)/(2*tp+fp+fn))
    return np.mean(f1)

#Arguments
args = {
    'xsize': 200,
    'ysize': 40,
    'n_mels': 40,
    'mel_freq': 16000,
    'batch_size': 64,
    'seed': 0,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
    'sampling': 32000, #Resampling frequency in Hertzs
    'windowing': 5.0, #Size of chunks in seconds
    'sliding': 5.0, #Window slide in seconds
    'nfft': 25, #FFT size for STFT (msec)
    'nhop': 25, #Hop length for STFT (msec)
}

#Use train soundscapes to calibrate
init_random(**args)
calibration_data = pd.read_csv('../input/birdclef-2021/train_soundscape_labels.csv')
calibration_files = np.sort(glob.glob('../input/birdclef-2021/train_soundscapes/*.ogg'))
print('Loading calibration data...')
calibset,calibid = load_data(calibration_files, **args)

#Estimate baseline where everything is nocall
baseline = pd.DataFrame({'row_id': calibid,'birds':['nocall' for i in calibid]})
f1 = compute_micro_average_f1_score(baseline,calibration_data)
print('Baseline F1 score for naive result: {0:.3f}'.format(f1))

#Load models
activity_model = torch.load('../input/birdcall-activitydetectorlogmelcnn/birdcall_activity_detector.pytorch').to(args['device'])
classification_model = torch.load('../input/birdcall-classificationlogmelcnn/birdcall_classifier.pytorch').to(args['device'])

#Generate birdcall predictions for calibration
print('Generating predictions for calibration...')
act_predictions, cls_predictions = evaluate_model(calibset,activity_model,classification_model,**args)
calibration_birds = predict_birds(act_predictions,cls_predictions,activity_model.optimal_threshold,classification_model.vocab)
calibration = pd.DataFrame({'row_id': calibid,'birds':calibration_birds})
f1 = compute_micro_average_f1_score(calibration,calibration_data)
print('F1 score for calibration: {0:.3f}'.format(f1))

#Load test soundscapes
test_files = np.sort(glob.glob('../input/birdclef-2021/test_soundscapes/*.ogg'))
print('Loading test data...')
testset,testid = load_data(test_files, **args)

print('Generating predictions for test...')
act_predictions, cls_predictions = evaluate_model(testset,activity_model,classification_model,**args)
birds = predict_birds(act_predictions,cls_predictions,activity_model.optimal_threshold,classification_model.vocab)

#Produce output
results = pd.DataFrame({'row_id': testid,'birds':birds})
results.to_csv('submission.csv',index=False)