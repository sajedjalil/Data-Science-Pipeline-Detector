#PyTorch CNN bird call classifier from 5-second audio chunks
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
    if np.max(np.absolute(audio)) == 0:
        return [], 0
    if f!=kwargs['sampling']:
        audio=resample(audio,f,kwargs['sampling'])
    nsec = len(audio)/kwargs['sampling']
    samples = int(nsec/kwargs['sliding'])
    whole_samples = samples - (kwargs['windowing'] - kwargs['sliding'])
    return audio, max(0,int(whole_samples))
    
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
def load_data(files, randomize=False, **kwargs):
    n_samples = len(files) * kwargs['max_samples_per_file']
    #Allocate, read and store
    dataset = torch.zeros((n_samples,1,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    labels = torch.zeros((n_samples),dtype=torch.int)
    idx = 0
    for i in tqdm(range(len(files)),disable=(kwargs['verbose']<2)):
        audio, whole_samples = read_resample(files[i],**kwargs)
        if whole_samples <= 0:
            continue
        feats = feat2img(gen_logfft(audio,normalise=True,**kwargs),**kwargs)
        activity_predictions = evaluate_model(torch.unsqueeze(torch.from_numpy(feats),1),kwargs['activity_model'],**kwargs)
        feats = feats[activity_predictions[:,1]>=kwargs['activity_model'].optimal_threshold,:,:]
        if feats.shape[0] > kwargs['max_samples_per_file']:
            idx2 = list(range(feats.shape[0]))
            random.shuffle(idx2)
            feats = feats[idx2[0:kwargs['max_samples_per_file']],:,:]
        dataset[idx:idx+feats.shape[0],0,:,:] = torch.from_numpy(feats)
        labels[idx:idx+feats.shape[0]] = args['vocab'][files[i].split('/')[-2]]
        idx = idx + feats.shape[0]
    dataset = dataset[:idx,:,:,:]
    labels = labels[:idx]
      
    #Randomize, only for training sets
    if randomize:
        idx = list(range(idx))
        random.shuffle(idx)
        dataset = dataset[idx]
        labels = labels[idx]
    return dataset, labels

#Return the posteriors for a given dataset
def evaluate_model(testset,model,**kwargs):
    testlen = testset.shape[0]
    predictions = np.zeros((testlen,len(model.vocab)))
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    with torch.no_grad():
        model = model.eval()
        for b in range(nbatches):
            #Obtain batch
            X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
            #Propagate
            posteriors = model(X)
            predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size']),:] = posteriors.detach().cpu().numpy()
    return np.exp(predictions)

def compute_accuracy(predictions,labels,at):
    accs = {a:0.0 for a in at}
    estimated = np.argsort(predictions,axis=1)[:,::-1]
    estimated = estimated[:,0:max([a for a in at])]
    for a in at:
        est = estimated[:,0:a]
        errors = np.clip(np.min(np.absolute(est - np.expand_dims(labels,1)),axis=1),a_min=0,a_max=1)
        accs[a] = 100 * (len(labels) - np.sum(errors)) / len(labels)
    return accs

#Arguments
args = {
    'min_total_files': 20,
    'train_files_per_bird': 80,
    'cv_files_per_bird': 10,
    'max_samples_per_file': 50,
    'xsize': 200,
    'ysize': 40,
    'n_mels': 40,
    'mel_freq': 16000,
    'num_blocks': 3,
    'channels': 32,
    'dropout': 0.0,
    'reduce_size': True,
    'embedding_size': 128,
    'epochs': 10,
    'batch_size': 64,
    'learning_rate': 0.001,
    'seed': 0,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
    'augment': False,
    'augmentation_prob': 1.0,
    'augmentation_noise': 10,
    'sampling': 32000, #Resampling frequency in Hertzs
    'windowing': 5.0, #Size of chunks in seconds
    'sliding': 1.0, #Window slide in seconds
    'nfft': 25, #FFT size for STFT (msec)
    'nhop': 25, #Hop length for STFT (msec)
}

init_random(**args)
#Mapping of outputs
train_data = pd.read_csv('../input/birdclef-2021/train_metadata.csv')
args['vocab'] = OrderedDict({t:i for i,t in enumerate(np.unique(train_data['primary_label']))})
args['inv_vocab'] = {i:v for i,v in enumerate(args['vocab'])}

#Split data
train_files = []
valid_files = []
for folder in np.sort(glob.glob('../input/birdclef-2021/train_short_audio/*')):
    files = np.sort(glob.glob(folder+'/*.ogg'))
    if len(files) < args['min_total_files']:
        continue
    random.shuffle(files)
    train_files.append(list(files[0:min(args['train_files_per_bird'],len(files)-args['cv_files_per_bird'])]))
    valid_files = valid_files + list(files[-args['cv_files_per_bird']:])
    files = files[:-args['cv_files_per_bird']]
    files = [f for i in range(math.ceil(args['train_files_per_bird']/len(files))) for f in files]
    train_files.append(list(files[0:args['train_files_per_bird']]))

#Load bird call activity detection model
args['activity_model'] = torch.load('../input/birdcall-activitydetectorlogmelcnn/birdcall_activity_detector.pytorch').to(args['device'])

#Load training, validation data
print('Loading validation data: {0:d} files'.format(len(valid_files)))
validset, validlabels = load_data(valid_files, randomize=False,**args)
print('{0:d} validation samples'.format(validset.shape[0]))

#Mean and standard deviation for input normalisation
args['mean'] = torch.mean(validset.float())
args['std'] = torch.std(validset.float())

#Model, optimiser and criterion
model = SimpleCNN(**args).to(args['device'])
optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
criterion = nn.NLLLoss(reduction='mean').to(args['device'])

#Iterate through number of epochs
print('Training...')
for ep in range(1,args['train_files_per_bird']+1):
    #Train an epoch
    trainset, trainlabels = load_data([files[ep-1] for files in train_files if len(files)>=ep], randomize=True,**args)
    loss = train_model(trainset,trainlabels,model,optimizer,criterion,**args)
    #Get the posteriors for the validation set
    val_preds = evaluate_model(validset,model,**args)
    accs = compute_accuracy(val_preds,validlabels.numpy(),at=[1,5,10])
    print('Epoch: {0:d} of {1:d}. Training loss: {2:.2f}, validation accuracy: top1:{3:.2f}%, top5:{4:.2f}%, top10:{5:.2f}%'.format(ep,args['train_files_per_bird'],loss,accs[1],accs[5],accs[10]))
    
model = model.cpu()
torch.save(model,'birdcall_classifier.pytorch')