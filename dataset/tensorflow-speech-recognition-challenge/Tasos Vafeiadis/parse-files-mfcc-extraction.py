import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import re
import librosa
import lzma

lzma.open('train.7z')
DATADIR = './data'  # train and test data
OUTDIR = './CERTH'  # our model output
L = 16000           # global length variable
#==============================================================================#
#                          Dataset Manipulation                                #
#==============================================================================#
final_labels = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(final_labels)}
name2id = {name: i for i, name in id2name.items()}
print(sorted(name2id.items(), key=lambda item: item, reverse=False))

#==============================================================================#
#                           Load the given train/val                           #
#==============================================================================#
def load_data(data_dir):
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    file_pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")  # Regular expression for file pattern
    validation_set = set()   # Empty set
    for entry in validation_files:
        r = re.match(file_pattern, entry)
        if r:
            validation_set.add(r.group(3))
    possible = set(final_labels)
    train, val = [], []
    for entry in all_files:
        r = re.match(file_pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'
            label_id = name2id[label]
            sample = (label, label_id, uid, entry)
            if uid in validation_set:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))

    columns_list = ['label', 'label_id', 'user_id', 'wav_file']

    train_df = pd.DataFrame(train, columns = columns_list)
    valid_df = pd.DataFrame(val, columns = columns_list)

    return train_df, valid_df

trainset, validation_set = load_data(DATADIR)
print("The shape of the training folder including silence files is:", trainset.shape)
print("The shape of the validation folder is:", validation_set.shape)

# Separating silence data from the rest
silence_files = trainset[trainset.label == 'silence']
print("The shape of the silence files is:", silence_files.shape)
trainset_nosil = trainset[trainset.label != 'silence']
print("The shape of the training folder without silence files is:", trainset_nosil.shape)

#==============================================================================#
#                      Pad audio clips less than 1 second                      #
#==============================================================================#    
def pad_audio(data, fs=L, T=1):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape   
    N_pad = N_tar - shape[0]
    # print("Padding with %s seconds of silence" % str(N_pad/fs))
    shape = (N_pad,) + shape[1:] 
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((np.zeros(shape), data))
        else:
            return np.hstack((np.zeros(shape), data))
    else:
        return data

#==============================================================================#
#                    Clip audio clips greater than 1 second                    #
#==============================================================================#
def chop_audio(samples, L=L, num=20):
    for i in range(num):
        begin = np.random.randint(0, len(samples) - L)
        samples = samples[begin: begin + L]
        return samples

#==============================================================================#
#                                 Read wav file                                #
#==============================================================================#
def read_wav_file(fname):
    wav, sr = librosa.load(fname, sr=16000)
    # print('Audio Sampling Rate: '+ str(sr) + ' samples/sec')
    # print('Total Samples: '+ str(np.size(wav)))
    # secs = np.size(wav)/sr
    # print('Audio Length: '+ str(secs) + ' s')  # Some Recordings are longer than 1 minute
    return wav

#==============================================================================#
#                             Extract features                                 #
#==============================================================================#
features_df = []
print("#==============================================================================#")
print("#                     Started parsing the silence files                        #")
print("#==============================================================================#")
wav_length = 0
for k in tqdm(range(0,len(silence_files.wav_file))):
    x = silence_files.wav_file.values[k]
    wav_data = np.concatenate([read_wav_file(x)])
    wav_length += len(wav_data)/L # in seconds
    # print("Old length:", len(wav_data))
    if len(wav_data) > L:
        n_samples = chop_audio(wav_data)
        # print("New Length:", len(n_samples))
    else:
        n_samples = pad_audio(wav_data)
        # print("New Length:", len(n_samples))
    
    # Extract y_harmonic and y_percussive
    y_harmonic, y_percussive = librosa.effects.hpss(n_samples)
    # Get mfccs every 20 ms with hop size of 10 ms
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=L, n_mfcc=13, hop_length=int(0.010*L), n_fft=int(0.025*L))
    features_df = mfccs
print("#==============================================================================#")
print("#                     Finished parsing the silence files                       #")
print("#==============================================================================#")

#features_df = np.append(features_df)
#print(features_df.shape)