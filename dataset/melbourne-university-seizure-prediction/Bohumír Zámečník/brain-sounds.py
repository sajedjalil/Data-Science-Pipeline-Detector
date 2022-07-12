"""
Would you like to listen to brain? How it sounds? Are they any
patterns? Any differences between intraictal and preitcal samples?

This script converts some samples on EEG data into sound (a WAV file).
It's ~100x faster (converting sampling rate 400 Hz to 44.1 kHz),
ie. 10 minutes are compressed into ~5 seconds.
The 16 channels are scaled to [-1, 1] range separately and are put
one after other in mono.

It uses scipy wavfile for WAV output. The soundfile library is better
and allows compressed lossless FLAC, but it's not available in Kaggle
Kernels, so try it at home to save ~50% file size.
"""

import os
import numpy as np
from scipy.io import loadmat

try:
    # Allows other file formats, such as FLAC.
    # But is not available in Kaggle Kernels.
    import soundfile as sf
    def save_audio(output_file, data, sample_rate):
        sf.write(output_file, data, sample_rate)
except ImportError:
    from scipy.io import wavfile
    def save_audio(output_file, data, sample_rate):
        wavfile.write(output_file, sample_rate, np.int16(data * 2 ** 15))

from sklearn.preprocessing import MinMaxScaler
import sys


def convert(mat):
    # structure:
    # mat: dict
    # mat['dataStruct']: ndarray (1, 1) of type dtype([('data', 'O'),
    # ('iEEGsamplingRate', 'O'), ('nSamplesSegment', 'O'),
    # ('channelIndices', 'O'), ('sequence', 'O')])
    #
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

def auralize(input_file, sample_rate=44100):
    prefix, ext = os.path.basename(input_file).split('.')
    data = convert(loadmat(input_file))['data']
    channel_count = data.shape[1]
    # scale to [-1, 1], put all channels after each other
    y = np.vstack([MinMaxScaler(feature_range=(-1, 1)).fit_transform(data[:, i:i+1]) for i in range(channel_count)])
    save_audio('%s_%d.%s' % (prefix, sample_rate, 'wav'), y, sample_rate)

input_dir = '../input/train_1/'
for f in ['1_100_0', '1_100_1', '1_101_0', '1_101_1', '1_102_0', '1_102_1']:
    auralize('%s/%s.mat' % (input_dir, f))

