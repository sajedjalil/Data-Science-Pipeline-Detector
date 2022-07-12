# -*- coding: utf-8 -*-
"""
Created on 

@author: MB
@Copyright: Michel Brabants
"""

from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf 
from pathlib import Path
import pandas as pd
import joblib
import math
import re
from pydub import AudioSegment
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import os 
import tensorflow as tf
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, GlobalMaxPool2D, LeakyReLU, BatchNormalization, ReLU, Flatten
from tensorflow.keras.models import Model
from collections import Counter
from sklearn.metrics import confusion_matrix, hamming_loss, accuracy_score 
import matplotlib.pyplot as plt 
import seaborn as sns
import IPython.display 
from IPython.display import Audio
    
#preprocess audio with pcen
from distutils.version import LooseVersion, StrictVersion

submission=True

if not submission:
    #disable gpu, but not doing much. overheating.
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

if (StrictVersion(librosa.__version__) < StrictVersion('0.6.1')):
    print('Librosa is too old') #implementation of pdf: Per-Channel Energy Normalization: Why and How

# should be 232 bands/buckets, similar to amount to capture the human voice
fft_buckets = 232
sr = 22050

use_test_model = False

project_dir = Path.cwd().parent #overwrite if needed

cleaned_audio_dir = project_dir / "cleaned_audio"
cleaned_audio_dir.mkdir(exist_ok=True)

train_session = 3
data_dir = project_dir / "train_sessions" / f"{train_session}"

predictions_dir = project_dir / "predictions"
predictions_dir.mkdir(exist_ok=True)


glob_bird_codes_map = None

# resample to librosa standard
def get_audio_data(path: Path, sr=22050):    
    #mono: put it all together (no use for stereo at the moment)
    print(f"get_audio_data: loading {path}")
    return librosa.load(path, sr, mono=True)

# function to get start and end indices for audio sub-sample 
def windows(data, window_size): 
    start = 0 
    while start < len(data): 
        yield int(start), int(start + window_size) 
        start += (window_size / 2)

def extract_features(info: pd.DataFrame, bands=232, frames=64, sr=22050):
    """info contains pandas frame with 2 columns: filepath and label.
    Splitting up per file to lower memory-usage and better usability"""
    window_size = 512 * (frames - 1)
    #general stuff
    class_labels = [] #vector (multi-label indicating presence of class)
    audio_samples_info = pd.DataFrame(data=[], columns=['filepath', 'start', 'end']) #allow us to investigate: tuple: filename, start, end 
    features = None
    
    
    total_entries=info.shape[0]
    
    print(f'extract_features: starting to process {total_entries} recordings')
    current_entry = 0
    
    # for each audio sample
    for index, recording in info.iterrows():
        current_entry += 1
        file_name = recording.filepath
        class_label = recording.label #tuple multilabelbinary
        sound_data, sr = get_audio_data(file_name, sr=22050)
        print(f'extract_features: {class_label}: processing file {file_name} ({current_entry}/{total_entries})')
        features_cur_run = None #temporary storage so we can save now and then
        
        features_cur_run, rec_info = get_stored_audio_features(file_name)
        if features_cur_run is not None and rec_info is not None:
            print(f'extract_features: {class_label}: {file_name}: loaded from storage')
            audio_samples_info = audio_samples_info.append(rec_info, ignore_index=True)
            class_labels.extend([class_label]*features_cur_run.shape[0])
            print(f'extract_features: {class_label}: {file_name}: added {features_cur_run.shape[0]} labels')
        else:
            pcen_specgrams_full = []
            log_specgrams_hp = []
            
            # for each audio signal sub-sample window of data
            for (start, end) in windows(sound_data, window_size):
                if(len(sound_data[start:end]) == window_size):
                    signal = sound_data[start:end]

                    # get the log-scaled mel-spectrogram
                    melspec_full = librosa.feature.melspectrogram(signal, n_mels=bands)
                    pcenspec_full = librosa.pcen(melspec_full, sr, time_constant=0.06, gain=0.8, bias=10, power=0.25, eps=1e-06) #, ref=np.max
                    pcenspec_full = pcenspec_full * (2**31)
                    pcenspec_full = pcenspec_full.T.flatten()[:, np.newaxis].T

                    # get the log-scaled, averaged values for the
                    # harmonic and percussive components
                    y_harmonic, y_percussive = librosa.effects.hpss(signal)
                    melspec_harmonic = librosa.feature.melspectrogram(y_harmonic, n_mels=bands)
                    melspec_percussive = librosa.feature.melspectrogram(y_percussive, n_mels=bands)
                    logspec_harmonic = librosa.amplitude_to_db(melspec_harmonic)
                    logspec_percussive = librosa.amplitude_to_db(melspec_percussive)
                    logspec_harmonic = logspec_harmonic.T.flatten()[:,np.newaxis].T
                    logspec_percussive = logspec_percussive.T.flatten()[:, np.newaxis].T
                    logspec_hp = np.average([logspec_harmonic, logspec_percussive], axis=0)
                    pcen_specgrams_full.append(pcenspec_full)
                    log_specgrams_hp.append(logspec_hp)
                    class_labels.append(class_label)
                    audio_samples_info = audio_samples_info.append({'filepath': recording.filepath, 'start': start, 'end': end}, ignore_index=True)# do not store audio-data, only ranges

            # create the first two feature maps for current file
            pcen_specgrams_full = np.asarray(pcen_specgrams_full).reshape( len(pcen_specgrams_full), bands, frames, 1 ) #would expect frames to come first and then bands, but it switched around here. Now, for the processing, it probably doesn't matter, but link to frame is not so linear (transposed)
            log_specgrams_hp = np.asarray(log_specgrams_hp).reshape( len(log_specgrams_hp), bands,  frames, 1)
            features_cur_run = np.concatenate((pcen_specgrams_full, log_specgrams_hp, np.zeros(np.shape(pcen_specgrams_full))), axis=3)
            # create the third feature map which is the delta (derivative)
            # of the log-scaled mel-spectrogram
            for i in range(len(features_cur_run)):
                features_cur_run[i, :, :, 2] = librosa.feature.delta(features_cur_run[i, :, :, 0])

            store_recording_audio_features(recording.filepath, features_cur_run, audio_samples_info)
    
        #append and clear current run
        if features is None:
            features = features_cur_run.astype(np.float32)
            print(f'extract_features: new features. shape is {features.shape}')
        else:
            # some more mem-copying, but better than running out of memory. can be improved of course to multi-recording
            features = np.concatenate((features, features_cur_run.astype(np.float32)), axis=0)
            print(f'extract_features: concatenated new features. new shape is {features.shape}')
    
    #float32 should be enough. limit memory
    return features, np.array(class_labels, dtype=np.bool_), audio_samples_info

def spec_windows_speedy(data, window_size, start_position=0): 
    '''size is the feature-vectors window-size. start_position would allow us to not take the whole file into memory at once. streaming. It should be the end of the previous run then'''
    start = start_position 
    while start < len(data): 
        yield int(start), int(start + window_size) 
        start += window_size
        
audio_frames = 64 #improve: use everywhere
audio_hopsize = 512 #improve: use everywhere

def get_window_size(frames=None):
    if frames is not None:
        audio_frames = frames
        
    return audio_hopsize * (audio_frames - 1)

def get_model_input_audio_sample_size(frames=None):
    '''multiples of get_window_size, for now 1'''
    if frames is not None:
        audio_frames = frames
        
    return get_window_size(audio_frames)

def get_audio_feature_extraction_input_pd():
    '''no time for classes now'''
    return pd.DataFrame(data=[], columns=['filepath', 'label', 'audio_samples'])

def extract_features_speedy(info: pd.DataFrame, bands=232, frames=64, use_cache=True, sr=22050, predict=False):
    """Use feature cache to only calculate 1/2 of spectogram for subsequent features. 
    1. Generate spectograms continously for input (pcen, harmonic, difference), without overlap. parameter to store it to disk (non-overlapping). (max) input = 0.5s seconds of audio
       Little bit less accurate with the padding each time-frame and non-repetitive parsing, but we could parse a minimum of (sound-window:1.6 seconds) to limit the impact
    2. Given start-sample-location (time) (smallest chunk=512 samples:hop length), give back the spectogram.
    predict = False, ignore labels (None), expect raw audio-data and a path for an imaginary file, but under which we'll store the features. Could be programmed cleaner."""
    window_size = get_window_size(frames)
    print(f'extract_features_speedy: Feature vector window size is: {window_size/sr} seconds, {window_size} samples')
    #general stuff
    class_labels = [] #vector (multi-label indicating presence of class)
    audio_samples_info = pd.DataFrame(data=[], columns=['filepath', 'start', 'end']) #allow us to investigate: tuple: filename, start, end 
    features = None
    
    
    total_entries=info.shape[0]
    
    print(f'extract_features_speedy: starting to process {total_entries} recordings')
    current_entry = 0
    
    # for each audio sample
    for index, recording in info.iterrows():
        current_entry += 1
        file_name = recording.filepath
        class_label = None 
        
        sound_data = []
        if predict:
            #use raw audio input
            sound_data = recording.audio_samples
        else:
            # training data
            sound_data, sr = get_audio_data(file_name, sr=sr)
            class_label = recording.label #tuple multilabelbinary
        
        print(f'extract_features_speedy: {class_label}: processing file {file_name} ({current_entry}/{total_entries})')
        features_cur_run = None #temporary storage so we can save now and then
        
        if use_cache:
            features_cur_run, rec_info = get_stored_audio_features(file_name)
        
        if features_cur_run is not None and rec_info is not None:
            print(f'extract_features_speedy: {class_label}: {file_name}: loaded from storage')
            audio_samples_info = audio_samples_info.append(rec_info, ignore_index=True)
            class_labels.extend([class_label]*features_cur_run.shape[0])
            print(f'extract_features_speedy: {class_label}: {file_name}: added {features_cur_run.shape[0]} labels')
        else:
            #specrams, continously for recording
            pcen_specgrams_recording = []
            hp_specgrams_recording = []
            
            #the overlapped specgrams
            pcen_specgrams_overlapping = []
            hp_specgrams_overlapping = []
            
            for (start, end) in spec_windows_speedy(sound_data, window_size):
                samples_in_signal = len(sound_data[start:end])
                signal_smaller_than_window = (samples_in_signal < window_size)
                
                process_signal = True
                if signal_smaller_than_window:
                    if samples_in_signal < sr:
                        print(f'extract_features_speedy: need atleast 1 second of audio for meaningfull content')
                        process_signal = False
                    else:
                        #pad vector with 0db (silence) for spectograms (db-scale)
                        print(f'extract_features_speedy: insufficient audio-samples (start: {start}, samples: {samples_in_signal}) for vector sample-windows-size {window_size}. Suffixing with silence (0db)')
                
                if process_signal:
                    signal = sound_data[start:end]
                    pcenspec_window,hp_spec_window = extract_feature_spectograms(signal, bands, frames, sr)
                    pcen_specgrams_recording.extend(list(pcenspec_window))
                    hp_specgrams_recording.extend(list(hp_spec_window))
            
            print(f'extract_features_speedy: collected audio feature of 232 buckets: {len(pcen_specgrams_recording)}, {len(hp_specgrams_recording)}')
            
            # for each audio signal sub-sample window of data. pop specgrams recording when no longer needed (overlapping frames here)
            
            samples_processed = 0 # done processing
            
            for (start, end) in windows(sound_data, window_size):
                print(f'extract_features_speedy: processing overlapping region: {start}-{end}({end-start} <-> {window_size})')
                
                #start-sample/512 (hop-size) = index in recording-spec-buffer
                #length = windows_size/512 (hop-size) -> frames
                start_index = math.floor(start/512) #every second turn frame should have been cut in half: 31,5, but we'll take roudn to floor. end should have been +64, which will be floor(x,5+64). NExt run shold be perfect integer again -> (x,5+65). ceil could generate overrun also.
                print(f'extract_features_speedy: {frames} vectors from recording from index {start_index} ({start/512})')
                
                if(len(sound_data[start:end]) != window_size):
                    print(f'extract_features_speedy: incomplete frame. data is normally padded with 0. not a full window {len(sound_data[start:end])}')
                
                #check if padded (only if enough data was there)
                if (start_index+frames ) > len(pcen_specgrams_recording):
                    print(f'extract_features_speedy: incomplete frame. data was not padded to add this data as a frame. stopping here. last {{len(sound_data[start:end])}} not represented as features.')
                    break
                
                pcen_specgrams_overlapping.append(pcen_specgrams_recording[start_index:start_index+frames]) #group feature vectors for each window: 64 frames into 1 list, creating a (64,232)-array again. internally stored as sequence in numpy, so doesn't matter
                hp_specgrams_overlapping.append(hp_specgrams_recording[start_index:start_index+frames])
                class_labels.append(class_label)
                audio_samples_info = audio_samples_info.append({'filepath': recording.filepath, 'start': start, 'end': end}, ignore_index=True)# do not store audio-data, only ranges

                #remove all vectors up to index,as we move forward
                
                
                print(f'extract_features_speedy: removing {start_index} frames (audio-features)')
                
                #pcen_specgrams_recording = pcen_specgrams_recording[start_index:]
                #print(f'size 2 {len(pcen_specgrams_overlapping)}')
                #hp_specgrams_recording = hp_specgrams_recording[start_index:]
                
                samples_processed += (start_index*512)
                
                print(f'extract_features_speedy: still {len(pcen_specgrams_recording)}, {len(hp_specgrams_recording)} left')
            
            #for whole data at once, merge values in 4th dimension
            # create the first two feature maps for current file
            pcen_specgrams_overlapping = np.asarray(pcen_specgrams_overlapping)
            print(f'pcen_specgrams_overlapping-shape: {pcen_specgrams_overlapping.shape}')
            pcen_specgrams_overlapping = pcen_specgrams_overlapping.reshape( len(pcen_specgrams_overlapping), bands, frames, 1 ) #not same as original paper, but I like the mapping to frame. why mess up the data? Now, for the processing, it probably doesn't matter, but link to frame is not so linear (transposed). It works remarkably well.
            hp_specgrams_overlapping = np.asarray(hp_specgrams_overlapping).reshape( len(hp_specgrams_overlapping), bands,  frames, 1)
            
            features_cur_run = np.concatenate((pcen_specgrams_overlapping, hp_specgrams_overlapping, np.zeros(np.shape(pcen_specgrams_overlapping))), axis=3)
            # create the third feature map which is the delta (derivative), insteaf of the zero's. pcen-delta between the vectors for each sampling-window. So, we take the values of the next window (1,6/2 seconds also into account a bit. So, for "1,5*window-size" as they overlap -> time shift difference, so window may not be too large.). If we like to have more input (time) into model at once, add more resulting feature-vectors.
            # of the log-scaled mel-spectrogram
            for i in range(len(features_cur_run)):
                features_cur_run[i, :, :, 2] = librosa.feature.delta(features_cur_run[i, :, :, 0])

            #store_recording_audio_features(recording.filepath, features_cur_run, audio_samples_info) # more storage, less computing
    
        #append and clear current run
        if features is None:
            features = features_cur_run.astype(np.float32)
            print(f'extract_features: new features. shape is {features.shape}')
        else:
            # some more mem-copying, but better than running out of memory. can be improved of course to multi-recording
            features = np.concatenate((features, features_cur_run.astype(np.float32)), axis=0)
            print(f'extract_features: concatenated new features. new shape is {features.shape}')
    
    #float32 should be enough. limit memory
    return features, np.array(class_labels, dtype=np.bool_), audio_samples_info

def extract_feature_spectograms(signal, bands, frames, sr=22050):
    '''64 elements with 232 values (frames=64,232) is shape of return vectors for each spectogram'''
    
    # get the log-scaled pcen-spectrogram
    melspec = librosa.feature.melspectrogram(signal, n_mels=bands, sr=sr) #returns shape=(n_mels, t)]. needs transpose to switch rows and columns -> (frames, bands)
    pcenspec = librosa.pcen(melspec, sr, time_constant=0.06, gain=0.8, bias=10, power=0.25, eps=1e-06) #, ref=np.max
    pcenspec = pcenspec * (2**31)
    pcenspec = pcenspec.T
    print(f'extract_feature_spectograms: pcen-spec shape: {pcenspec.shape}')
    
    frames_to_add = None
    if pcenspec.shape[0] > frames:
        print(f'extract_feature_spectograms: logic error shape')
    elif pcenspec.shape[0] < frames:
        frames_to_add = np.zeros((frames-pcenspec.shape[0], pcenspec.shape[1]))
        pcenspec = np.append(pcenspec, frames_to_add, axis=0)
        print(f'extract_feature_spectograms: pcenspec padded shape: {pcenspec.shape}')
        
    #first mels from frame 1, then those of frame 2, ... in a column (2nd dimensio has all the values and 1 element in first dimension)
    #pcenspec = pcenspec.flatten()[:, np.newaxis].T #put vector in sub-dimension (add dimension) This way the code can easily count the number of overlapping spectograms
    
    
    # get the log-scaled, averaged values for the
    # harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(signal)
    melspec_harmonic = librosa.feature.melspectrogram(y_harmonic, n_mels=bands, sr=sr)
    melspec_percussive = librosa.feature.melspectrogram(y_percussive, n_mels=bands, sr=sr)
    logspec_harmonic = librosa.amplitude_to_db(melspec_harmonic)
    logspec_percussive = librosa.amplitude_to_db(melspec_percussive)
    
    logspec_harmonic = logspec_harmonic.T
    print(f'extract_feature_spectograms: logspec_harmonic shape: {logspec_harmonic.shape}')
    logspec_percussive = logspec_percussive.T
    print(f'extract_feature_spectograms: logspec_percussive shape: {logspec_percussive.shape}')
    
    if frames_to_add is not None:
        logspec_harmonic = np.append(logspec_harmonic, frames_to_add, axis=0)
        print(f'extract_feature_spectograms: logspec_harmonic padded shape: {logspec_harmonic.shape}')
        
        logspec_percussive = np.append(logspec_percussive, frames_to_add, axis=0)
        print(f'extract_feature_spectograms: logspec_percussive padded shape: {logspec_percussive.shape}')
        
    logspec_harmonic = logspec_harmonic.flatten()[:,np.newaxis].T
    logspec_percussive = logspec_percussive.flatten()[:, np.newaxis].T
    logspec_hp = np.average([logspec_harmonic, logspec_percussive], axis=0)
    logspec_hp = logspec_hp.reshape(frames, bands) #still frame first order
    print(f'extract_feature_spectograms: hp shape: {logspec_hp.shape}')
    
    return (pcenspec, logspec_hp)

def store_recording_audio_features(filepath: Path, features, audio_samples):
    """Store the processed features for a given filepath, based on audio_samples_info and the features, which are matching in row-index.
    Can also be partially. It's stored in the directory of the recording, same name as the file"""
    save_dir = filepath.parent
    if not save_dir.exists():
        save_dir.mkdir()
    
    if not save_dir.is_dir():
        print(f'store_recording_audio_features: {save_dir} should be a directory. Is a file currently. Please delete file to continue.')
        
    file_for_features = save_dir / (filepath.name + '.pkl')
    if file_for_features.exists():
        print(f'store_recording_audio_features: {file_for_features}: exists already. Overwriting')
        file_for_features.unlink(missing_ok=True)
    
    file_for_audio_samples = save_dir / (filepath.name + '.audio_samples.pkl')
    if file_for_audio_samples.exists():
        print(f'store_recording_audio_features: {file_for_audio_samples}: exists already. Overwriting')
        file_for_audio_samples.unlink(missing_ok=True)
    
    # assuming only complete files for now, which we do
    joblib.dump(features, file_for_features)
    sample_info_for_recording = audio_samples[audio_samples.filepath==filepath]
    #should be same amount of rows
    if sample_info_for_recording.shape[0] != features.shape[0]:
        print(f'store_recording_audio_features: logic error: {file_for_audio_samples}: feature ({features.shape}) and recording ({sample_info_for_recording.shape}) shape do not match!')
    
    joblib.dump(sample_info_for_recording, file_for_audio_samples)
    print(f'store_recording_audio_features: {file_for_features}: saved {features.shape[0]} features')
    
def get_stored_audio_features(filepath: Path):
    """See if audio features where already store, if so load them"""
    save_dir = filepath.parent
    
    feature_file = save_dir / (filepath.name + '.pkl')
    
    if not feature_file.exists():
        return (None, None)
    else:
        print(f'get_stored_audio_features: {filepath}: loading features')
        
    info_file = save_dir / (filepath.name + '.audio_samples.pkl')
    
    if not info_file.exists():
        return (None, None)
    else:
        print(f'get_stored_audio_features: {filepath}: loading info')
    
    features = joblib.load(feature_file)
    if 0 == features.ndim:
        print(f'get_stored_audio_features: {filepath}: error: 0-dimensional. error on storing?')
        return (None, None)
    
    sample_info = joblib.load(info_file)
    
    return (features.astype(np.float32), sample_info)

def recording_has_been_cleaned(bird_code, orig_rec_file_name):
    bird_recordings_dir = cleaned_audio_dir / bird_code
    if (not bird_recordings_dir.exists()) or (not bird_recordings_dir.is_dir()):
        #print(f'get_cleaned_recording: {bird_code}: directory does not exist or is not a directory')
        return False
    
    orig_rec_recordings_dir = cleaned_audio_dir / bird_code / orig_rec_file_name
    if (not orig_rec_recordings_dir.exists()) or (not orig_rec_recordings_dir.is_dir()):
        #print(f'get_cleaned_recording: {bird_code}/{orig_rec_file_name}: directory does not exist or is not a directory')
        return False
    else:
        return True
    
def get_cleaned_recording(bird_code, orig_rec_file_name, filename):
    bird_recordings_dir = cleaned_audio_dir / bird_code
    if (not bird_recordings_dir.exists()) or (not bird_recordings_dir.is_dir()):
        print(f'get_cleaned_recording: {bird_code}: directory does not exist or is not a directory')
        return None
    
    orig_rec_recordings_dir = cleaned_audio_dir / bird_code / orig_rec_file_name
    if (not orig_rec_recordings_dir.exists()) or (not orig_rec_recordings_dir.is_dir()):
        print(f'get_cleaned_recording: {bird_code}/{orig_rec_file_name}: directory does not exist or is not a directory')
        return None
    
    recording_file = orig_rec_recordings_dir / filename
    if not recording_file.exists():
        print(f'get_cleaned_recording: {bird_code}/{orig_rec_file_name}/{filename}: file does not exist')
        return None
    
    #print(f"get_cleaned_recording: found {recording_file}")
    return recording_file


def get_processed_birds():
    '''processed bird, nothing more. not used for labelling'''
    if glob_bird_codes_map is not None:
        bird_codes_map = glob_bird_codes_map
    else:
        bird_codes_map = pd.Series(joblib.load( Path('../input/submission1part3/bird_codes_map.pkl')))

    #manually drop swathr for this test
    #ird_codes_map = bird_codes_map.drop(labels=["swathr"])
    #joblib.dump( bird_codes_map, data_dir / 'bird_codes_map.pkl')
    #done

    bird_codes_map_int = {value: code for code, value in bird_codes_map.items()}
    
    return (bird_codes_map, bird_codes_map_int)

def save_processed_birds(bird_codes_map):
    joblib.dump( bird_codes_map, data_dir / 'bird_codes_map.pkl')
    glob_bird_codes_map = bird_codes_map

def get_traindf():
    #labels for each file
    traindf = pd.read_csv(Path.cwd().parent / "data" / "birdsong-recognition" / "train.csv")


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 5)

    #set sorting
    traindf['bird_seen'] = pd.Categorical(traindf['bird_seen'], ordered=True, categories=['yes','no'])
    #train_birds = traindf.loc[(traindf.secondary_labels == '[]') & (traindf.background.isna())].sort_values(by='rating', ascending=False)

    #add extra columns
    traindf['longitude_simple'] = pd.to_numeric(traindf.longitude, downcast='integer', errors='coerce').fillna(255).astype(np.int16)
    traindf['latitude_simple'] = pd.to_numeric(traindf.latitude, downcast='integer', errors='coerce').fillna(255).astype(np.int16)
    traindf['date_month'] = traindf.date.apply(lambda x: pd.to_numeric(re.search('\d+-(\d+)-\d+', x).group(1), downcast='integer', errors='coerce')).astype(np.uint8)
    traindf['cleaned'] = traindf.apply(lambda row: recording_has_been_cleaned(row.ebird_code, os.path.splitext(row.filename)[0]), axis=1)
    traindf['used_for_model_training'] = False
    traindf['filepath'] = traindf.apply(lambda row: get_traindf_recording(row.ebird_code, row.filename), axis=1) 

    #print(train_birds[['latitude_simple', 'longitude_simple']].drop_duplicates()) # value 255 is invalid
    #print(train_birds['date_month'].unique()) #bad months/no months have value 0
    #train_birds
    
    return traindf

def get_traindf_easy_recs(traindf):
    return traindf.loc[(traindf.secondary_labels == '[]') & (traindf.background.isna())].sort_values(by='rating', ascending=False)

def show_traindf_grouped_quick(traindf):
    '''differentiation in location, time, ...'''
    for bird in traindf.ebird_code.unique():
        print(f"bird {bird}:")
        
        bird_data = traindf.loc[(traindf.ebird_code == bird)]
        
        bird_data = bird_data.sort_values(by=['rating', 'bird_seen', 'duration'], ascending=False)
        print(f'bird {bird}: max entries fullfilling criteria: {bird_data.shape[0]}')
        
        group = bird_data.groupby(['date_month', 'latitude_simple', 'longitude_simple'], group_keys=False)
        bird_data_top = group.sample(15, replace=True).sample(5).drop_duplicates()
        
        
def get_traindf_grouped(traindf):
    '''differentiation in location, time, ...'''      
    bird_data = traindf.sort_values(by=['rating', 'bird_seen', 'duration'], ascending=False)

    group = bird_data.groupby(['ebird_code', 'date_month', 'latitude_simple', 'longitude_simple'], group_keys=False)
    
    return group

def show_traindf_grouped(traindf_grouped):
    bird_data_top = traindf_grouped.sample(15, replace=True).sample(5).drop_duplicates()
    
    

def get_traindf_grouped_samples(traindf, max_duration=300):
    return traindf.apply(lambda s: s[(s.duration.cumsum() <= max_duration)] )

def get_cleaned_traindf(bird_codes_map=None):
    # load processed audio - cleaned audio dir, pandas files to link to original recording.
    # link to original recording using bird code and filename, labels is list for now, which we can on-hot encode later

    #not doing swathr yet
    if bird_codes_map is None:
        bird_codes_map, _ = get_processed_birds()
        

    data = { 
        "ebird_code": [],
        "orig_rec_filename": [],
        "filename": [],
        "labels": []
    }
    
    for bird in bird_codes_map.keys():
        if bird in ['other']:
            print(f'skipping {bird}')
            continue
        bird_dir = cleaned_audio_dir / bird
        print(f'processing {bird}')

        if not bird_dir.exists():
            print(f"Dir {bird_dir} does not exist")
            continue

        name_parser = re.compile(r'(\S+)-\d+.wav|(\S+).wav|(\S+)\s+') #labels are separated by space. audacity adds an integer, in order, to identical file-names (combinations).
        #recordings are stored in a directory which has the same name as the original recording's filename
        for rec_dir in bird_dir.iterdir():
            if rec_dir.is_dir():
                rec_audio_files = rec_dir.glob('*.wav')
                #print(f'{bird}: recording {rec_dir.name}: processing')
                #print([x.name for x in rec_audio_files])
                #labels for each file
                for rec_file in rec_audio_files:
                    name_parsed = name_parser.findall(rec_file.name)
                    rec_labels = [t[0]+t[1]+t[2]  for t in name_parsed]
                    data['ebird_code'].append(bird)
                    data['orig_rec_filename'].append(rec_dir)
                    data['filename'].append(rec_file.name)
                    data['labels'].append(rec_labels) #list-object of label-names

    cleaned_traindf = pd.DataFrame(data)
    
    cleaned_traindf['duration'] = cleaned_traindf.apply(lambda row: AudioSegment.from_wav(get_cleaned_recording(row.ebird_code, row.orig_rec_filename, row.filename)).duration_seconds, axis=1)
    
    cleaned_traindf['purpose'] = '' #training,validation,test -> need to keep track what was fed to model already
    cleaned_traindf['from_last_test_run'] = False
    #load recordings evaluated in last test run
    try:
        test_recs = joblib.load( Path.cwd() / "test_recs.pkl")
        for label in test_recs:
            for orig_rec_name in test_recs[label]:
                cleaned_traindf.loc[(cleaned_traindf.ebird_code == label) & (cleaned_traindf.orig_rec_filename == (cleaned_audio_dir / label / orig_rec_name)), 'from_last_test_run'] = True
    except:
        print(f'unable to find test_recs.pkl')
    
    return cleaned_traindf

def get_cleaned_recs_other(cleaned_traindf):
    other_sounds = cleaned_traindf.loc[(cleaned_traindf.labels.apply( lambda list: 1 == len(list) and re.match(r'other', list[0], flags=re.IGNORECASE) is not None ))]
    
    return other_sounds

#unknown birds too
def get_cleaned_recs_bird(cleaned_traindf):
    other_sounds = cleaned_traindf.loc[(cleaned_traindf.labels.apply( lambda list: 1 == len(list) and re.match(r'bird', list[0], flags=re.IGNORECASE) is not None ))]
    
    return other_sounds

def get_cleaned_recs_bird_ok(cleaned_traindf):
    """the samples for the birds which identified the bird correctly, without a lot of other (loud) sounds"""
    cleaned_process = cleaned_traindf.loc[(cleaned_traindf.labels.apply(lambda list: 'ok' in list or 'ok_with_noise' in list))]
    
    return cleaned_process

def get_traindf_samples_per_group(cleaned_traindf, max_duration=500):
    '''group by bird and get some samples for each'''
    #shuffle all rows first, so we get different entries on each run
    cleaned_traindf = cleaned_traindf.sample(frac=1)
    cleaned_process_grouped = cleaned_traindf.groupby('ebird_code', group_keys=False)
    cleaned_process_sample = cleaned_process_grouped.apply(lambda s: s[(s.duration.cumsum() <= max_duration)] )
    return cleaned_process_sample

def cleaned_recs_replace_label_with_bird_code(cleaned_traindf, bird_codes_map=get_processed_birds()[0]):
    '''map labels used in audacity/audio file tagging and map them to our datamodel. Based on birds_codes_map to integer mapping'''
    
    #map labels used in audacity/audio file tagging and map them to our datamodel

    #unknown birds in other for now. limit scope for time.
    map_dict = defaultdict(lambda: 'other', {'ok': 'ok', 'ok_with_noise': 'ok_with_noise', 'bird': 'other'} )

    #joblib.dump(map_dict, Path.cwd() / "map_dict.pkl")

    for bird in bird_codes_map.keys():
        map_dict.update({'ok_with_noise': bird, 'ok': bird })
        map_dict[bird] = bird #in reruns, do not corrupt data
        print(map_dict)
        bird_rows = cleaned_traindf.loc[(cleaned_traindf.ebird_code == bird)]
        #
        #
        
        cleaned_traindf.loc[bird_rows.index, 'labels'] = bird_rows.loc[:,'labels'].apply(lambda x: [map_dict[i] for i in x])
        #if it contains the bird-label, set also general bird-label, if nto set yet
        #bird_df = cleaned_traindf.loc[bird_rows.index]
        #model can not support this generalising for now
        #bird_df = bird_df.loc[bird_df.loc[(bird_df.labels.apply(lambda a: (bird in a) and ('bird' not in a)))].index]
        #add_bird = cleaned_traindf.loc[bird_rows.loc[(bird_rows.labels.apply(lambda a: (bird in a) and ('bird' not in a)))].index]
        #print(bird_df)
        #bird_df.labels.apply(lambda x: x.append('bird'))

        map_dict.pop(bird)
        
    return cleaned_traindf

def bird_codes_map_add(bird_codes_map, label):
    """only add if it does not exist yet. returns reverse map"""
    if bird_codes_map.get(label) is None:
        print(f'adding other {label}')
        bird_codes_map[label] =  len(bird_codes_map)
        joblib.dump( bird_codes_map, data_dir / 'bird_codes_map.pkl')
        
    else:
        print(f'label {label} exists')
        
    bird_codes_map_int = {value: code for code, value in bird_codes_map.items()}
    return bird_codes_map_int

def cleaned_recs_show_samples(cleaned_traindf, samples_per_cat=2):
    '''show some visualisations for the different categories'''
    
    cleaned_process_sample_temp = cleaned_traindf.sample(samples_per_cat, replace=True).drop_duplicates(subset=['ebird_code', 'orig_rec_filename', 'filename'])
    for index, (ebird_code, orig_rec_filename, filename, labels, duration) in cleaned_process_sample_temp.iterrows():    
        rec = get_cleaned_recording(ebird_code, orig_rec_filename, filename)
        snd_data, sr = get_audio_data(rec, sr)
        #snd_data_pcen = pcen(snd_data, sr, time_constant=0.06, gain=0.8, bias=10, power=0.25)
        print(f'bird: {ebird_code}, file: {filename}: first=orig, second=pcen')

        fig = plt.figure(figsize=(15, 6))
        stft = librosa.stft(snd_data)
        log_stft = librosa.amplitude_to_db(stft)
        plt.subplot(1, 3, 1)
        librosa.display.specshow(log_stft, sr=sr, x_axis='time', y_axis='linear') 
        plt.title("stft spectogram") 
        plt.colorbar(format='%+02.0f dB') 

        S = librosa.feature.melspectrogram(snd_data, sr=sr,n_mels=128)

        # https://musicinformationretrieval.com/magnitude_scaling.html
        plt.subplot(1, 3, 2)
        log_S = librosa.amplitude_to_db(S) 
        librosa.display.specshow(log_S, sr=sr, x_axis='time',y_axis='mel') 
        plt.title("mel(ody) spectogram") 
        plt.colorbar(format='%+02.0f dB') 

        #pcen-version
        plt.subplot(1, 3, 3)
        S_pcen = pcen(S, sr, time_constant=0.06, gain=0.8, bias=10, power=0.25, eps=1e-06)
        librosa.display.specshow(S_pcen * (2**31), sr=sr, x_axis='time',y_axis='mel') 
        plt.title("pcen spectogram") 
        plt.colorbar(format='%+02.0f dB')
        plt.show()

mlb_df_store = Path.cwd() / "mlb_df_columns.pkl"
mlb_df_store_ok_model = Path('../input/submission-1/mlb_df_columns_ok_model.pkl')
        
def get_mlbdf_columns():
    mlb_df_columns = None
    mystore = None
    if use_test_model:
        mystore = mlb_df_store
    else:
        mystore = mlb_df_store_ok_model
        
    if mystore.exists():
        mlb_df_columns = joblib.load(mystore)
        
    return mlb_df_columns
    
def cleaned_recs_get_mlbdf(cleaned_traindf):
    '''accross runs, the order of the final feature-vector should be consistent. each bird is a boolean-location in the vector. Each feature can contain multiple sound/classes'''
    #create a class for each possible label (multilabel possible for an audiopiece)
    
    mlb_df_columns = []
    mlb_df = None
    
    #load available labels in dataset
    data_labels = set()
    cleaned_traindf.labels.apply(lambda x: data_labels.update(x))

    #load existing labels and check if we need to add any
    mlb_df_columns = get_mlbdf_columns()
    if mlb_df_columns is not None:
        joblib.dump(mlb_df_columns, Path.cwd() / "mlb_df_columns.pkl.backup") #just to be sure
        print(f'cleaned_recs_get_mlbdf: loaded mlb: {mlb_df_columns}')
        model_set = set(mlb_df_columns)
        #extend columns if needed
        data_labels_included = data_labels.issubset(model_set)
        print(f'Labels of new data already present in model: {data_labels_included}')

        new_labels = None

        if not data_labels_included:
            new_labels = data_labels.difference(model_set)
            print(f'new labels in new dataset, to be added to model: ' + ", ".join(new_labels))
            mlb_df_columns = list(mlb_df_columns)
            mlb_df_columns.extend(list(new_labels))
            print(f'extended columns: {mlb_df_columns}')
    else:
        print(f'no previous labels known')

    #mlb_df_columns =    ['balori', 'fiespa', 'sora', 'other'] #this is the order
    print(f'check if order is correct!')
    print(f'mlb columns: {mlb_df_columns}')
    mlb = MultiLabelBinarizer(classes=mlb_df_columns)
    mlb_df = pd.DataFrame(mlb.fit_transform(cleaned_traindf.labels), columns=mlb.classes_)


     #the ordering of the columns needs to stay consistent across runs for our model
    
    mystore = None
    if use_test_model:
        mystore = mlb_df_store
    else:
        mystore = mlb_df_store_ok_model
    
    joblib.dump(mlb_df_columns, mystore)
    
    return mlb_df

def cleaned_recs_to_audio_feature_extraction_input(cleaned_traindf, mlb_df):
    '''df to input for feature extraction. mlb_df is the multilabel matrix for each cleaned_traindf-row. row-ordering should match'''
    #label is a tuple/vector 
    cleaned_process_extract_info = pd.DataFrame({"filepath": cleaned_traindf.apply(lambda row: get_cleaned_recording(row.ebird_code, row.orig_rec_filename, row.filename), axis=1), "label": list(mlb_df.itertuples(index=False, name=None))})
    # if label, was ok, then the label is the bird
    
    
    return cleaned_process_extract_info

def audio_features_get_weighted_samples_per_label(mlb_df, features, labels, audio_info, indices_used=None):
    '''we just take the minimum one for now. Features will be augmented a bit, so small difference is allowed'''
    labels_df = pd.DataFrame(labels, columns=mlb_df.columns.values)

    min_count = 0
    min_code = None
    
    max_count = 0
    
    rows_for_bird = {}
    
    for code in labels_df.columns:
        rows_for_bird_found = labels_df.loc[labels_df[code]==True]
        rows_for_bird[code] = feature_count = rows_for_bird_found.shape[0]
        print(f'label {code}: #features: {feature_count}')
        
        if min_count != 0:
            min_count = min(min_count, feature_count)
            min_code = code
        else:
            min_count = feature_count
            min_code = code
            
        if max_count < feature_count:
            max_count = feature_count

    select_count = int(min_count + (max_count-min_count)*0.5) #we have data augmentation
    print(f'min: {min_count}, max count: {max_count}, select count: {select_count}')
    
    if indices_used is not None:
        indices_for_bird_to_be_deleted = []
        #drop indices if used in previous run
        for code in rows_for_bird:
            #want ot have select_count left
            nr_of_may_be_deleted = rows_for_bird[code] - min_count
            print(f'{nr_of_may_be_deleted} rows for {code} may be deleted')
            
            labels_df_used = labels_df.loc[indices_used]
            #do not delete for the code with the least samples.
            indices_for_bird_which_may_be_deleted = labels_df_used.loc[(labels_df_used[code]==True) & (labels_df_used[min_code]==False)].index
            indices_to_be_deleted_here = indices_for_bird_which_may_be_deleted[:nr_of_may_be_deleted]
            indices_for_bird_to_be_deleted.extend(indices_to_be_deleted_here)
            print(f'indices_for_bird_to_be_deleted: {indices_to_be_deleted_here}')
            
        audio_info.drop(indices_for_bird_to_be_deleted, inplace=True)
        audio_info.reset_index(drop=True, inplace=True)
        labels = np.delete(labels, indices_for_bird_to_be_deleted, axis=0)
        labels_df = pd.DataFrame(labels, columns=mlb_df.columns.values)
        features = np.delete(features, indices_for_bird_to_be_deleted, axis=0)
        if (features.shape[0] == audio_info.shape[0]) and (features.shape[0] == labels.shape[0]):
              print(f'delete ok: {features.shape}')
        else:
              print(f'audio_features_get_weighted_samples_per_label: logic issue deleting indices: feature shapes: {features.shape}, audio info shape: {audio_info.shape}, labels shape: {labels.shape}, indices: {indices_for_bird_to_be_deleted}') 

    #allow-upsampling
    
    indices = []

    #item can have multiple labels -> issue still
    for code in labels_df.columns:
        #increase rows to select_count
        rows_for_bird = labels_df[labels_df[code]==True]    
        missing_nr_rows = max(0, min_count - len(rows_for_bird))
        if missing_nr_rows > 0:
            print(f'missing rows for {code}: {missing_nr_rows}')
        
        indices.extend(rows_for_bird.sample(max(0,min_count - missing_nr_rows)).index) # a bit uneven maybe, but better than nothing. it's the multi-label thing. augmentatin helps hopefully

    
    labels_weighted = labels[indices]
    features_weighted = features[indices]
    audio_info_weighted = audio_info.loc[indices]

    
    
    #
    
    #modified features if indices_used is not none. others are in place
    return (features_weighted, labels_weighted, audio_info_weighted, features, labels, indices)

def get_traindf_recording(bird_code, filename):
    '''for traindf'''
    train_data_dir = project_dir / "data" / "birdsong-recognition" / "train_audio"
    bird_recordings_dir = train_data_dir / bird_code
    if (not bird_recordings_dir.exists()) or (not bird_recordings_dir.is_dir()):
        print(f'get_recording: {bird_code}/{filename}: directory does not exist or is not a directory')
        return None
    recording_file = bird_recordings_dir / filename
    if not recording_file.exists():
        print(f'get_recording: {bird_code}/{filename}: file does not exist')
        return None
    return recording_file

def get_last_model(predict=True):
    '''for predictions by default: trainable off'''
    if use_test_model:
        model = tf.keras.models.load_model(Path.cwd() / 'test_model.model')
    else:
        model = tf.keras.models.load_model(Path('../input/model1h5/classify_model.h5'))
    
    if predict:
        model.trainable=False
    else:
        model.trainable=False
        for layer in model.layers:
            if not layer.name.startswith('block') and not layer.name.startswith('input'):
                layer.trainable = True
    
    for layer in model.layers:
        if layer.trainable:
            print(f'{layer.name}: {layer.trainable}')
            
    model.summary() 
    return model

def save_model(model):
    if use_test_model:
        model = tf.keras.models.save_model(model, Path.cwd() / 'test_model.model')
    else:
        model = tf.keras.models.save_model(model, Path.cwd().parent / "train_sessions" / "4" / "classify_model.model")

def compile_model(model):
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.95)
    #nadam = tf.keras.optimizers.Nadam(clipvalue=0.95) #tf.keras.optimizers.Adagrad()
    adamax = tf.keras.optimizers.Adamax(clipvalue=0.99)
    optim = adamax

    #https://keras.io/api/metrics/accuracy_metrics/#binaryaccuracy-class
    model.compile(loss='mean_absolute_error', #percentage? mean_absolute_percentage_error, mean_absolute_error
                  optimizer=optim,metrics=[tf.keras.metrics.BinaryAccuracy()]) 
    
    return model

def retrain_model_audio_feature_input(mlb_df, features, labels, audio_info, cleaned_df, epochs=100,model=None):
    print(f'using mlb_df: {mlb_df.head(5)}')
    
    if model is None:
        model = get_last_model(predict=False)
        
    compile_model(model)
    
    

    features_weighted, labels_weighted, audio_info_weighted, features, labels, indices_used = audio_features_get_weighted_samples_per_label(mlb_df, features, labels, audio_info)
    
    print(f'audio_info: {audio_info_weighted.shape}')
    print(f'features: {features_weighted.shape}')
    print(f'labels: {labels_weighted.shape}')
    
    folds = math.ceil(audio_info.shape[0]/audio_info_weighted.shape[0])
    #folds = 2
    cur_fold = 0
    
    last_history = None
    
    while (cur_fold < folds) and (features_weighted.shape[0] > 10):
        cur_fold += 1
        print(f'fold: {cur_fold}/{folds}')
        
        feature_data = np.array(list(zip(features_weighted, labels_weighted, range(0,features_weighted.shape[0]))))

        #extract correct train, test, validate sets, so we limit extraction-time
        rng = np.random.default_rng()
        rng.shuffle(feature_data)
        train, validate, test = np.split(feature_data, [int(.6*len(feature_data)), int(.8*len(feature_data))])
        print(f'train shape: {train.shape}, val shape: {validate.shape}, test shape: {test.shape}')

        #extract for all sets
        # extract train dataset features 
        train_base_features = np.array([item[0] for item in train])
        train_labels = np.array([item[1] for item in train])
        train_orig_index = np.array([item[2] for item in train])

        # extract validation dataset features 
        validate_base_features = np.array([item[0] for item in validate])
        validate_labels = np.array([item[1] for item in validate]) 
        validate_orig_index = np.array([item[2] for item in validate]) 

        # extract test dataset features 
        test_base_features = np.array([item[0] for item in test])
        test_labels = np.array([item[1] for item in test])
        test_orig_index = np.array([item[2] for item in test]) #link to original sample. the original index for sample at location within this index

        print(f'{train_base_features.shape} label shape: {train_labels.shape}')

        labels_df_train = pd.DataFrame(train_labels, columns=mlb_df.columns.values)

        for code in labels_df_train.columns:
            rows_for_bird = labels_df_train[labels_df_train[code]==True]    
            print(f'train: {code}: #features: {rows_for_bird.shape[0]}')

        labels_df_val = pd.DataFrame(validate_labels, columns=mlb_df.columns.values)

        for code in labels_df_val.columns:
            rows_for_bird = labels_df_val[labels_df_val[code]==True]    
            print(f'val: {code}: #features: {rows_for_bird.shape[0]}')

        last_history = train_model(model, train_base_features, train_labels, validate_base_features, validate_labels, epochs)
        
        stats_on_test_data(model, test_base_features, test_labels)
              
        print(f'get features for next fold:')
        train_features_orig_indices = train_orig_index[:]
        print(f'train used for test: {train_features_orig_indices}')
        mark_rec_used_as(cleaned_df, audio_info_weighted.iloc[train_features_orig_indices].filepath, 'training') #audio_info.iloc[test_orig_index[false_negatives.index]]
        
        validate_features_orig_indices = validate_orig_index[:]
        print(f'train used for test: {validate_features_orig_indices}')
        mark_rec_used_as(cleaned_df, audio_info_weighted.iloc[validate_features_orig_indices].filepath, 'validation') #audio_info.iloc[test_orig_index[false_negatives.index]]
        
        test_features_orig_indices = test_orig_index[:]
        print(f'features used for test: {test_features_orig_indices}')
        mark_rec_used_as(cleaned_df, audio_info_weighted.iloc[test_features_orig_indices].filepath, 'test') #audio_info.iloc[test_orig_index[false_negatives.index]]
        
        features_weighted, labels_weighted, audio_info_weighted, features, labels, indices_used = audio_features_get_weighted_samples_per_label(mlb_df, features, labels, audio_info, indices_used=indices_used)        
        #using test features this way is not so good at the moment    
        
    return (model, last_history)
              
def mark_rec_used_as(cleaned_df, filepaths, purpose):
    if purpose not in ('training', 'validation', 'test'):
        print(f'mark_rec_used_as: invalid purpose: {purpose}')
        return
    
    #improve: if used for training, should never be changed again
    count = 0
    count_skipped = 0
    for path in filepaths:
        cleaned_df.loc[(cleaned_df.orig_rec_filename == path.parent) & (cleaned_df.filename==path.name)].purpose = purpose
        count += 1
    
    print(f'mark_rec_used_as: marked {count} recs for {purpose}')

def train_model(model, train_features, train_labels, validate_features, validate_labels, epochs=100):
    

    for layer in model.layers:
        print(f'{layer.name}: {layer.trainable}')
            
    if use_test_model:
        model_save_path = Path.cwd() / 'test_model.model'
    else:
        model_save_path = project_dir / "train_sessions" / "4" / "a" / "classify_model.model"
        
    #started with batch size 16
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='min', restore_best_weights=True)
    model_save = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=25, verbose=1, mode='min')

    #data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.2,
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.1,
       fill_mode='nearest')

    datagen.fit(train_features)


    # train the classify-model to learn the birds
    history = model.fit(datagen.flow(train_features, train_labels, batch_size=16),steps_per_epoch=len(train_features) / 16,epochs=epochs, 
                        batch_size=16, 
                        validation_data=(validate_features,  
                        validate_labels),shuffle=True, verbose=1, callbacks=[ model_save, reduce_lr_loss, early_stopping])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)) 
    t = f.suptitle('Deep Neural Net Performance', fontsize=12) 
    f.subplots_adjust(top=0.85, wspace=0.2) 
    epochs = list(range(1,len(history.epoch)+1)) 
    ax1.plot(epochs, history.history['binary_accuracy'], label='Train Accuracy') 
    ax1.plot(epochs, history.history['val_binary_accuracy'], label='Validation Accuracy') 
    ax1.set_ylabel('Accuracy Value') 
    ax1.set_xlabel('Epoch') 
    ax1.set_title('Accuracy') 
    plt.ylim(0.2, 1)
    l1 = ax1.legend(loc="best") 
    ax2.plot(epochs, history.history['loss'], label='Train Loss') 
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss') 
    ax2.set_ylabel('Loss Value') 
    ax2.set_xlabel('Epoch') 
    ax2.set_title('Loss') 
    l2 = ax2.legend(loc="best") 
    plt.ylim(0, 20)
    plt.show()

    return history
              

def confusion_per_label(y_pred, y_true, labels):
    """input are multi-label matrices"""
    conf_matrix_per_label = {}

    for label_index in range(len(labels)):
        y_true_for_label = y_true[:, label_index]
        y_pred_for_label = y_pred[:, label_index]
        conf_matrix_per_label[labels[label_index]] = confusion_matrix(y_pred=y_pred_for_label, y_true=y_true_for_label, labels=[True, False])

    return conf_matrix_per_label
              
def stats_on_test_data(model, test_features, test_labels):

    # return best class for each feature-input

    mlb_df_columns = get_mlbdf_columns()

    predictions = model.predict(test_features)
    print(predictions.shape)
    predictions[predictions>=0.5] = True
    predictions[predictions<0.5] = False

    predictions = predictions.astype(np.bool_)
    predictions_df = pd.DataFrame(data=predictions, columns=mlb_df_columns)
    test_labels_df = pd.DataFrame(data=test_labels, columns=mlb_df_columns)

    
     

    print("accuracy_score:", accuracy_score(test_labels, predictions))
    print("Hamming_loss:", hamming_loss(test_labels, predictions))



    trues_per_test_sample = test_labels_df.sum(axis=1)
    print(f'find invalid samples:')
    

    #metrics: https://towardsdatascience.com/taking-the-confusion-out-of-confusion-matrices-c1ce054b3d3e

    label_confusion = confusion_per_label(predictions, test_labels, mlb_df_columns)
    label_confusion

    count = 1
    for label in label_confusion:
        sum_diagonal = np.trace(label_confusion[label])
        sum_total = np.sum(label_confusion[label])
        accuracy = sum_diagonal/sum_total
        misclassification = 1-accuracy
        precision = label_confusion[label][0,0]/np.sum(label_confusion[label][:,0]) #true positive/predicted positives
        recall = label_confusion[label][0,0] / np.sum(label_confusion[label][0,:])
        print(f'{label}')
        print(f'accuracy: {accuracy:.2f}')
        print(f'misclassification: {misclassification:.2f}')
        print(f'precision: {precision:.2f}')
        print(f'recall: {recall:.2f}')
        fig = plt.subplot(len(label_confusion),1,count)
        sns.heatmap(label_confusion[label], annot=True, ax = fig, fmt='g', cmap='Greens')

        # labels, title and ticks
        fig.set_xlabel('Predicted labels')
        fig.set_ylabel('True labels')
        fig.set_title(f'Confusion Matrix: {label}')
        fig.xaxis.set_ticklabels(["True", "False"])
        fig.yaxis.set_ticklabels(["True", "False"])
        count += 1

        plt.show()

    print(f'totals:')
    total_confusion = np.zeros((2,2))

    for label in label_confusion:
        total_confusion = total_confusion + label_confusion[label]

    fig = plt.subplot(1,1,1)
    sns.heatmap(total_confusion, annot=True, ax = fig, fmt='g', cmap='Greens')

    # labels, title and ticks
    fig.set_xlabel('Predicted labels')
    fig.set_ylabel('True labels')
    fig.set_title(f'Confusion Matrix: {label}')
    fig.xaxis.set_ticklabels(["True", "False"])
    fig.yaxis.set_ticklabels(["True", "False"])
    plt.show()

    sum_diagonal = np.trace(total_confusion)
    sum_total = np.sum(total_confusion)
    accuracy = sum_diagonal/sum_total
    misclassification = 1-accuracy
    precision = total_confusion[0,0]/np.sum(total_confusion[:,0]) #true positive/predicted positives
    recall = total_confusion[0,0] / np.sum(total_confusion[0,:])
    print(f'total:')
    print(f'accuracy: {accuracy:.2f}')
    print(f'misclassification: {misclassification:.2f}')
    print(f'precision: {precision:.2f}')
    print(f'recall: {recall:.2f}')

    

def make_prediction_from_files (filepaths, mlb_df_columns=get_mlbdf_columns()):
    '''the extract features pandas dataframe'''
    #unkown labels (all 0)
    audio_feature_input = pd.DataFrame({"filepath": filepaths, "label": list(tuple(np.zeros((1,len(mlb_columns)), dtype=np.bool_)))*birds_to_process.shape[0]})
    return make_prediction_from_df(audio_feature_input, mlb_df_columns)

def make_prediction_from_df (audio_feature_input, mlb_df_columns=get_mlbdf_columns()):
    '''the extract features pandas dataframe-input
    predictions-dataframe gives the prediction for each feature vector. feature-vectors overlap 1/2 if you check the audio_info'''
    features, labels, audio_info = extract_features_speedy(audio_feature_input)
    
    #predict with model
    #map to audio
    predictions = model.predict(features)
    print(predictions.shape)
    predictions[predictions>=0.5] = True
    predictions[predictions<0.5] = False

    predictions = predictions.astype(np.bool_)
    predictions_df = pd.DataFrame(data=predictions, columns=mlb_df_columns)
    
    return (predictions_df, (features, labels, audio_info))

#select bird which has been unprocessed yet
def get_unprocessed_bird(traindf):
    processed_birds = set(traindf.loc[(traindf.cleaned==True)].ebird_code.unique())
    all_birds = set(traindf.ebird_code.unique())
    return (all_birds - processed_birds)

def add_bird_to_model(is_bird=True, bird_code=None):
    '''should be label and normally is bird'''
    label_to_add = None
    if bird_code is None:
        traindf = get_traindf()
        unprocessed_birds = get_unprocessed_bird(traindf)

        if len(unprocessed_birds) > 0:
            label_to_add = unprocessed_birds.pop()
        else:
            print(f'all birds processed (cleaned). none uncleaned.')
            return (None, None)
    else:
        label_to_add = bird_code
    
    print(f'add_bird_to_model: adding label {label_to_add}')
    
    if is_bird:
        #add "bird" to bird_codes_map, which should add it to the matrix-columns mlb
        processed_birds_map, _ = get_processed_birds()
        
        if label_to_add not in processed_birds_map:
            bird_codes_map_add(processed_birds_map, label_to_add)
            save_processed_birds(processed_birds_map)
        
        
        
    
    
    model = get_last_model(predict=False)
    
    output_layer = model.get_layer(index=-1)
    
    
    
    weights = output_layer.get_weights()
    bias = output_layer.bias

    #first elements in weights are the .. weights, second the bias
    #https://keras.io/api/layers/base_layer/#get_weights-method
    print('shape')
    
    
    #so, in the second dimension, we need to add a weight-values for each of the 256 incoming connections
    #initialize to 0 as there should be "no" match initially. could also do slightly random around 0
    #mu, sigma = 0, 0.0001 # mean and standard deviation
    #weights_neuron = np.random.normal(mu, sigma, (weights[0].shape[0],1))
    #average all weights in the second dimension
    weights_neuron = np.average(weights[0], axis=1)
    weights_neuron = weights_neuron[:,np.newaxis]
    
    
    new_weights = np.concatenate((weights[0], weights_neuron), axis=1) # change to this for weights: initializers.RandomNormal(stddev=0.01)
    new_bias = np.concatenate((weights[1], np.zeros((1,))), axis=0)

    new_output = model.layers[-2].output
    
    #extend last layer
    new_output = Dense(output_layer.output_shape[1]+1, name='output_layer')(new_output)
    updated_model = Model(inputs=model.input, outputs=new_output)
    new_output_layer = updated_model.get_layer(name='output_layer')
    new_output_layer.set_weights([new_weights, new_bias])
    model = updated_model

    model.trainable=False
    for layer in model.layers:
        if not layer.name.startswith('block') and not layer.name.startswith('input'):
            layer.trainable = True
            print(f'{layer.name}: {layer.trainable}')

    model = compile_model(model)
    
    
    return (model, label_to_add)

def train_model_with_new_samples(model, label=''):
    '''train on new bird with test_recs.pkl into account (cleaning samples-phase)'''
    global use_test_model
    use_test_model=True #for extending
    
    print(f'train_model: start for {label}')
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    
    cleaned_df = get_cleaned_traindf()
    
    #the birds
    birds_df = get_cleaned_recs_bird_ok(cleaned_df)
    
    #limit recs we're going to process to features
    birds_df_sampled = get_traindf_samples_per_group(birds_df)
    
    
    
    
    other_df = get_cleaned_recs_other(cleaned_df)
    
    
    unknown_birds = get_cleaned_recs_bird(cleaned_df)
    
    
    recs_to_process = birds_df_sampled.append(other_df, ignore_index=True).append(unknown_birds, ignore_index=True)
    
    
    
    
    
    extra_recs_to_add = cleaned_df[(cleaned_df.from_last_test_run==True) & (cleaned_df.labels.apply(lambda x: ('ok' in x or 'bird' in x)))]
    recs_to_process = recs_to_process.append(extra_recs_to_add, ignore_index=True) #make sure testrun-stuff is inside. easy way
    
    #convert labels
    recs_to_process = cleaned_recs_replace_label_with_bird_code(recs_to_process)
    
    mlb_df = cleaned_recs_get_mlbdf(recs_to_process)
    
    
    print(f'start audio feature extraction:')
    feature_input = cleaned_recs_to_audio_feature_extraction_input(recs_to_process, mlb_df)
    features, labels, audio_info = extract_features_speedy(feature_input)
    print(f'ended audio feature extraction')
    
    #save for reruns
    joblib.dump(audio_info, Path.cwd() / "audio_info.pkl")
    joblib.dump(features, Path.cwd()/ "features.pkl")
    joblib.dump(labels, Path.cwd() / "labels.pkl")
    
    #audio_info = joblib.load(Path.cwd() / "audio_info.pkl")
    #features = joblib.load(Path.cwd()/ "features.pkl")
    #labels = joblib.load(Path.cwd() / "labels.pkl")
    print(f'start training model:')
    model, history = retrain_model_audio_feature_input(mlb_df, features, labels, audio_info, cleaned_df, epochs=20, model=model)
    print(f'training ended for label {label}')
    
    return (model, history)
    
def train_model_with_new_samples_multiple_times(model, label=''):
    '''not 100% correct, although I keep track what was used for training'''
    
    #for i in range(2):
    model, history = train_model_with_new_samples(model, label)
    

def get_artifical_samples_for_bird(new_bird = None, nr_of_recs_to_process=15, judge=False):
    '''delete first test_recs.pkl
    ifnew_bird = None, we get more samples for the birds in the model'''
    global use_test_model
    use_test_model=True #for extending
    
    test_recs_file = Path.cwd() / "test_recs.pkl"
    
    if test_recs_file.exists():
        print('deleting {test_recs_file.name} for getting the next samples...')
    else:
        print('{test_recs_file.name} does not exist. strange ...')
    
    test_recs_file.unlink(missing_ok=True)
    
    #labels for each file
    traindf = get_traindf()
    
    #birds_codes_map is not used anymore for labelling, just an overview of processed birds. and we actually keep track of that now with the presence of a "cleaned"-dir for the bird
    #this is quicker though
    bird_map, bird_map_int = get_processed_birds()
    
    
    #get processed birds, incluing new label!
    birds_df = traindf.loc[traindf.ebird_code.apply(lambda x: x in bird_map)]
    
    #get uncleaned recordings
    birds_recs_unprocessed_df = birds_df.loc[birds_df.cleaned==False]
    
    #if new bird, filter on new bird
    if new_bird is not None:
        birds_recs_unprocessed_df = birds_recs_unprocessed_df.loc[birds_recs_unprocessed_df.ebird_code==new_bird]
    
    birds_recs_unprocessed_df_easy = get_traindf_easy_recs(birds_recs_unprocessed_df)
    
    birds_recs_unprocessed_df_easy_g = get_traindf_grouped(birds_recs_unprocessed_df_easy)
    show_traindf_grouped(birds_recs_unprocessed_df_easy_g)
    
    print(f'some unprocessed recordings for the current birds:')
    birds_to_sample = birds_recs_unprocessed_df_easy_g_sampled = get_traindf_grouped_samples(birds_recs_unprocessed_df_easy_g)
    #
    
    mlb_columns = get_mlbdf_columns()
    
    
    if new_bird is None:
        #make sure they are in the model
        birds_to_sample = birds_recs_unprocessed_df_easy_g_sampled_in_model = birds_recs_unprocessed_df_easy_g_sampled[birds_recs_unprocessed_df_easy_g_sampled.ebird_code.apply(lambda bird: bird in mlb_columns)]
    
    print(f'sampling these birds:')
    
    
    if birds_to_sample.shape[0] < nr_of_recs_to_process:
        print(f'only found {birds_to_sample.shape[0]} recordings instead of {nr_of_recs_to_process}.')
        nr_of_recs_to_process = birds_to_sample.shape[0]
        if nr_of_recs_to_process <= 0:
            print(f'no records found! CHECK')
            return
        
    birds_to_process = birds_to_sample.sample(nr_of_recs_to_process)
    predict_recs(birds_to_process.filepath, new_bird=new_bird, manual_judge=judge)
    
              
def predict_recs(recs_list, manual_judge=False, new_bird=None, split_the_predictions=True):
    recs_list = list(recs_list)
    
              
    #predicting
    model = get_last_model()
    
              
    audio_feature_input = get_audio_feature_extraction_input_pd()
    mlb_columns = get_mlbdf_columns()
              
    audio_feature_input = pd.DataFrame({"filepath": recs_list, "label": list(tuple(np.zeros((1,len(mlb_columns)), dtype=np.bool_)))*len(recs_list)})
    
    
    features, labels, audio_info = extract_features_speedy(audio_feature_input)
    #predict with model
    #map to audio
    predictions = model.predict(features)
    print(predictions.shape)
    predictions_simple = np.copy(predictions)
    predictions_simple[predictions_simple>=0.5] = True
    predictions_simple[predictions_simple<0.5] = False

    predictions_simple = predictions_simple.astype(np.bool_)
    
    predictions_df_more = pd.DataFrame(data=predictions, columns=mlb_columns)
    predictions_df_more[:10]
              
    predictions_df = pd.DataFrame(data=predictions_simple, columns=mlb_columns)
    pd.set_option('display.max_rows', None)
    
              
    #relate to audio
    audio_info_predictions = pd.concat([audio_info,predictions_df], axis=1)
    
    #just predict
    if split_the_predictions:
        split_predictions(audio_info_predictions, mlb_columns, manual_judge, new_bird)
    
    return audio_info_predictions
    
def split_predictions(audio_info_predictions, mlb_columns, display_gui=False, new_bird=None):
    '''new_bird required when display_gui=False'''
    from ipywidgets import GridspecLayout, Output
    
    if display_gui:
        grid = GridspecLayout(audio_info_predictions.shape[0], 2)
    
    import soundfile as sf

    audio_data = []
    last_audio_file_path = None

    test_recs = {}

    test_recs_file = Path.cwd() / "test_recs.pkl"
    if test_recs_file.exists():
        test_recs = joblib.load(test_recs_file) #append/update. should be deleted after model retrained.

    parts_dir = None
        
    prediction_for_file_found = True
    for index, row in audio_info_predictions.iterrows():
        if last_audio_file_path != row.filepath:
            if not prediction_for_file_found:
              #write whole file, which is unknown to us for any birds
              sf.write((parts_dir / f"ok-{str(index)}.wav"), audio_data, sr, subtype='PCM_16') 
            prediction_for_file_found = False
            
            if display_gui:
                parts_dir_base = predictions_dir
            else:
                #assume ok
                parts_dir_base = cleaned_audio_dir / row.filepath.parent.name #bird-type
            
            #write audio-data for each part to a file and edit it. same structure as cleaned dir
            rec_name = os.path.splitext(row.filepath.name)[0]
            parts_dir = parts_dir_base / rec_name #append recording-id without bird-prefix
            parts_dir.mkdir(exist_ok=True, parents=True)
            
            audio_data, sr = get_audio_data(Path(row.filepath))
            last_audio_file_path = row.filepath
            
        
        feature_audio = audio_data[row.start:row.end]
        
        if display_gui:
            out = Output(layout={'border': '1px solid black'})
            out.append_display_data(IPython.display.HTML(row.to_frame().to_html()))
            grid[index,0] = out
 
            out_2 = Output(layout={'border': '1px solid black'})
            out_2.append_display_data(IPython.display.Audio(data=feature_audio, rate=sr))
            grid[index,1] = out_2


        
        if display_gui:
            sf.write((parts_dir / (str(index) + '.wav')), feature_audio, sr, subtype='PCM_16')
        elif new_bird is not None:
            #guesss. keep "other"-label, if a non-other-label is found, replace with the bird
            use_label_other = False
            bird_detected = False
            new_bird_detected = False
            for label in mlb_columns:
              if label == 'other':
                  use_label_other = (row[label] == True)
              else:
                  bird_detected = (row[label] == True)
                  new_bird_detected = bird_detected and (label == new_bird)
            #only write if bird detected
            if bird_detected:
              prediction_for_file_found = True
              sample_filename = f"ok{str(' other') if use_label_other else ''}-{str(index)}.wav"
              sf.write((parts_dir / sample_filename), feature_audio, sr, subtype='PCM_16')
              #indication these are generated
              (parts_dir / "these_are_artificial").touch(exist_ok=True)
        else:
            print(f'error: no new_bird-code provided and display is not active')
            

        bird_code = row.filepath.parent.name

        if bird_code not in test_recs:
            test_recs[bird_code] = []

        test_recs[bird_code].append(rec_name)

    
    joblib.dump(test_recs, Path.cwd() / "test_recs.pkl")
       

def train_a_new_bird(bird_code=None, is_new=True):
    '''Also support reruns for birds'''
    
    model = None
    label_to_add = None
    
    if bird_code is None:
        model, label_to_add = add_bird_to_model()
        if label_to_add is not None:
            print(f'train_a_new_bird: adding new bird {label_to_add}')
        else:
            print(f'train_a_new_bird: no more birds')
            return None
    else:
        label_to_add = bird_code
        if is_new:
            model, label_to_add = add_bird_to_model(bird_code=bird_code)
    
    #best-effort samples to help me hopefully. strill old model
    print(f'train_a_new_bird: adding artifical samples for bird {label_to_add}')
    get_artifical_samples_for_bird(label_to_add)
    
    #save new model for usage
    
    #try to train the new model
    print(f'train_a_new_bird: start training with artificial samples for {label_to_add}')
    train_model_with_new_samples_multiple_times(model, label_to_add)
    
    return label_to_add
    
def add_birds_to_model_artificially(number_of=100):
    while train_a_new_bird() is not None and number_of > 0:
        print(f'-----DONE----')
        print(f'-----START----')
        number_of -= 1
        
def make_submission(test = False):
    base_dir = Path.cwd().parent / "input" / "birdsong-recognition"
    
    #if test:
    #    base_dir = Path.cwd().parent / "data" / "birdsong-recognition"
    
    TEST_FOLDER = None
    meta_file = None
    if test:
        TEST_FOLDER = base_dir / "example_test_audio"
        meta_file = 'example_test_audio_summary.csv'
    else:
        TEST_FOLDER = base_dir / 'test_audio'
        meta_file = 'test.csv'
    
    test_info = pd.read_csv(base_dir / meta_file)
    
    #try:
    audio_filename = None
    audio_data = None
    #predict per audiofile and keep track of predictions
    #simple dict: filename and predictions-structure for that file
    predictions_per_file = {}
    preds_list = []
    
    for index, row in test_info.iterrows():
        print(f'processing row-index {index}')
        site = start_time = row_id = audio_id = None
        if test:
            site= 'site_1' #5 secs
            start_time = row['seconds']
            audio_id = 'BLKFR-10-CPL_20190611_093000.pt540' if 'BLKFR-10-CPL' == row['filename'] else 'ORANGE-7-CAP_20190606_093000.pt623'
            row_id = row['filename_seconds']
        else:
           # Get test row information
            site = row['site']
            start_time = row['seconds'] - 5
            row_id = row['row_id']
            audio_id = row['audio_id']

        audio_filename = str(audio_id) + '.mp3'
        predictions_df = None #(filepath, start, end, possible labels)

        if audio_filename in predictions_per_file:
            #load predictions
            predictions_df = predictions_per_file[audio_filename]
            #if test:
            #    
        else:
            #create predictions
            audio_file = TEST_FOLDER / audio_filename
            print(f'predicting  {audio_file}')
            if audio_file.exists():
                predictions_df = predict_recs([audio_file], split_the_predictions=False)
                predictions_per_file[audio_filename] = predictions_df
            else:
                print(f'can not find audio file {audio_file}')
                continue


        # Get the test sound clip
        if site in ('site_1', 'site_2'):
            #process sound for 5 seconds
            start_sample = seconds_to_sample_index(start_time)
            end_sample = seconds_to_sample_index(start_time+5)
            predictions_for_timeslot = predictions_df.loc[(predictions_df.start >= start_sample) & (predictions_df.end <= end_sample)]
        else:
            #use all timeslot
            predictions_for_timeslot = predictions_df

        predicted_birds = []
        my_preds = predictions_for_timeslot.any(bool_only=True)
        for index, value in my_preds.items():
            if index not in ('start', 'end', 'filepath', 'other', 'bird'):
                if value:
                    predicted_birds.append(index)

        # Store prediction
        if len(predicted_birds) > 0:
            preds_list.append([row_id, " ".join(predicted_birds)])
        else:
            preds_list.append([row_id, "nocall"])

    preds_df = pd.DataFrame(preds_list, columns=['row_id', 'birds'])
    preds_df.to_csv('submission.csv', index=False)
    print('Saved file: submission.csv')
    #except:
    #    print(f'exception. oh no!')
        

def load_test_clip(path, start_time, duration=5):
    return librosa.load(path, offset=start_time, duration=duration)[0]

def seconds_to_sample_index(time):
    '''time in seconds to sample index'''
    global sr
    return math.floor(time*sr)


make_submission()

    
    
    
    
              
    
    
    
    


    