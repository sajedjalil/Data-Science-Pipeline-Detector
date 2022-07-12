# %% [code]
import numpy as np
import pandas as pd
from math import ceil
from scipy import signal
import tensorflow as tf
#from keras.utils import Sequence
from tensorflow.keras.utils import Sequence


def id2path(idx,is_train=True):
    path = "../input/g2net-gravitational-wave-detection"
    folder = 'train' if is_train else 'test'
    return f'{path}/{folder}/{idx[0]}/{idx[1]}/{idx[2]}/{idx}.npy'

def loadSample(idx, is_train=True):
     if  isinstance(idx, list):
        return np.array([loadSample(id1, is_train) for id1 in idx])
     else:
        return np.load(id2path(idx,is_train))

bHP, aHP = signal.butter(8, (20, 500), btype='bandpass', fs= 2048)
def filterSig(waves, a=aHP, b=bHP):
    '''Apply a 20Hz high pass filter to the three events'''
    return np.array([signal.filtfilt(b, a, wave) for wave in waves]) #lfilter introduces a larger spike around 20hz

def prepare(idx, is_train, scale=1.3e+22, data_format='channels_first'):
    if data_format not in ['channels_last', 'channels_first']:
        raise ValueError("Expected data_format to be 'channels_last' or 'channels_first'.")
    
    if  isinstance(idx, list):
        return np.array([prepare(id1, is_train, scale, data_format) for id1 in idx])
    
    waves = filterSig(loadSample(idx,is_train)*scale)    #[3, len(signal)]
    if data_format == 'channels_last':
        waves = waves.T                                  #[len(signal), 3 ]  
    return waves

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_wave(idx, wave, label):
    #define the dictionary -- the structure -- of our single example
    data = {
        'idx'    : _bytes_feature(idx.encode()),
        'length' : _int64_feature(wave.shape[1]),
        'wave'   : _bytes_feature(serialize_array(wave)),
    }
    if label is not None:
        data['label'] =  _int64_feature(label) 
    #create an Example, wrapping the single features
    return tf.train.Example(features=tf.train.Features(feature=data))

def write_waves_to_tfr(idList, waves, labels, filename:str="waves"):
    writer = tf.io.TFRecordWriter(filename+".tfr") 

    if labels is not None:
        for idx, wave, label in zip(idList, waves, labels):
            out = parse_single_wave(idx, wave, label)
            writer.write(out.SerializeToString())
    else:
         for idx, wave in zip(idList, waves):
            out = parse_single_wave(idx, wave, None)
            writer.write(out.SerializeToString())
    writer.close()
    print(f"Wrote {len(waves)} elements to TFRecord")

def writeBatch(lst, is_train =True, scale=1.3e+22, basename=None):

    idList = lst.id.values
    waves = [(filterSig(loadSample(idx,is_train))*scale).astype(np.float32) for idx in idList]
    target = lst.target.values if is_train else None
    
    if basename is None:
        basename = "train" if is_train else "test"
    
    write_waves_to_tfr(idList, waves, target, f"{basename}{lst.id.values[0]}")

def parse_tfr_element(element, labeled):
    data = {
        'idx'    : tf.io.FixedLenFeature([], tf.string),
        'length' : tf.io.FixedLenFeature([], tf.int64, default_value=2048*2),
        'wave'   : tf.io.FixedLenFeature([], tf.string)
    }
    if labeled:
        data['label'] = tf.io.FixedLenFeature([], tf.int64)

    content = tf.io.parse_single_example(element, data)
    idx    = content['idx']
    length = content['length']
    
    wave   = content['wave']
    wave = tf.io.parse_tensor(wave, out_type=tf.float32)
    wave = tf.reshape(wave, shape=[3,2048*2]) #HACK to define shape

    if labeled:
        label  = content['label']
        return wave, label
    else:
        return wave, idx
    
def get_dataset(filenames, labeled=True, batch_size=128,  buffer_size=tf.data.experimental.AUTOTUNE, drop_remainder=True):
    #dataset = load_dataset(filenames, labeled=labeled, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(lambda e: parse_tfr_element(e, labeled=labeled), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.cache() #Seems to blow memory limits of notebook
    dataset = dataset.shuffle(batch_size*8) 
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder) #drop_remainder=True fixes training loss: nan
    return dataset

# Use for submission data
class Dataset(Sequence):
    def __init__(self,idx, y=None,batch_size=128,shuffle=True):
        self.idx = idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        if y is not None:
            self.is_train=True
        else:
            self.is_train=False
        self.y = y
    def __len__(self):
        return ceil(len(self.idx)/self.batch_size)
    def __getitem__(self,ids):
        batch_ids = self.idx[ids * self.batch_size:(ids + 1) * self.batch_size]
        if self.y is not None:
            batch_y = self.y[ids * self.batch_size: (ids + 1) * self.batch_size]
            
        list_x = np.array([filterSig(loadSample(x,self.is_train))*1.3e+22 for x in batch_ids]).astype(np.float32)
        batch_X = np.stack(list_x)
        if self.is_train:
            return batch_X, batch_y
        else:
            return batch_X
    
    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            ids_y = list(zip(self.idx, self.y))
            shuffle(ids_y)
            self.idx, self.y = list(zip(*ids_y))
            

from tensorflow.python.client import device_lib
def accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Device:", tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        device = "TPU"
    except:
        strategy = tf.distribute.get_strategy()
        if "GPU" in [d.device_type for d in device_lib.list_local_devices()]:
            device ="GPU"
        else:
            device = "CPU"
    print(device, "Number of replicas:", strategy.num_replicas_in_sync)
   
    return strategy, device


from timeit import default_timer as timer
#from keras.callbacks import Callback

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)
        
        
        
        
    