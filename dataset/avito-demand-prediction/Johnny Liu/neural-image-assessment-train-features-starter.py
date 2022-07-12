import numpy as np
import pandas as pd
from math import ceil
import cv2
import zipfile
from tqdm import tqdm
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from multiprocessing import Value, Pool
import gzip
from pathlib import PurePath
from scipy import sparse
from threading import Thread
from queue import Queue
import gc
from PIL import Image
import cv2
from keras.preprocessing import image



"""Copy Keras pre-trained model files to work directory from:
https://www.kaggle.com/gaborfodor/keras-pretrained-models

Code from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
print('---------------hanle------------------')
import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)

images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

print('---------------hanle end------------------')


fname = '../input/avito-demand-prediction/train_jpg.zip'
pool = 'avg' # one of max of avg
batch_size = 64
im_dim = 128
n_channels = 3
limit = None # Limit number of images processed (useful for debug)
resize_mode = 'fit' # One of fit or crop
bar_iterval = 30 # in seconds
empty_im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8) # Used when no image is present

def resize_fit(im, inter=cv2.INTER_AREA):
    height, width, _ = im.shape
    
    if height > width:
        new_dim = (width*im_dim//height, im_dim)
    else:
        new_dim = (im_dim, height*im_dim//width)
        
    imr = cv2.resize(im, new_dim, interpolation=inter)
    
    h, w = imr.shape[:2]

    off_x = (im_dim-w)//2
    off_y = (im_dim-h)//2
    
    im_out = np.zeros((im_dim, im_dim, n_channels), dtype=imr.dtype)

    im_out[off_y:off_y+h, off_x:off_x+w] = imr
    
    del imr
    
    return im_out


def resize_crop(im, inter=cv2.INTER_AREA):
    height, width, _ = im.shape
    
    if height > width:
        offy = (height-width) // 2
        imc = im[offy:offy+width]
    else:
        offx = (width-height) // 2
        imc = im[:, offx:offx+height]
        
    return cv2.resize(imc, (im_dim, im_dim), interpolation=inter)


def resize(im, inter=cv2.INTER_AREA):
    if resize_mode == 'fit':
        return resize_fit(im, inter)
    else:
        return resize_crop(im, inter)


def generate_files(n_items):
    print("Starting generate_files...")

    # Open Zip file
    train_zip = zipfile.ZipFile(fname)
    
    # Open train csv (get only images-ids)
    ids = train_zip.namelist()[1:]
    
    n_items.value = len(ids)
    print("Total items:", n_items.value)

    # Iterate over ids
    for im_id in ids:
        zfile = im_id#'data/competition_files/train_jpg/{}.jpg'.format(im_id)
        try:
            zinfo = train_zip.getinfo(zfile)
            zbuf = np.frombuffer(train_zip.read(zinfo), dtype='uint8')
        except KeyError:
            zbuf = None
        im_id = im_id.replace('data/competition_files/train_jpg/','')
        im_id = im_id.replace('.jpg','')
        yield (im_id, zbuf)
        
    print("Finished generate_files")
    
# Based on https://gist.github.com/everilae/9697228
class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, iterator, queue_maxsize):
        self._iterator = iterator
        self._sentinel = object()
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                self._queue.put(value)

        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value

        self._thread.join()

# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si,axis=1)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std
    
def im_decode_resize(params):
    item_id, zbuf = params
    if zbuf is None:
        return item_id, None
    if zbuf.size <= 0:   # exclude dirs and blanks
        return item_id,None
    img1 =  cv2.imdecode(zbuf, cv2.IMREAD_ANYCOLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img1, 'RGB')
    target_size = (128, 128)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    return item_id,x
    # if zbuf is None:
    #     return item_id, None
    # else:
    #     try:
    #         im = resize( cv2.imdecode(zbuf, cv2.IMREAD_COLOR) )
    #     except Exception as e:
    #         print('Error decoding item_id', item_id, e)
    #         # Fallback to empty image
    #         im = None
        
    #     return item_id, im

def predict_batch(model, X_batch):
    # Predict step
    X_batch = preprocess_input(np.array(X_batch, dtype=np.float32))
    features_batch = model.predict_on_batch(X_batch)

    # We will convert to sparse-float16 to save space in memory and disk.
    # A subsample analysis results in 2/3 of the features being zero
    return sparse.csr_matrix( features_batch.astype(np.float16) )
    
    
if __name__ == '__main__':
    print("Loading model...")
    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('../input/titu1994neuralimageassessment/inception_resnet_weights.h5')
    print("Loading model done...")
    n_items = Value('i', -1)  # Async number of items
    sparse_features = []
    all_its = []
    std_feats = []
    # items_ids = []
    pool = Pool(2)
    bar = None
    X_batch = []
    try:
        # Threaded generator is usful for both parallel blocking read and to limit
        # items buffered by pool.imap (may cause OOM)
        generator = ThreadedGenerator( generate_files(n_items), 50 )
        for item_id, im in pool.imap(im_decode_resize, generator):
            if bar is None:
                bar = tqdm(total=n_items.value, mininterval=bar_iterval, unit_scale=True)
                
            # Replace None with empty image
            if im is None:
                im = empty_im
            
            X_batch.append(im)
            # items_ids.append(item_id)
            del im
            all_its.append(item_id)
            if len(X_batch) == batch_size:
                
                X_batch1 = preprocess_input(np.array(X_batch, dtype=np.float32))
                features_batch = model.predict_on_batch(X_batch1)
                res = mean_score(features_batch)
               
                std_temp = []
                for score in features_batch:
                    std0 = std_score(score)
                    std_temp.append(std0)
                std_feats.append(np.array(std_temp)[...,None])
                sparse_features.append(res[...,None])
                del X_batch,features_batch
                X_batch = []
                bar.update(batch_size)
            # x = np.expand_dims(im, axis=0)
            
            # x = xception.preprocess_input(x)
            # preds = model.predict(x)
            # print(preds.shape)
            # sparse_features.append(preds)
            # del im,x
            # bar.update(1)
        if  len(X_batch) >0:
            X_batch1 = preprocess_input(np.array(X_batch, dtype=np.float32))
            features_batch = model.predict_on_batch(X_batch1)
            res = mean_score(features_batch)
            std_temp = []
            for score in features_batch:
                std0 = std_score(score)
                std_temp.append(std0)
            std_feats.append(np.array(std_temp)[...,None])
            sparse_features.append(res[...,None])

            bar.update(len(X_batch))

    finally:
        pool.close()
        del pool, model, X_batch
    
        if bar:
            bar.close()
    
        gc.collect()
    
    print('Concating sparse matrix...')
    features = np.vstack(sparse_features)
    print(features.shape)
    features = np.hstack([np.array(all_its)[...,np.newaxis],features,np.vstack(std_feats)])
    cols = ['image','mean_nima','std_nima']
    print(features.shape,'total')
    print('Saving to csv...')
    pd.DataFrame(features,columns=cols).to_csv('train_xception.csv')
    # sparse.save_npz('train_features.npz', features, compressed=True)
    # np.save('test_features.npy',features)

    print('All done! Good luck')