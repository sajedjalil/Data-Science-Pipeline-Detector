import numpy as np
import pandas as pd
from math import ceil
import cv2
import zipfile
from tqdm import tqdm
from keras.applications.vgg16 import VGG16, preprocess_input
from multiprocessing import Value, Pool
import gzip
from pathlib import PurePath
from scipy import sparse
from threading import Thread
from queue import Queue
import gc

fname = '../input/avito-demand-prediction/train_jpg.zip'
pool = 'avg' # one of max of avg
batch_size = 64
im_dim = 96
n_channels = 3
limit = None # Limit number of images processed (useful for debug)
resize_mode = 'fit' # One of fit or crop
bar_iterval = 10 # in seconds
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
    ids = pd.read_csv('../input/avito-demand-prediction/train.csv', usecols=['image'], nrows=limit)['image'].tolist()
    
    n_items.value = len(ids)
    print("Total items:", n_items.value)

    # Iterate over ids
    for im_id in ids:
        zfile = 'data/competition_files/train_jpg/{}.jpg'.format(im_id)
        try:
            zinfo = train_zip.getinfo(zfile)
            zbuf = np.frombuffer(train_zip.read(zinfo), dtype='uint8')
        except KeyError:
            zbuf = None
            
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
        

def im_decode_resize(params):
    item_id, zbuf = params
    
    if zbuf is None:
        return item_id, None
    else:
        try:
            im = resize( cv2.imdecode(zbuf, cv2.IMREAD_COLOR) )
        except Exception as e:
            print('Error decoding item_id', item_id, e)
            # Fallback to empty image
            im = None
        
        return item_id, im

def predict_batch(model, X_batch):
    # Predict step
    X_batch = preprocess_input(np.array(X_batch, dtype=np.float32))
    features_batch = model.predict_on_batch(X_batch)
    
    # We will convert to sparse-float16 to save space in memory and disk.
    # A subsample analysis results in 2/3 of the features being zero
    return sparse.csr_matrix( features_batch.astype(np.float16) )
    
    
if __name__ == '__main__':
    print("Loading model...")
    model = VGG16(weights=None, pooling=pool, include_top=False)
    model.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    
    n_items = Value('i', -1)  # Async number of items
    sparse_features = []
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
    
            if len(X_batch) == batch_size:
                sparse_features.append( predict_batch(model, X_batch) )
                del X_batch
                X_batch = []
                bar.update(batch_size)
    
        # Predict last batch
        if len(X_batch) > 0:
            sparse_features.append( predict_batch(model, X_batch) )
            bar.update(len(X_batch))
    
    finally:
        pool.close()
        del pool, model, X_batch
    
        if bar:
            bar.close()
    
        gc.collect()
    
    print('Concating sparse matrix...')
    features = sparse.vstack(sparse_features)
    
    print('Saving sparse matrix...')
    sparse.save_npz('features.npz', features, compressed=True)

    print('All done! Good luck')