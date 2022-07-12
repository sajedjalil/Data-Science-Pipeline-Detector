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

fname = '../input/avito-demand-prediction/test_jpg.zip'
pool = 'avg' # one of max of avg
batch_size = 64
im_dim = 96
n_channels = 3
limit = None # Limit number of images processed (useful for debug)
resize_mode = 'fit' # One of fit or crop
bar_iterval = 10 # in seconds

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

    test_zip = zipfile.ZipFile(fname)
        
    test_items = test_zip.infolist()
    
    if limit is not None:
        test_items = test_items[:limit+1]  # Add one item, that is the zip folder
        
    n_items.value = len(test_items)
    print("Total items:", n_items.value)

    for idx, zinfo in enumerate(test_items):
        if zinfo.filename.endswith('.jpg'):
            zpath = PurePath(zinfo.filename)
            item_id = zpath.stem
            zbuf = np.frombuffer(test_zip.read(zinfo), dtype='uint8')
            
            yield (item_id, zbuf)

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
    try:
        im = resize( cv2.imdecode(zbuf, cv2.IMREAD_COLOR) )
    except Exception as e:
        print('Error decoding item_id', item_id, e)
        # Fallback to empty image
        im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8)
    
    return item_id, im
    
print("Loading model...")
model = VGG16(weights=None, pooling=pool, include_top=False)
model.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def predict_batch(model, X_batch):
    # Predict step
    X_batch = preprocess_input(np.array(X_batch, dtype=np.float32))
    features_batch = model.predict_on_batch(X_batch)
    
    # We will convert to sparse-float16 to save space in memory and disk.
    # A subsample analysis results in 2/3 of the features being zero
    return sparse.csr_matrix( features_batch.astype(np.float16) )
    


n_items = Value('i', -1)  # Async number of items
sparse_features = []
items_ids = []
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

        X_batch.append(im)
        items_ids.append(item_id)
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

features = sparse.vstack(sparse_features)
print(features.shape)

print('Loading dataframe with images ids...')
df = pd.read_csv('../input/avito-demand-prediction/test.csv', usecols=['image'])

# Convert zip-file order to df order
print('Converting zip-file order to df order...')
didx = {k:i for i, k in enumerate(items_ids)}
del items_ids
gc.collect()
fidx = df['image'].map(didx).fillna(-1).astype(int).tolist()  # Fill nan with -1 to access last element of features, with will be an empty feature

fempty = np.zeros((1, features.shape[1]), dtype=features.dtype)
features = sparse.vstack([features, fempty], format='csr')

features = features[fidx]

# Sanity check
assert features.shape[0] == len(df)

# Save features
print('Saving sparse matrix...')
sparse.save_npz('features.npz', features, compressed=True)
