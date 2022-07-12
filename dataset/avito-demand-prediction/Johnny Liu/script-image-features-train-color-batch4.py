import numpy as np
import pandas as pd
from math import ceil
import cv2
import zipfile
from zipfile import ZipFile
from tqdm import tqdm

from multiprocessing import Value, Pool
import gzip
from pathlib import PurePath
from scipy import sparse
from threading import Thread
from queue import Queue
import gc
import time

from matplotlib.colors import rgb_to_hsv

# define function
def image_colorfulness(file):
    file,arr = file
    if arr.size <= 0:   # exclude dirs and blanks
        return file,0
    image = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return file,stdRoot + (0.3 * meanRoot)
debug=0
test=0
batch = 250000
batch_id = 4
bar_iterval = 60 # in seconds
# Open Zip file
start_time = time.time()
print('start loading image')
# get filenames
path = 'train_jpg'
if test:
    path = 'test_jpg'
train_zip = ZipFile('../input/%s.zip'%path)

print('end loading image')

filenames = train_zip.namelist()[1:] # exclude the initial directory listing
print('total',len(filenames))
filenames = filenames[batch*(batch_id-1):batch*batch_id]
print('images',len(filenames),batch*(batch_id-1),batch*(batch_id-1)+len(filenames))
if debug:
    filenames = filenames[:1000]
    print('debug images........',len(filenames))

def generate_files(n_items):
    print("Starting generate_files...")
    
    n_items.value = len(filenames)
    print("Total items:", n_items.value)

    # Iterate over ids
    for filename in filenames:
        try:
            # zinfo = train_zip.getinfo(zfile)
            zbuf = np.frombuffer(train_zip.read(filename), dtype='uint8')
        except KeyError:
            zbuf = None
            
        yield (filename, zbuf)
        
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
    
    col_nams = ['image','colorfull']
    n_items = Value('i', -1)  # Async number of items
    sparse_features = []
    # items_ids = []
    pool = Pool(2)
    bar = None
    h_list_trn = []
    try:
        # Threaded generator is usful for both parallel blocking read and to limit
        # items buffered by pool.imap (may cause OOM)
        generator = ThreadedGenerator( generate_files(n_items), 50 )
        for data in pool.imap(image_colorfulness, generator):
            if bar is None:
                bar = tqdm(total=n_items.value, mininterval=bar_iterval, unit_scale=True)
    
            h_list_trn.append(data)
            del data
            bar.update(1)
            
    finally:
        pool.close()
        del pool
        if bar:
            bar.close()
        gc.collect()
    
    print(type(h_list_trn))
    hash_df_trn = pd.DataFrame(h_list_trn,columns=col_nams)

    hash_df_trn['image'] = hash_df_trn.image.str.replace("data/competition_files/%s/"%path, "")
    hash_df_trn['image'] = hash_df_trn.image.str.replace(".jpg", "")
    hash_df_trn.sort_values('image').head()
    hash_df_trn.to_csv('%s_img_feat_saturation.csv'%path)
    hash_df_trn.head()
    print('tatle time %f minutes.'%((time.time()-start_time)/60))
    
    train_zip.close()
    print('All done! Good luck')