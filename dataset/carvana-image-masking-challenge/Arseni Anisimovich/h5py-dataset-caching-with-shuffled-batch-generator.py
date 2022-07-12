import os
import logging
import matplotlib
matplotlib.use("Agg")
import h5py
import numpy as np
from skimage.io import imread
from dask import delayed, threaded, compute

logging.getLogger('dataset loader').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, file: str=None, cache_dir: str=None, batch_size: int=64):
        self.batch_size = batch_size
        self.file = file
        if not os.path.exists(self.file):
            self.h5_file = h5py.File(self.file, 'w')
            self.cache(cache_dir)
        else:
            self.h5_file = h5py.File(self.file, 'r')
            logger.info(f"Loaded h5py dataset with {len(self.h5_file['names'])} examples.")

    @staticmethod
    def read_img(filename):
        return np.clip(imread(filename), 0, 255).astype(np.uint8)

    def cache(self, cache_dir):
        logger.info(f'Creating cache files in {cache_dir}')
        train_files = os.listdir(os.path.join(cache_dir, "train"))
        x_data = self.h5_file.create_dataset('x_data', shape=(len(train_files), 1280, 1918, 3), dtype=np.uint8)
        y_data = self.h5_file.create_dataset('y_data', shape=(len(train_files), 1280, 1918, 1), dtype=np.uint8)
        names = self.h5_file.create_dataset('names', shape=(len(train_files),), dtype=h5py.special_dtype(vlen=str))

        logger.info(f'There are {len(train_files)} files in train')
        for i, fn in enumerate(train_files):
            img = self.read_img(os.path.join(cache_dir, "train", fn))
            x_data[i, :, :, :] = img
            y_data[i, :, :, :] = imread(os.path.join(os.path.join(cache_dir, "train_masks"),
                                                     fn.replace('.jpg', '_mask.gif'))).reshape(1280, 1918, 1)
            names[i] = fn
            if i % 100 == 0:
                logger.info(f"Processed {i} files.")

    def batch_iterator(self, number_of_examples: int=None, batch_size: int=None, num_epochs: int=10, shuffle=False):
        """Generates a batch iterator for a dataset."""
        if batch_size is None:
            batch_size = self.batch_size
        names = self.h5_file['names']
        data_size = len(names)
        if number_of_examples is not None:
            data_size = number_of_examples
        x_dat = self.h5_file['x_data']
        y_dat = self.h5_file['y_data']
        num_batches_per_epoch = int((data_size-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.arange(data_size)
            if shuffle:
                shuffle_indices = np.random.permutation(shuffle_indices)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                batch_indices = sorted(list(shuffle_indices[start_index:end_index]))
                yield epoch, batch_num, \
                      compute([delayed(x_dat.__getitem__)(i) for i in batch_indices], get=threaded.get), \
                      compute([delayed(y_dat.__getitem__)(i) for i in batch_indices], get=threaded.get), \
                      compute([delayed(names.__getitem__)(i) for i in batch_indices], get=threaded.get)

