import io
import pandas as pd
import numpy as np
import threading
import bson
from tqdm import tqdm
import struct
import random

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras import backend as K


def read_bson(bson_path, num_records, with_categories):
    """
    Reads BSON
    """
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        records_read = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            records_read += 1
            pbar.update()
    pbar.close()
    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def make_category_tables(categories_path):
    """
    Converts category name into an index [0, N-1]
    """
    categories_df = pd.read_csv(categories_path, index_col="category_id")
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


def get_obs(fname, offset, length):
    fobj = open(fname, 'rb')
    fobj.seek(offset)
    res = bson.BSON.decode(fobj.read(length))
    fobj.close()
    return res


class BSONGenerator(object):
    def __init__(self, dataset, metadata, image_data_generator, lock,
                 num_classes, batch_size=16, shuffle=True, target_size=(90,90,3)):
        self.dataset = dataset
        self.metadata = metadata
        self.batch_size = batch_size
        self.lock = lock
        self.image_data_generator = image_data_generator
        self.num_classes = num_classes
        self.target_size = target_size
        self.shuffle = shuffle

    def __data_generation(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.target_size, dtype=K.floatx())
        batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                entry = self.metadata.iloc[j]
                num_imgs, offset, length, target = entry
                obs = get_obs(self.dataset, offset, length)
                keep = np.random.choice(len(obs['imgs']))
                byte_str = obs['imgs'][keep]['picture']

                img = load_img(io.BytesIO(byte_str), target_size=self.target_size)

                x = img_to_array(img)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)

            batch_x[i] = x
            batch_y[i, target] = 1

        return batch_x, batch_y

    def __get_exploration_order(self, sample_ids):
        indexes = np.arange(len(sample_ids))
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def generate(self, sample_ids):
        while 1:
            indexes = self.__get_exploration_order(sample_ids)

            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                index_array = [sample_ids[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                batch_x, batch_y = self.__data_generation(index_array)

                yield batch_x, batch_y


if __name__ == "__main__":
    train_bson_file_path = '../input/train.bson'
    # train_bson_file_path = './data/train_example.bson'
    category_file_path = '../input/category_names.csv'

    num_train_examples = 7069896
    # N_TRAIN = 1000
    num_classes = 5270
    BS = 32
    N_THREADS = 4

    # mapping the catigores into 0-5269 range
    cat2idx, idx2cat = make_category_tables(category_file_path)
    # Scanning the metadata
    meta_data = read_bson(train_bson_file_path, num_train_examples, with_categories=True)
    meta_data.category_id = np.array([cat2idx[ind] for ind in meta_data.category_id])

    lock=threading.Lock()
    
    train_data_generator = ImageDataGenerator(rescale=1./255,
                                          zoom_range=0.1,
                                          width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          horizontal_flip=True)

    gen = BSONGenerator(train_bson_file_path, meta_data, train_data_generator, 
    lock, num_classes)
    
    # Split into training & validation sets
    validation_size = 5000
    validation_ids = random.sample(range(num_train_examples), validation_size)
    train_ids = list(set(range(num_train_examples)) - set(validation_ids))

    # Training generator
    train_generator = gen.generate(train_ids)
