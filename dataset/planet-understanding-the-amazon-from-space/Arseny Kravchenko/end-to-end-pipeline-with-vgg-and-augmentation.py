from itertools import chain
from threading import Lock
import logging
from os import listdir, path, environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import numpy as np
import pandas as pd
import joblib
from skimage.io import imread
from skimage.transform import rescale, rotate

from sklearn.model_selection import KFold
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras import backend as K
from imgaug import augmenters as iaa

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

MAIN_DIR = '/home/arseny/kaggle_data/'

USE_TIFF = False

TRAIN_LABELS = MAIN_DIR + 'train_v2.csv'
TRAIN_DIR = 'train-tif-v2/' if USE_TIFF else 'train-jpg/'
TEST_DIR = 'test-tif-v2/' if USE_TIFF else 'test-jpg/'
# please copy all test images in one directory

TRAIN_DIR = MAIN_DIR + TRAIN_DIR
TEST_DIR = MAIN_DIR + TEST_DIR

DTYPE = np.float16

NEW_SIZE = 224
SCALE_COEFF = NEW_SIZE / 256
N_FOLDS = 3


def form_double_batch(X, y1, y2, batch_size):
    idx = np.random.randint(0, X.shape[0], int(batch_size))
    return X[idx], y1[idx], y2[idx]


def rotate_determined(img):
    img1 = img
    img2 = rotate(img, 90, preserve_range=True)
    img3 = rotate(img, 180, preserve_range=True)
    img4 = rotate(img, 270, preserve_range=True)
    arr = np.array([img1, img2, img3, img4]).astype(np.float16)
    return arr


def get_rotate_angle():
    return np.random.choice([0, 90, 180, 270])


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def double_batch_generator(X, y1, y2, batch_size):
    seq = iaa.Sequential([iaa.Sometimes(.8, iaa.Affine(rotate=get_rotate_angle(),
                                                       mode='reflect')),
                          iaa.Fliplr(p=.25)
                          ],
                         random_order=False)

    while True:
        x_batch, y1_batch, y2_batch = form_double_batch(X, y1, y2, batch_size)
        new_x_batch = seq.augment_images(x_batch)
        new_x_batch = np.array(new_x_batch).astype('float16')
        yield new_x_batch, {'labels': y1_batch, 'weather': y2_batch}


def process_image(fname):
    img = rescale(imread(fname), SCALE_COEFF, preserve_range=True, mode='reflect')
    return img.astype(DTYPE)


class Dataset:
    def __init__(self, batch_size=64, fold=2):
        self.df = pd.read_csv(TRAIN_LABELS)
        self.train_folds, self.test_folds = self.get_folds()
        self.fold = fold
        self.batch_size = batch_size
        self.labels, self.reverse_labels, self.weather, self.reverse_weather = self.get_labels()

    @staticmethod
    def get_folds():
        train_files = np.array(listdir(TRAIN_DIR))
        folder = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        trains, tests = zip(*folder.split(train_files))
        return trains, tests

    def get_labels(self):
        labels = self.df.tags.values
        weather = {'partly_cloudy', 'clear', 'cloudy', 'haze'}
        labels = list(set(chain(*[x.split(' ') for x in labels])) - weather)
        weather = list(weather)

        weather.sort()
        labels.sort()
        labels = {name: i for i, name in enumerate(labels)}
        reverse_labels = {i: name for i, name in enumerate(labels)}
        weather = {name: i for i, name in enumerate(weather)}
        reverse_weather = {i: name for i, name in enumerate(weather)}

        return labels, reverse_labels, weather, reverse_weather

    def cache_train(self):
        logger.info('Creating cache file for train')
        file = h5py.File('train.h5', 'w')
        train_files = listdir(TRAIN_DIR)

        x_data = file.create_dataset('train_x', shape=(len(train_files), 224, 224, 3), dtype=DTYPE)
        y_weather = file.create_dataset('train_weather', shape=(len(train_files), 4), dtype=DTYPE)
        y_labels = file.create_dataset('train_labels', shape=(len(train_files), 13), dtype=DTYPE)
        x_data_cropped = file.create_dataset('train_x_cropped', shape=(len(train_files) * 4, 224, 224, 3), dtype=DTYPE)
        names = file.create_dataset('train_names', shape=(len(train_files) * 4,), dtype=h5py.special_dtype(vlen=str))

        for i, (x, y_l, y_w, fname) in enumerate(self.get_images()):
            x_data[i, :, :, :] = x
            y_weather[i, :] = y_w
            y_labels[i, :] = y_l

            for j, img_cropped in enumerate(rotate_determined(x)):
                x_data_cropped[4 * i + j, :, :, :] = img_cropped
                names[4 * i + j] = fname

        file.close()

    def cache_test(self):
        logger.info('Creating cache file for test')
        file = h5py.File('test.h5', 'w')
        test_files = listdir(TEST_DIR)

        x_data = file.create_dataset('test_x', shape=(len(test_files) * 4, 224, 224, 3), dtype=DTYPE)
        x_names = file.create_dataset('test_names', shape=(len(test_files) * 4,), dtype=h5py.special_dtype(vlen=str))

        images = [(f, process_image(path.join(TEST_DIR, f))) for f in listdir(TEST_DIR)]

        for i, (f, img) in enumerate(images):
            for j, img_cropped in enumerate(rotate_determined(img)):
                x_data[4 * i + j, :, :, :] = img_cropped
                x_names[4 * i + j] = f
        file.close()

    def update_fold(self):
        if self.fold + 1 < len(self.train_folds):
            self.fold += 1
            logger.info('Switching to fold {}'.format(self.fold))
            return self.fold

        logger.info('It was a final fold')
        return

    def get_train(self, fold):
        try:
            file = h5py.File('train.h5', 'r')
        except OSError:
            self.cache_train()
            file = h5py.File('train.h5', 'r')

        x_data = file['train_x_cropped']
        x_names = file['train_names']
        idx = self.test_folds[fold]
        idx = np.hstack([np.array([4 * x, 4 * x + 1, 4 * x + 2, 4 * x + 3]) for x in idx]).tolist()
        return x_data[idx], x_names[idx]

    def get_test(self):
        try:
            file = h5py.File('test.h5', 'r')
        except OSError:
            self.cache_test()
            file = h5py.File('test.h5', 'r')

        x_data = file['test_x']
        x_names = file['test_names']

        return x_data, x_names

    def make_double_generator(self, use_train=True, batch_size=None):
        try:
            file = h5py.File('train.h5', 'r')
        except OSError:
            self.cache_train()
            file = h5py.File('train.h5', 'r')

        idx = self.train_folds[self.fold] if use_train else self.test_folds[self.fold]
        x_data = file['train_x']
        y_data_labels = file['train_labels']
        y_data_weather = file['train_weather']
        x_data, y_data_labels, y_data_weather = map(lambda x: x[idx.tolist()][:],
                                                    (x_data, y_data_labels, y_data_weather))

        return double_batch_generator(x_data, y_data_labels, y_data_weather,
                                      batch_size if batch_size else self.batch_size)

    def encode_target(self, tags):
        target_labels = np.zeros(len(self.labels))
        target_weather = np.zeros(len(self.weather))
        for tag in tags.split(' '):
            if tag in self.labels:
                target_labels[self.labels[tag]] = 1
            else:
                target_weather[self.weather[tag]] = 1
        return target_labels, target_weather

    def get_images(self):
        pd.read_csv(TRAIN_LABELS)
        images_dir = TRAIN_DIR

        ext = 'tif' if USE_TIFF else 'jpg'
        for i, row in self.df.iterrows():
            fname = '{}{}.{}'.format(images_dir, row.image_name, ext)
            x = process_image(fname)
            y_label, y_weather = map(lambda y: y.astype(np.int8), self.encode_target(row.tags))

            if not i % 1000:
                logger.info('{} images loaded'.format(i))

            yield x, y_label, y_weather, row.image_name


class Master:
    def __init__(self, batch_size=64, fold=0):
        self.batch_size = batch_size
        self.dataset = Dataset(batch_size=batch_size, fold=fold)
        self.fold = fold

    def get_callbacks(self, name):
        model_checkpoint = ModelCheckpoint('networks/{}_current_{}.h5'.format(name, self.fold),
                                           monitor='val_loss',
                                           save_best_only=True, verbose=0)
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=4)
        return [model_checkpoint, es, reducer]

    def get_vgg(self, shape):
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=shape)
        vgg.layers = vgg.layers[:15]
        gap = GlobalAveragePooling2D()(vgg.output)
        drop = Dropout(0.3)(gap)
        dense = Dense(1024)(drop)
        dense = LeakyReLU(alpha=.01)(dense)
        drop2 = Dropout(0.3)(dense)
        dense2 = Dense(128)(drop2)
        dense2 = LeakyReLU(alpha=.01)(dense2)
        out_labels = Dense(13, activation='sigmoid', name='labels')(dense2)
        out_weather = Dense(4, activation='softmax', name='weather')(dense2)

        model = Model(inputs=vgg.input, outputs=[out_labels, out_weather])
        model.compile(optimizer=Adam(clipvalue=3),
                      loss={'labels': 'binary_crossentropy', 'weather': 'categorical_crossentropy'},
                      )
        return model

    def fit(self):
        base_size = NEW_SIZE if NEW_SIZE else 256
        shape = (base_size, base_size, 4) if USE_TIFF else (base_size, base_size, 3)

        model = self.get_vgg(shape)

        logger.info('Fitting started')

        model.fit_generator(self.dataset.make_double_generator(use_train=True),
                            epochs=500,
                            steps_per_epoch=500,
                            validation_data=self.dataset.make_double_generator(use_train=True),
                            workers=4,
                            validation_steps=100,
                            callbacks=self.get_callbacks('united')
                            )

        new_fold = self.dataset.update_fold()
        if new_fold:
            self.fold = new_fold
            K.clear_session()
            self.fit()

    def _wrap_prediction(self, name, pred_l, pred_w):
        pred = {self.dataset.reverse_labels[i]: pred_l[i] for i in range(pred_l.shape[0])}
        pred.update({self.dataset.reverse_weather[i]: pred_w[i] for i in range(pred_w.shape[0])})
        pred['image_name'] = name.split('.')[0]
        return pred

    def wrap_predicitions(self, names, labels, weather):
        return joblib.Parallel(n_jobs=8, backend='threading')(
            joblib.delayed(self._wrap_prediction)(name, pred_l, pred_w)
            for name, pred_l, pred_w in zip(names, labels, weather))

    def make_predictions(self):
        test_data, test_names = self.dataset.get_test()

        test_result = []
        train_result = []

        for fold in range(N_FOLDS):
            logger.info('Prediction started for fold {}'.format(fold))
            model = load_model('networks/united_current_{}.h5'.format(fold))

            labels_test, weather_test = model.predict(test_data, batch_size=96)

            train_data, train_names = self.dataset.get_train(fold)
            labels_train, weather_train = model.predict(train_data, batch_size=96)

            labels_test, weather_test, labels_train, weather_train = map(lambda x: x.astype('float16'),
                                                                         (labels_test, weather_test,
                                                                          labels_train, weather_train))

            logger.info('Data transformation started for fold {}'.format(fold))

            test_result += list(self.wrap_predicitions(test_names, labels_test, weather_test))
            train_result += list(self.wrap_predicitions(train_names, labels_train, weather_train))

            K.clear_session()

        train_result, test_result = map(lambda x: pd.DataFrame(x).groupby(['image_name']).agg(np.mean).reset_index(),
                                        (train_result, test_result))

        train_result.to_csv('train_probs.csv', index=False)
        test_result.to_csv('test_probs.csv', index=False)

        final = []
        threshold = .2
        for _, row in test_result.iterrows():
            row = row.to_dict()
            name = row.pop('image_name')
            tags = [k for k, v in row.items() if v > threshold]
            final.append({'image_name': name,
                          'tags': ' '.join(tags)})
        pd.DataFrame(final).to_csv('result.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    master = Master(batch_size=64, fold=0)
    master.fit()
    master.make_predictions()
