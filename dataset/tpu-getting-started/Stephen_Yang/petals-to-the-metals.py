import numpy as np
import pandas as pd 
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from kaggle_datasets import KaggleDatasets
import os, math
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()
print("Number of accelerators: ", strategy.num_replicas_in_sync)
gcs_path = KaggleDatasets().get_gcs_path('tpu-getting-started')
gcs_train = gcs_path+'/tfrecords-jpeg-192x192/train/*.tfrec'
gcs_val = gcs_path+'/tfrecords-jpeg-192x192/val/*.tfrec'
gcs_test = gcs_path+'/tfrecords-jpeg-192x192/test/*.tfrec'
gfile_train = tf.io.gfile.glob(gcs_train)
gfile_val = tf.io.gfile.glob(gcs_val)
gfile_test = tf.io.gfile.glob(gcs_test)
batch_size = 16*strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE
img_size=192
def tfrecord_read(path):
    tfrecord_format = {
        'image':tf.io.FixedLenFeature([], tf.string),
        'class':tf.io.FixedLenFeature([], tf.int64)
    }
    data = tf.io.parse_single_example(path, tfrecord_format)
    image = tf.image.decode_jpeg(data['image'], channels=3)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.reshape(image, [img_size, img_size, 3])
    label = tf.cast(data['class'], tf.int32)
    return image, label
def image_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.flip_up_down(image)
    image = tf.image.random_saturation(image, 0, 2)
    image = tf.image.rot90(image)
    image = tf.image.transpose(image)
    return image, label
def make_dataset(filenames, order=False, train=False):
    option_no_order = tf.data.Options()
    if not order:
        option_no_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(tfrecord_read, num_parallel_calls=AUTO)
    dataset = dataset.cache()
    if train:
        dataset = dataset.map(image_augmentation, num_parallel_calls=AUTO)
    return dataset
def train_val_batch(filenames, train=False, order=False):
    dataset = make_dataset(filenames, order, train)
    if train:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset
train_dataset = train_val_batch(gfile_train, train=True)
val_dataset = train_val_batch(gfile_val)
print(train_dataset)
print(val_dataset)
epochs=10

#start_lr = 0.001
#exp_decay = 0.1
#def lr_schedule(epochs):
#  def lr(epochs, start_lr, exp_decay):
#    return start_lr * math.exp(-exp_decay*epochs)
#  return lr(epochs, start_lr, exp_decay)
def lr_schedule(epochs,start = 0.00001, min_lr = 0.00001, max_lr = 0.00005,
                   rampup_epochs=5, sustain_epochs=0, decay = 0.8):
    #One cycle learning rate Definition
    def lr(epochs, start, min_lr, max_lr, rampup_epochs, sustain_epochs, decay):
        if epochs < rampup_epochs:
            lr = ((max_lr - start)/ rampup_epochs * epochs + start)
        elif epochs < rampup_epochs+sustain_epochs:
            lr = max_lr
        else:
            lr = ((max_lr - min_lr)*decay**(epochs - rampup_epochs - sustain_epochs) + min_lr)
        return lr
    return lr(epochs, start, min_lr, max_lr, rampup_epochs, sustain_epochs, decay)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=True)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, mode='auto',)
with strategy.scope():
    #pretrain = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=[img_size, img_size, 3])
    #pretrain = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=[img_size, img_size, 3])
    pretrain = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=[img_size, img_size, 3])

    model = tf.keras.Sequential([
        pretrain,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(104, activation='sigmoid')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    model.summary()
    history = model.fit(train_dataset, validation_data = val_dataset, steps_per_epoch=350, epochs=epochs
                        , callbacks=[lr_callback, earlystop_callback])
plt.plot(np.arange(len(history.history['sparse_categorical_accuracy'])), history.history['sparse_categorical_accuracy'])
plt.plot(np.arange(len(history.history['val_sparse_categorical_accuracy'])), history.history['val_sparse_categorical_accuracy'])
plt.show()
    
def testrecord_read(path):
    tfrecord_format = {
        'image':tf.io.FixedLenFeature([], tf.string),
        'id':tf.io.FixedLenFeature([], tf.string)
    }
    data = tf.io.parse_single_example(path, tfrecord_format)
    image = tf.image.decode_jpeg(data['image'], channels=3)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.reshape(image, [img_size, img_size, 3])
    idx = data['id']
    return image, idx
def make_testset(filenames, order=False, train=False):
    option_no_order = tf.data.Options()
    if not order:
        option_no_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(testrecord_read, num_parallel_calls=AUTO)
    dataset = dataset.cache()
    if train:
        dataset = dataset.map(image_augmentation, num_parallel_calls=AUTO)
    return dataset    
def test_val_batch(filenames):
    dataset = make_testset(filenames)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    return dataset
testset = test_val_batch(gfile_test)
value = model.predict(testset.map(lambda image, idx : image))
result = np.argmax(value, axis=-1)
idx = testset.map(lambda idx, image: idx)
#submission = pd.read_csv(r'../input/tpu-getting-started/sample_submission.csv')
testset_idx = testset.map(lambda image, idx: idx).unbatch()
testset_idx = next(iter(testset_idx.batch(7382))).numpy().astype('U')
np.savetxt('submission.csv',np.rec.fromarrays([testset_idx, result]),fmt=['%s', '%d'],delimiter=',',header='id,label',comments='',)
    
    

    