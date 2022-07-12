"""
This is the 4th notebook I'm making using EfficientNet on TPUs. The full list:
    1. https://www.kaggle.com/xhlulu/flowers-tpu-concise-efficientnet-b7
    2. https://www.kaggle.com/xhlulu/plant-pathology-very-concise-tpu-efficientnet
    3. https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
References:
    1. https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
"""
import os
os.system('pip install /kaggle/input/efficientnet-keras-source-code/ -q')

import efficientnet.tfkeras as efn
import pandas as pd
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Helper functions
def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    return strategy

def decode_image(path, label=None, target_size=(512, 512)):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, target_size)
    
    return img if label is None else img, label

def data_augment(img, label=None):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img if label is None else img, label

def build_dataset(paths, bsize, labels=None, cache=True,
                  decode_fn=decode_image, augment_fn=data_augment,
                  augment=True, repeat=True, shuffle=1024):
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache() if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(batch_size).prefetch(AUTO)
    
    return dset


# ############### Variables and configurations ###############
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16
GCS_DS_PATH = KaggleDatasets().get_gcs_path('cassava-leaf-disease-classification')

# ############### Loading and preprocess CSVs ###############
load_dir = "/kaggle/input/cassava-leaf-disease-classification/"
df = pd.read_csv(load_dir + 'train.csv')
df['paths'] = GCS_DS_PATH + "/train_images/" + df.image_id
sub_df = pd.read_csv(load_dir + 'sample_submission.csv')

# ############### Splitting and defining the dataset ###############
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = build_dataset(
    train_df.paths, train_df.label, bsize=BATCH_SIZE)
valid_dataset = build_dataset(
    valid_df.paths, valid_df.label, bsize=BATCH_SIZE, 
    repeat=False, shuffle=False, augment=False)

# ############### Build and compile the model ###############
with strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB7(
            input_shape=(512, 512, 3),
            weights='noisy-student',
            include_top=False),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    model.summary()

# ############### Train the model ###############
steps_per_epoch = train_df.shape[0] // BATCH_SIZE
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model.h5', save_best_only=True)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=3, min_lr=1e-6)

model.fit(
    train_dataset, 
    epochs=20,
    verbose=2,
    callbacks=[checkpoint, lr_reducer],
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset)