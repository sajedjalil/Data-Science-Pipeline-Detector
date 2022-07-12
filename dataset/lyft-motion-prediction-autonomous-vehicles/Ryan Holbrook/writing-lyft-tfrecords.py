# %% [code]
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Imports #

# import kaggle_l5kit

!pip uninstall typing -yq
!pip install l5kit -q

import numpy as np
import torch
import tensorflow as tf

from l5kit.configs import load_config_data
from torch.utils.data import DataLoader
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from typing import Dict
from tempfile import gettempdir
import os
import multiprocessing as mp
# -

# # Load Data #

# +
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/home/kaggle"
# get config
cfg = {'format_version': 4, 
       'model_params': {'model_architecture': 'resnet50', 'history_num_frames': 0, 'history_step_size': 1, 'history_delta_time': 0.1, 'future_num_frames': 50, 'future_step_size': 1, 'future_delta_time': 0.1}, 
       'raster_params': {'raster_size': [224, 224], 'pixel_size': [0.5, 0.5], 'ego_center': [0.25, 0.5], 'map_type': 'py_semantic', 'satellite_map_key': 'aerial_map/aerial_map.png', 'semantic_map_key': 'semantic_map/semantic_map.pb', 'dataset_meta_key': 'meta.json', 'filter_agents_threshold': 0.5}, 
       'train_data_loader': {'key': 'scenes/sample.zarr', 'scene_indices': [-1], 'perturb_probability': 0, 'batch_size': 12, 'shuffle': True, 'num_workers': 16}, 
       'val_data_loader': {'key': 'scenes/sample.zarr', 'scene_indices': [-1], 'perturb_probability': 0, 'batch_size': 12, 'shuffle': False, 'num_workers': 16}, 
       'train_params': {'checkpoint_every_n_steps': 10000, 'max_num_steps': 5, 'eval_every_n_steps': 10000}}

dm = LocalDataManager(None)
rasterizer = build_rasterizer(cfg, dm)

# -

# # Serialize with tf.Example #

# +
from tensorflow.train import BytesList
from tensorflow.train import Example, Features, Feature

def encode_batch(batch):
    encoded = {k: tf.io.serialize_tensor(v.numpy()).numpy() 
               for k, v in batch.items()}
    return encoded

def make_example(batch):
    features = Features(
        feature={k: Feature(bytes_list=BytesList(value=[tensor]))
                 for k, tensor in batch.items()}
    )
    return Example(features=features).SerializeToString()
    

def make_batch_generator(dataloader):
    def generator():
        for batch in dataloader:
            batch = encode_batch(batch)
            batch = make_example(batch)
            yield batch
    return generator

# # Write TFRecord Shards #

os.mkdir("/kaggle/working/tfrecords")

# ## Training ##

# +

train_cfg = cfg["train_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=train_cfg["shuffle"],
    batch_size=train_cfg["batch_size"], 
    num_workers=train_cfg["num_workers"],
)

train_generator = make_batch_generator(train_dataloader)
train_encoded = tf.data.Dataset.from_generator(train_generator, output_types=tf.string)

os.mkdir("/kaggle/working/tfrecords/training")

def write_training_shard(dataset, total_shards, shard_num):
    shard = dataset.shard(total_shards, shard_num)
    writer = tf.data.experimental.TFRecordWriter(
        "/kaggle/working/tfrecords/training/shard_{:03d}.tfrecord".format(shard_num)
    )
    writer.write(shard)

NUM_SHARDS = 16*16 #==256
for shard in range(NUM_SHARDS):
    write_training_shard(train_encoded, NUM_SHARDS, shard)
# -

# ## Validation ##

# +

val_cfg = cfg["val_data_loader"]

val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
val_dataset = AgentDataset(cfg, val_zarr, rasterizer)
val_dataloader = DataLoader(
    val_dataset,
    shuffle=val_cfg["shuffle"],
    batch_size=val_cfg["batch_size"], 
    num_workers=val_cfg["num_workers"],
)

val_generator = make_batch_generator(val_dataloader)
val_encoded = tf.data.Dataset.from_generator(val_generator, output_types=tf.string)

os.mkdir("/kaggle/working/tfrecords/validation")

def write_validation_shard(dataset, total_shards, shard_num):
    shard = dataset.shard(total_shards, shard_num)
    writer = tf.data.experimental.TFRecordWriter(
        "/kaggle/working/tfrecords/validation/shard_{:03d}.tfrecord".format(shard_num)
    )
    writer.write(shard)

NUM_VAL_SHARDS = 16*16 #==256
for shard in range(NUM_VAL_SHARDS):
    write_validation_shard(val_encoded, NUM_VAL_SHARDS, shard)

# -

