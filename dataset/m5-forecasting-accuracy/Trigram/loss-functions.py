__author__ = "Trigram"

import tensorflow as tf
import torch, numpy as np, pandas as pd

def rmse(predictions, targets):
    return tf.keras.backend.mean(tf.math.sqrt((tf.convert_to_tensor((predictions - targets) ** 2))))

import torch
def rmse(predictions, targets):
    return torch.mean(torch.sqrt((torch.from_numpy((predictions - targets) ** 2))))

def rmsse(self,valid_y: pd.DataFrame, valid_preds: pd.DataFrame, lv: int):
    score = tf.convert_to_tensor(np.array((valid_y.values - valid_preds.values) ** 2))
    scale = getattr(self, f'lv{lv}_scale')
    return (score / scale).map(tf.sqrt)

import torch
def rmsse(self, valid_y: pd.DataFrame, valid_preds: pd.DataFrame, lv: int):
    score = torch.from_numpy(np.array((valid_y.values - valid_preds.values) ** 2))
    scale = lv
    return (score / scale).map(torch.sqrt)

## Usage guide ###################################
"""
    You can use this by adding it as a utility script and calling:
    import loss_functions
    rmse = lambda x: loss_functions.rmse(x) # just a demo
    
    These loss functions are supported by Torch and TF Neural Nets.
"""
##################################################