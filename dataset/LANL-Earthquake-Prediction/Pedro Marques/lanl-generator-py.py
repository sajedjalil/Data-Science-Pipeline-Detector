"""
This module creates balanced train / validation datasets so that the validation set
the same rate of samples of each class as the train train. Where a class is defined
as an interval of 0.5 secs in time_to_failure.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os

from tensorflow import keras
from typing import List, Tuple
import random

class FoldGenerator(object):
    """ Creates 5 folds with 1/5 of the data used for test dataset for each fold.
    Assumes that the segments have been assigned into classes by classify_segments.

    The implementation creates a list of segments of each classes and uses a window
    of 1/5 of the samples that assigns to the test set in each fold. The last fold is
    unbalanced and has more elements in the test set.
    """
    def __init__(self, df_segments: pd.DataFrame, random_state=42, n_classes=33):
        df = df_segments
        class_segments = [ df[df.xclass == i].index.values for i in range(n_classes) ]
        np.random.seed(random_state)
        self.class_segments = [ np.random.permutation(segments) for segments in class_segments ]
        self.kfolds = 5

    def __len__(self):
        return self.kfolds

    def __getitem__(self, index):
        train_segments = []
        test_segments = []
        for xclass in self.class_segments:
            wsize = len(xclass) // self.kfolds
            start = index * wsize
            if index == self.kfolds - 1:
                end = None
            else:
                end = (index + 1) * wsize
            test_segments.extend(xclass[start:end])
            if start > 0:
                train_segments.extend(xclass[:start])
            if end:
                train_segments.extend(xclass[end:])

        return train_segments, test_segments
    
class random_state(object):
    """ Aux class to save the state of the random number generator
    """
    def __init__(self, rstate_ref: List[Tuple]):
        if len(rstate_ref) == 0:
            rstate_ref.append(None)
        if len(rstate_ref) != 1:
            raise ValueError('rstate_ref must be a List with 0 or 1 elements')
        self.rstate_ref = rstate_ref

    def __enter__(self):
        self.rsaved = random.getstate()
        if self.rstate_ref[0]:
            random.setstate(self.rstate_ref[0])
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.rstate_ref[0] = random.getstate()
        random.setstate(self.rsaved)
        

class SegmentGenerator(keras.utils.Sequence):
    """ Generates training examples given a set of indices (obtained FoldGenerator).
    It performs data augmentation by randomly shuffling the start of the segment
    when the dataset owns the next contiguos segment.
    i.e. If the train dataset contains blocks N and N+1, when rand_offset is set, the
    start offset for block N is randomly shifted forward by up to rand_offset.
    This is can be used to avoid overfitting.
    """
    def __init__(self, df, feature_gen, indices: List[int], rand_offset=0, random_seed=42, batch_size=32):
        self.dataframe = df
        self.feature_gen = feature_gen
        self.segment_set = set(indices)
        self.segment_list = indices.copy()
        self.rstate_ref = []
        self.rand_offset = rand_offset
        self.batch_size = batch_size

        with random_state(self.rstate_ref):
            random.seed(random_seed)
            random.shuffle(self.segment_list)

    def __len__(self):
        return (len(self.segment_list) - 1) // self.batch_size + 1

    def __getitem__(self, index):
        idx_start = index * self.batch_size
        idx_end = min(idx_start + self.batch_size, len(self.segment_list))

        X_list = []
        Y_list = []

        for seg_index in self.segment_list[idx_start:idx_end]:
            start = seg_index * 150_000

            # randomly shuffle the start offset if we are allowed to use the
            # next contiguos segment
            if self.rand_offset and (seg_index + 1) in self.segment_set:
                with random_state(self.rstate_ref):
                    offset = random.randint(0, self.rand_offset)
                start += offset
            df = self.dataframe[start:start + 150_000]
            data = self.feature_gen.generate(df)
            if isinstance(data[0], list):
                if not X_list:
                    X_list = [[] for _ in data[0]]
                for i, v in enumerate(data[0]):
                    X_list[i].append(v)
            else:
                X_list.append(data[0])
            Y_list.append(data[1])

        if X_list and isinstance(X_list[0], list):
            X = [np.stack(items) for items in X_list]
        else:
            X = np.stack(X_list)
        Y = np.stack(Y_list)
        return X, Y

    def on_epoch_end(self):
        with random_state(self.rstate_ref):
            random.shuffle(self.segment_list)


def get_lanl_classes():
    """ Classes by time_to_failure. 16.2 is greater than any sample (i.e. +inf).
    """
    classes = np.concatenate([np.arange(0.5, 16.0, 0.5), np.array([16.2])])
    return classes


def classify_segments(df: pd.DataFrame, classes: np.array):
    """ Assign segments to classes so that FoldGenerator can create balanced folds.
    """
    end_offsets = np.array(list(range(150_000 - 1, df.shape[0], 150_000)))
    offsets = end_offsets - 150_000 + 1
    ttf = df['time_to_failure'].values[end_offsets]
    delta = np.diff(ttf)

    boundary = np.where(delta > 0, True, False)
    boundary = np.concatenate([np.array([False]), boundary])

    params = {
        'start': offsets,
        'end': end_offsets,
        'time_to_failure': ttf,
        'xclass': np.digitize(ttf, classes),
        'boundary': boundary,
    }
    df_segments = pd.DataFrame(params)

    # override the class for elements on boundary
    df_segments.loc[df_segments[df_segments.boundary == True].index, 'xclass'] = 32
    return df_segments
