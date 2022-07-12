#!/usr/bin/env python3

##### 
##### ./kaggle_compile.py src/pipelines/multi_output_df_cnn.py --commit
##### 
##### 2020-03-31 18:16:41+01:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master f0d1273 [ahead 3] multi_output_df_cnn | model_hparams = Fastest ImageDataGenerator CNN
##### 
##### f0d1273b1ca3b899345796760abd962c6d6b8e16
##### 

#####
##### START src/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os

import simplejson

settings = {}

settings['hparam_defaults'] = {

    "optimizer":     "RMSprop",
    "scheduler":     "constant",
    "learning_rate": 0.001,
    "min_lr":        0.001,
    "split":         0.2,
    "batch_size":    128,
    "fraction":      1.0,
    "epochs":         99,

    "patience": {
        'Localhost':    10,
        'Interactive':  0,
        'Batch':        10,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), 10),

    "loops": {
        'Localhost':   1,
        'Interactive': 1,
        'Batch':       1,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), 1),

    # Timeout = 120 minutes | allow 30 minutes for testing submit | TODO: unsure of KAGGLE_KERNEL_RUN_TYPE on Submit
    "timeout": {
        'Localhost':   "24h",
        'Interactive': "5m",
        'Batch':       "110m",
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), "110m")

}

settings['verbose'] = {

    "tensorboard": {
        'Localhost':   True,
        'Interactive': False,
        'Batch':       False,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), False),

    "fit": {
        'Localhost':   1,
        'Interactive': 2,
        'Batch':       2,
    }.get(os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost'), 2)

}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/bengaliai-cv19",
        "features":    "./input_features/bengaliai-cv19/",
        "models":      "./models",
        "submissions": "./",
        "logs":        "./logs",
    }
else:
    settings['dir'] = {
        "data":        "./input/bengaliai-cv19",
        "features":    "./input_features/bengaliai-cv19/",
        "models":      "./data_output/models",
        "submissions": "./data_output/submissions",
        "logs":        "./logs",
    }

####################
if __name__ == '__main__':
    for dirname in settings['dir'].values():
        try:    os.makedirs(dirname, exist_ok=True)  # BUGFIX: read-only filesystem
        except: pass
    for key,value in settings.items():       print(f"settings['{key}']:".ljust(30), str(value))

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        with open('settings.json', 'w') as file:
            print( 'settings', simplejson.dumps(settings, indent=4*' '))
            simplejson.dump(settings, file, indent=4*' ')


#####
##### END   src/settings.py
#####

#####
##### START src/dataset/Transforms.py
#####

import math
from typing import AnyStr, Dict, List, Union

import gc
import numpy as np
import pandas as pd
import skimage.measure
from pandas import DataFrame, Series

# from src.settings import settings



class Transforms():
    csv_filename         = f"{settings['dir']['data']}/train.csv"
    csv_data             = pd.read_csv(csv_filename).set_index('image_id', drop=True).astype('category')
    csv_data['grapheme'] = csv_data['grapheme'].cat.codes.astype('category')


    @classmethod
    #@profile
    def transform_Y(cls, df: DataFrame, Y_field: Union[List[str],str] = None) -> Union[DataFrame,Dict[AnyStr,DataFrame]]:
        ### Profiler: 0.2% of DatasetDF() runtime
        labels = df['image_id'].values
        try:             output_df = cls.csv_data.loc[labels]
        except KeyError: output_df = cls.csv_data.loc[[]]         # test dataset
        output_df = output_df[Y_field] if Y_field else output_df

        if isinstance(output_df, Series) or len(output_df.columns) == 1:
            # single model output
            output = pd.get_dummies( output_df )
        else:
            # multi model output
            output = {
                column: pd.get_dummies( output_df[column] )
                for column in output_df.columns
            }
        return output


    # Source: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/
    # noinspection PyArgumentList
    @classmethod
    #@profile
    def transform_X(cls,
                    train: DataFrame,
                    resize=2,
                    invert=True,
                    rescale=True,
                    denoise=True,
                    center=True,
                    normalize=True,
    ) -> np.ndarray:
        ### Profiler: 78.7% of DatasetDF() runtime
        train = (train.drop(columns='image_id', errors='ignore')
                 .values.astype('uint8')                   # unit8 for initial data processing
                 .reshape(-1, 137, 236)                    # 2D arrays for inline image processing
                )
        # Colors   |   0 = black      | 255 = white
        # invert   |   0 = background | 255 = line
        # original | 255 = background |   0 = line

        # Invert for processing
        train = cls.invert(train)

        if resize:
            train = cls.resize(train, resize)

        if rescale:
            train = cls.rescale(train)

        if denoise:
            train = cls.denoise(train)

        if center:
            train = cls.center(train)

        if not invert:
            train = cls.invert(train)  # un-invert

        if normalize:
            train = cls.normalize(train)

        train = train.reshape(*train.shape, 1)        # 4D ndarray for tensorflow CNN

        gc.collect()  # ; sleep(1)
        return train


    @classmethod
    #@profile
    def invert(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 0.5% of DatasetDF() runtime
        return (255-train)


    @classmethod
    #@profile
    def normalize(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 15.4% of DatasetDF() runtime
        train = train.astype('float16') / 255.0   # prevent division cast: int -> float64
        return train


    @classmethod
    #@profile
    def denoise(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 0.3% of DatasetDF() runtime
        train = train * (train >= 42)  # 42 is the maximum mean
        return train


    @classmethod
    #@profile
    def rescale(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 3.4% of DatasetDF() runtime
        ### Rescale lines to maximum brightness, and set background values (less than 2x mean()) to 0
        ### max(mean()) =  14, 38,  33,  25, 36, 42,  20, 37,  38,  26, 36, 35
        ### min(max())  = 242, 94, 105, 224, 87, 99, 247, 85, 106, 252, 85, 97
        ### max(min())  =  0,   5,   3,   0,  6,  3,   0,  5,   3,   0,  6,  4
        # try:
        #     print('max mean()',  max([ np.mean(train[i]) for i in range(train.shape[0]) ]))
        #     print('min  max()',  min([ np.max(train[i]) for i in range(train.shape[0]) ]))
        #     print('max  min()',  max([ np.min(train[i]) for i in range(train.shape[0]) ]))
        # except: pass
        train = np.array([
            (train[i].astype('float64') * 255./train[i].max()).astype('uint8')
            for i in range(train.shape[0])
        ])
        return train


    @classmethod
    #@profile
    def resize(cls, train: np.ndarray, resize: int) -> np.ndarray:
        ### Profiler: 29% of DatasetDF() runtime  (37% with [for in] loop)
        # NOTEBOOK: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing/
        # Out of the different resize functions:
        # - np.mean(dtype=uint8) produces fragmented images (needs float16 to work properly - but RAM intensive)
        # - np.median() produces the most accurate downsampling - but returns float64
        # - np.max() produces an image with thicker lines       - occasionally produces bounding boxes
        # - np.min() produces a  image with thiner  lines       - harder to read
        if isinstance(resize, bool) and resize == True:
            resize = 2
        if resize and resize != 1:
            resize_fn = np.max
            if   len(train.shape) == 2: resize_shape =    (resize, resize)
            elif len(train.shape) == 3: resize_shape = (1, resize, resize)
            elif len(train.shape) == 4: resize_shape = (1, resize, resize, 1)
            else:                       resize_shape = (1, resize, resize, 1)

            train = skimage.measure.block_reduce(train, resize_shape, cval=0, func=resize_fn)
            # train = np.array([
            #     skimage.measure.block_reduce(train[i,:,:], (resize,resize), cval=0, func=resize_fn)
            #     for i in range(train.shape[0])
            # ])
        return train


    @classmethod
    #@profile
    def center(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 12.3% of DatasetDF() runtime
        ### NOTE: cls.crop_center_image assumes inverted
        train = np.array([
            cls.crop_center_image(train[i,:,:], cval=0, tol=42)
            for i in range(train.shape[0])
        ])
        return train


    # DOCS: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    # NOTE: assumes inverted
    @classmethod
    def crop_center_image(cls, img, cval=0, tol=0):
        ### Profiler: 11% of DatasetDF() runtime
        org_shape   = img.shape
        img_cropped = cls.crop_image_border_px(img, px=1)
        img_cropped = cls.crop_image_background(img_cropped, tol=tol)
        pad_x       = (org_shape[0] - img_cropped.shape[0]) / 2.0
        pad_y       = (org_shape[1] - img_cropped.shape[1]) / 2.0
        padding     = (
            (math.floor(pad_x), math.ceil(pad_x)),
            (math.floor(pad_y), math.ceil(pad_y))
        )
        img_center = np.pad(img_cropped, padding, 'constant', constant_values=cval)
        return img_center


    @classmethod
    def crop_image_border_px(cls, img, px=1):
        ### Profiler: 0.1% of DatasetDF() runtime
        ### crop one pixel border from the image to remove any bounding box effects
        img  = img[px:img.shape[0]-px, px:img.shape[1]-px]
        return img


    # Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    # This is the fast method that simply remove all empty rows/columns
    # NOTE: assumes inverted
    @classmethod
    def crop_image_background(cls, img, tol=0):
        ### Profiler: 4.7% of DatasetDF() runtime
        img  = img[1:img.shape[0]-1, 1:img.shape[1]-1]  # crop one pixel border from image to remove any bounding box
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]


#####
##### END   src/dataset/Transforms.py
#####

#####
##### START src/callbacks/KaggleTimeoutCallback.py
#####

import math
import re
import time
from typing import Union

import tensorflow as tf


class KaggleTimeoutCallback(tf.keras.callbacks.Callback):
    start_python = time.time()


    def __init__(self, timeout: Union[int, float, str], from_now=False, verbose=False):
        super().__init__()
        self.verbose           = verbose
        self.from_now          = from_now
        self.start_time        = self.start_python if not self.from_now else time.time()
        self.timeout_seconds   = self.parse_seconds(timeout)

        self.last_epoch_start  = time.time()
        self.last_epoch_end    = time.time()
        self.last_epoch_time   = self.last_epoch_end - self.last_epoch_start
        self.current_runtime   = self.last_epoch_end - self.start_time


    def on_train_begin(self, logs=None):
        self.check_timeout()  # timeout before first epoch if model.fit() is called again


    def on_epoch_begin(self, epoch, logs=None):
        self.last_epoch_start = time.time()


    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch_end  = time.time()
        self.last_epoch_time = self.last_epoch_end - self.last_epoch_start
        self.check_timeout()


    def check_timeout(self):
        self.current_runtime = self.last_epoch_end - self.start_time
        if self.verbose:
            print(f'\nKaggleTimeoutCallback({self.format(self.timeout_seconds)}) runtime {self.format(self.current_runtime)}')

        # Give timeout leeway of 2 * last_epoch_time
        if (self.current_runtime + self.last_epoch_time*2) >= self.timeout_seconds:
            print(f"\nKaggleTimeoutCallback({self.format(self.timeout_seconds)}) stopped after {self.format(self.current_runtime)}")
            self.model.stop_training = True


    @staticmethod
    def parse_seconds(timeout) -> int:
        if isinstance(timeout, (float,int)): return int(timeout)
        seconds = 0
        for (number, unit) in re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])?", str(timeout)):
            if   unit == 'd': seconds += float(number) * 60 * 60 * 24
            elif unit == 'h': seconds += float(number) * 60 * 60
            elif unit == 'm': seconds += float(number) * 60
            else:             seconds += float(number)
        return int(seconds)


    @staticmethod
    def format(seconds: Union[int,float]) -> str:
        runtime = {
            "d":   math.floor(seconds / (60*60*24) ),
            "h":   math.floor(seconds / (60*60)    ) % 24,
            "m":   math.floor(seconds / (60)       ) % 60,
            "s":   math.floor(seconds              ) % 60,
        }
        return " ".join([ f"{runtime[unit]}{unit}" for unit in ["h", "m", "s"] if runtime[unit] != 0 ])


#####
##### END   src/callbacks/KaggleTimeoutCallback.py
#####

#####
##### START src/dataset/DatasetDF.py
#####

import gc
import os
from typing import AnyStr, Dict, Union

import glob2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from src.dataset.Transforms import Transforms
# from src.settings import settings


class DatasetDF():
    csv_data = Transforms.csv_data


    #@profile
    def __init__(self,
                 test_train   = 'train',
                 data_id: Union[str,int] = '0',
                 size         = -1,
                 fraction     =  1,
                 split: float = 0.1,
                 Y_field      = None,
                 shuffle      = True,
                 transform_X_args = {},
                 transform_Y_args = {},
        ):
        gc.collect()

        self.test_train = test_train
        self.data_id    = data_id
        self.Y_field    = Y_field
        self.split      = split    if self.test_train is 'train' else 0
        self.shuffle    = shuffle  if self.test_train is 'train' else False
        self.fraction   = fraction if self.test_train is 'train' else 1
        self.size       = size

        self.parquet_filenames = sorted(glob2.glob(f"{settings['dir']['data']}/{test_train}_image_data_{data_id}.parquet"))

        self.X:  Dict[AnyStr, np.ndarray]               = { "train": np.ndarray((0,)), "valid": np.ndarray((0,)) }
        self.Y:  Dict[AnyStr, Union[pd.DataFrame,Dict]] = { "train": pd.DataFrame(),   "valid": pd.DataFrame()   }
        self.ID: Dict[AnyStr, np.ndarray]               = { "train": np.ndarray((0,)), "valid": np.ndarray((0,)) }
        for parquet_filename in self.parquet_filenames:
            raw = {
                'train': pd.read_parquet(parquet_filename),
                'valid': None
            }
            # Use size=1 to create a reference dataframe with valid .input_size() + .output_size()
            if self.size > 0:
                raw['valid'] = raw['train'][size+1:size*2]
                raw['train'] = raw['train'][:size]
            else:
                if self.fraction < 1:
                    raw['train'], discard      = train_test_split(raw['train'], train_size=self.fraction, shuffle=self.shuffle)
                    del discard
                if self.split != 0:
                    raw['train'], raw['valid'] = train_test_split(raw['train'], test_size=self.split,     shuffle=self.shuffle, random_state=0)

            if raw['valid'] is None:
                raw['valid'] = pd.DataFrame(columns=raw['train'].columns)

            # Attempt to save memory by doing transform_X() within the loop
            # X can be transformed before np.concatenate, but multi-output Y must be done after pd.concat()
            for key, value in raw.items():
                X = Transforms.transform_X(value, **transform_X_args)
                if len(self.X[key]) == 0: self.X[key] = X
                else:                     self.X[key] = np.concatenate([ self.X[key], X ])
                self.Y[key]  = pd.concat([      self.Y[key],  value[['image_id']]       ])
                self.ID[key] = np.concatenate([ self.ID[key], value['image_id'].values  ])
            del raw; gc.collect()

        self.Y = {
            key: Transforms.transform_Y(value, **transform_Y_args)
            for key,value in self.Y.items()
        }
        pass


    def epoch_size(self):
        return self.X['train'].shape[0]


    def input_shape(self):
        return self.X['train'].shape[1:]  # == (137, 236, 1) / 2


    @classmethod
    def output_shape(cls, Y_field=None):
        if isinstance(Y_field, str):
            return cls.csv_data[Y_field].nunique()

        csv_data     = cls.csv_data[Y_field] if Y_field else cls.csv_data
        output_shape = (csv_data.drop(columns='image_id', errors='ignore')
                                .nunique()
                                .to_dict())
        return output_shape




if __name__ == '__main__' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    ### NOTE: loading all datasets at once exceeds 12GB RAM and crashes Python (on 16GB RAM machine)
    ### Runtime: 3m 12s - for in range(0,4)
    ### $ find ./src/ -name '*.py' | xargs perl -p -i -e 's/#@profile/@profile/'
    ### $ time python3 -m memory_profiler src/dataset/DatasetDF.py | less
    #@profile()
    def main():
        for data_id in range(0,4):
            for test_train in ['test', 'train']:
                dataset = DatasetDF(test_train=test_train, data_id=data_id, fraction=1)
                Y_shape = {}
                for key, Y in dataset.Y.items():
                    if isinstance(Y, dict): Y_shape[key] = { k:v.shape for k,v in Y.items() }
                    else:                   Y_shape[key] = Y.shape

                print(f"{test_train}:{data_id} dataset.image_filenames", dataset.parquet_filenames)
                print(f"{test_train}:{data_id} dataset.X",               { key: df.shape for key, df in dataset.X.items() })
                print(f"{test_train}:{data_id} dataset.Y", Y_shape)
                print(f"{test_train}:{data_id} dataset.input_shape()",   dataset.input_shape())
                print(f"{test_train}:{data_id} dataset.output_shape()",  dataset.output_shape())
                print(f"{test_train}:{data_id} dataset.epoch_size()",    dataset.epoch_size())
    main()

#####
##### END   src/dataset/DatasetDF.py
#####

#####
##### START src/util/logs.py
#####

import os
import time
from datetime import datetime
from typing import Dict, Union

import humanize
import simplejson

# from src.settings import settings



def model_stats_from_history(history, timer_seconds=0, best_only=False) -> Union[None, Dict]:
    if 'val_loss' in history.history:
        best_epoch            = history.history['val_loss'].index(min( history.history['val_loss'] )) if best_only else -1
        model_stats           = { key: value[best_epoch] for key, value in history.history.items() }
        model_stats['time']   = timer_seconds
        model_stats['epochs'] = len(history.history['loss'])
    else:
        model_stats = None
    return model_stats


python_start = time.time()
def log_model_stats(model_stats, logfilename, model_hparams, train_hparams):
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    with open(logfilename, 'w') as file:
        output = [
            "------------------------------",
            f"Completed",
            "------------------------------",
            f"model_hparams: {model_hparams}",
            f"train_hparams: {train_hparams}",
            "------------------------------",
        ]
        output += [ f"settings[{key}]: {value}" for key, value in settings.items() ]
        output.append("------------------------------")

        if isinstance(model_stats, dict):
            output.append(simplejson.dumps(
                { key: str(value) for key, value in model_stats.items() },
                sort_keys=False, indent=4*' '
            ))
        elif isinstance(model_stats, list):
            output += list(map(str, model_stats))
        else:
            output.append( str(model_stats) )

        output.append("------------------------------")
        output += [
            f"------------------------------",
            f"script started: { datetime.fromtimestamp( python_start ).strftime('%Y-%m-%d %H:%M:%S')}",
            f"script ended:   { datetime.fromtimestamp( time.time()  ).strftime('%Y-%m-%d %H:%M:%S')}",
            f"script runtime: { humanize.naturaldelta(  python_start - time.time() )}s",
            f"------------------------------",
        ]
        output = "\n".join(output)
        print(      output )
        file.write( output )
        print("wrote:", logfilename)


#####
##### END   src/util/logs.py
#####

#####
##### START src/vendor/CLR/clr_callback.py
#####

from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
import numpy as np

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


#####
##### END   src/vendor/CLR/clr_callback.py
#####

#####
##### START src/dataset/ParquetImageDataGenerator.py
#####

# Notebook: https://www.kaggle.com/jamesmcguigan/reading-parquet-files-ram-cpu-optimization/
# Notebook: https://www.kaggle.com/jamesmcguigan/bengali-ai-image-processing
import gc
import math
from collections import Callable

import glob2
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from pyarrow.parquet import ParquetFile


class ParquetImageDataGenerator(ImageDataGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def flow_from_parquet(
            self,
            glob_path:       str,
            transform_X:     Callable,
            transform_Y:     Callable,
            transform_X_args = {},
            transform_Y_args = {},
            batch_size       = 32,
            reads_per_file   = 2,
            resamples        = 1,
            shuffle          = False,
            infinite         = True,
            test             = False,
    ):
        """
            Source: ./venv/lib/python3.6/site-packages/keras_preprocessing/image/image_data_generator.py
            # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a numpy array of image data
                (in the case of a single image input) or a list
                of numpy arrays (in the case with
                additional inputs) and `y` is a numpy array
                of corresponding labels. If 'sample_weight' is not None,
                the yielded tuples are of the form `(x, y, sample_weight)`.
                If `y` is None, only the numpy array `x` is returned.
        """
        if test:
            shuffle  = False
            infinite = False

        for (X,Y) in self.cache_XY_generator(
                glob_path=glob_path,
                transform_X=transform_X,
                transform_X_args=transform_X_args,
                transform_Y=transform_Y,
                transform_Y_args=transform_Y_args,
                reads_per_file=reads_per_file,
                resamples=resamples,
                shuffle=shuffle,
                infinite=infinite,
        ):
            cache_size  = X.shape[0]
            batch_count = math.ceil( cache_size / batch_size )
            for n_batch in range(batch_count):
                X_batch = X[ batch_size * n_batch : batch_size * (n_batch+1) ].copy()
                if isinstance(Y, dict):
                    Y_batch = {
                        key: Y[key][ batch_size * n_batch : batch_size * (n_batch+1) ].copy()
                        for key in Y.keys()
                    }
                else:
                    Y_batch = Y[ batch_size * n_batch : batch_size * (n_batch+1) ].copy()
                yield ( X_batch, Y_batch )


    @classmethod
    def cache_XY_generator(
            cls,
            glob_path:        str,
            transform_X:      Callable,
            transform_X_args: {},
            transform_Y:      Callable,
            transform_Y_args: {},
            reads_per_file  = 3,
            resamples       = 1,
            shuffle         = False,
            infinite        = False,
    ):
        for cache in cls.cache_generator(
                glob_path=glob_path,
                reads_per_file=reads_per_file,
                resamples=resamples,
                shuffle=shuffle,
                infinite=infinite,
        ):
            X = transform_X(cache, **transform_X_args)
            Y = transform_Y(cache, **transform_Y_args)
            yield (X, Y)


    @classmethod
    def cache_generator(
            cls,
            glob_path,
            reads_per_file = 3,
            resamples      = 1,
            shuffle        = False,
            infinite       = False,
    ):
        filenames = sorted(glob2.glob(glob_path))
        if len(filenames) == 0: raise Exception(f"{cls.__name__}.batch_generator() - invalid glob_path: {glob_path}")

        gc.collect();  # sleep(1)   # sleep(1) is required to allow measurement of the garbage collector
        while True:
            for filename in filenames:
                num_rows    = ParquetFile(filename).metadata.num_rows
                cache_size  = math.ceil( num_rows / reads_per_file )
                for n_read in range(reads_per_file):
                    gc.collect();  # sleep(1)   # sleep(1) is required to allow measurement of the garbage collector
                    cache = (
                        pd.read_parquet(filename)
                            # .set_index('image_id', drop=True)  # WARN: Don't do this, it breaks other things
                            .iloc[ cache_size * n_read : cache_size * (n_read+1) ]
                            .copy()
                    )
                    for resample in range(resamples):
                        if shuffle:
                            cache = cache.sample(frac=1)
                        yield cache
            if not infinite: break

#####
##### END   src/dataset/ParquetImageDataGenerator.py
#####

#####
##### START src/models/MultiOutputCNN.py
#####

import inspect
import types
from typing import Dict, List, Union, cast

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPooling2D,
    MaxPooling2D,
)


# noinspection DuplicatedCode
def MultiOutputCNN(
        input_shape,
        output_shape: Union[List, Dict],
        cnns_per_maxpool=1,
        maxpool_layers=1,
        dense_layers=1,
        dense_units=64,
        dropout=0.25,
        regularization=False,
        global_maxpool=False,
        name='',
)  -> Model:
    function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name
    model_name    = f"{function_name}-{name}" if name else function_name
    # model_name  = seq([ function_name, name ]).filter(lambda x: x).make_string("-")  # remove dependency on pyfunctional - not in Kaggle repo without internet

    inputs = Input(shape=input_shape)
    x      = inputs

    for cnn1 in range(1,maxpool_layers+1):
        for cnn2 in range(1, cnns_per_maxpool+1):
            x = Conv2D( 32 * cnn1, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    if global_maxpool:
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)

    for nn1 in range(0,dense_layers):
        if regularization:
            x = Dense(dense_units, activation='relu',
                      kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.01))(x)
        else:
            x = Dense(dense_units, activation='relu')(x)

        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

    x = Flatten(name='output')(x)

    if isinstance(output_shape, dict):
        outputs = [
            Dense(output_shape, activation='softmax', name=key)(x)
            for key, output_shape in output_shape.items()
        ]
    else:
        outputs = [
            Dense(output_shape, activation='softmax', name=f'output_{index}')(x)
            for index, output_shape in enumerate(output_shape)
        ]

    model = Model(inputs, outputs, name=model_name)
    # plot_model(model, to_file=os.path.join(os.path.dirname(__file__), f"{name}.png"))
    return model


#####
##### END   src/models/MultiOutputCNN.py
#####

#####
##### START src/util/argparse.py
#####

import argparse
import copy
from typing import Dict, List



def argparse_from_dicts(configs: List[Dict], inplace=False) -> List[Dict]:
    parser = argparse.ArgumentParser()
    for config in list(configs):
        for key, value in config.items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', action='store_true', default=value, help=f'{key} (default: %(default)s)')
            else:
                parser.add_argument(f'--{key}', type=type(value),    default=value, help=f'{key} (default: %(default)s)')


    args, unknown = parser.parse_known_args()  # Ignore extra CLI args passed in by Kaggle

    outputs = configs if inplace else copy.deepcopy(configs)
    for index, output in enumerate(outputs):
        for key, value in outputs[index].items():
            outputs[index][key] = getattr(args, key)

    return outputs


def argparse_from_dict(config: Dict, inplace=False):
    return argparse_from_dicts([config], inplace)[0]


#####
##### END   src/util/argparse.py
#####

#####
##### START src/util/csv.py
#####

import os
from itertools import chain

import gc
import numpy as np
import pandas as pd
from pandas import DataFrame

# from src.dataset.DatasetDF import DatasetDF
# from src.dataset.ParquetImageDataGenerator import ParquetImageDataGenerator
# from src.dataset.Transforms import Transforms
# from src.settings import settings



### BUGFIX: Repeatedly calling model.predict(...) results in memory leak - https://github.com/keras-team/keras/issues/13118
def submission_df(model, output_shape):
    gc.collect()

    submission = pd.DataFrame(columns=output_shape.keys())
    # large datasets on submit, so loop
    for data_id in range(0,4):
        test_dataset      = DatasetDF(test_train='test', data_id=data_id, transform_X_args = {} )  # "normalize": True is default
        test_dataset_rows = test_dataset.X['train'].shape[0]
        batch_size        = 64
        for index in range(0, test_dataset_rows, batch_size):
            try:
                X_batch     = test_dataset.X['train'][index : index+batch_size]
                predictions = model.predict_on_batch(X_batch)
                # noinspection PyTypeChecker
                submission = submission.append(
                    pd.DataFrame({
                        key: np.argmax( predictions[index], axis=-1 )
                        for index, key in enumerate(output_shape.keys())
                        }, index=test_dataset.ID['train'])
                    )
            except Exception as exception:
                print('submission_df_generator()', exception)

        return submission

###
### Use submission_df() it seems to have more success on Kaggle
###
def submission_df_generator(model, output_shape):
    gc.collect()

    # if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive') == 'Interactive':
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        globpath = f"{settings['dir']['data']}/train_image_data_*.parquet"
    else:
        globpath = f"{settings['dir']['data']}/test_image_data_*.parquet"

    # large datasets on submit, so loop via generator to avoid Out-Of-Memory errors
    submission  = pd.DataFrame(columns=output_shape.keys())
    cache_index = 0
    for cache in ParquetImageDataGenerator.cache_generator(
            globpath,
            reads_per_file = 2,
            resamples      = 1,
            shuffle        = False,
            infinite       = False,
    ):
        try:
            cache_index      += 1
            batch_size        = 128
            test_dataset_rows = cache.shape[0]
            print(f'submission_df_generator() - submission: ', cache_index, submission.shape)
            if test_dataset_rows == 0: continue
            for index in range(0, test_dataset_rows, batch_size):
                try:
                    batch = cache[index : index+batch_size]
                    if batch.shape[0] == 0: continue
                    X           = Transforms.transform_X(batch)  # normalize=True is default
                    predictions = model.predict_on_batch(X)
                    submission  = submission.append(
                        pd.DataFrame({
                            key: np.argmax( predictions[index], axis=-1 )
                            for index, key in enumerate(output_shape.keys())
                        }, index=batch['image_id'])
                    )
                except Exception as exception:
                    print('submission_df_generator() - batch', exception)
        except Exception as exception:
            print('submission_df_generator() - cache', exception)

    return submission



# def df_to_submission(df: DataFrame) -> DataFrame:
#     print('df_to_submission() - input', df.shape)
#     output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
#     submission = DataFrame(columns=['row_id', 'target'])
#     for index, row in df.iterrows():
#         for output_field in output_fields:
#             try:
#                 index = f"Test_{index}" if not str(index).startswith('T') else index
#                 submission = submission.append({
#                     'row_id': f"{index}_{output_field}",
#                     'target': df[output_field].loc[index],
#                     }, ignore_index=True)
#             except Exception as exception:
#                 print('df_to_submission()', exception)
#     print('df_to_submission() - output', submission.shape)
#     return submission


# def df_to_submission(df: DataFrame) -> DataFrame:
#     print('df_to_submission_columns() - input', df.shape)
#     output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
#     submissions = {}
#     for output_field in output_fields:
#         if 'image_id' in df.columns:
#             submissions[output_field] = DataFrame({
#                 'row_id': df['image_id'] + '_' + output_field,
#                 'target': df[output_field],
#             })
#         else:
#             submissions[output_field] = DataFrame({
#                 'row_id': df.index + '_' + output_field,
#                 'target': df[output_field],
#             })
#
#     # Kaggle - Order of submission.csv IDs matters - https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69366
#     submission = DataFrame(pd.concat(submissions.values()))
#     submission['sort'] = submission['row_id'].apply(lambda row_id: int(re.sub(r'\D', '', row_id)) )
#     submission = submission.sort_values(by=['sort','row_id'])
#     submission = submission.drop(columns=['sort'])
#
#     print('df_to_submission_columns() - output', submission.shape)
#     return submission


def df_to_submission(df: DataFrame) -> DataFrame:
    print('df_to_submission_columns() - input', df.shape)
    output_fields = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
    if 'image_id' not in df.columns:
        df['image_id'] = df.index

    submission_rows = df.apply(lambda row: [{
        'row_id': row['image_id'] + '_' + output_field,
        'target': row[output_field],
    } for output_field in output_fields], axis=1, result_type='reduce' )

    submission = DataFrame(chain(*submission_rows.values))   # Hopefully in original sort order

    print('df_to_submission_columns() - output', submission.shape)
    return submission



def df_to_submission_csv(df: DataFrame, filename: str):
    submission = df_to_submission(df)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        submission.to_csv('submission.csv', index=False)
        print("wrote:", 'submission.csv', submission.shape)
    else:
        submission.to_csv(filename, index=False)
        print("wrote:", filename, submission.shape)


#####
##### END   src/util/csv.py
#####

#####
##### START src/util/hparam.py
#####

import math
import os
import re
import time
from typing import AnyStr, Dict, Union

import tensorflow as tf
from tensorboard.plugins.hparams.api import KerasCallback
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

# from src.callbacks.KaggleTimeoutCallback import KaggleTimeoutCallback
# from src.dataset.DatasetDF import DatasetDF
# from src.settings import settings
# from src.util.logs import model_stats_from_history
# from src.vendor.CLR.clr_callback import CyclicLR



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging
# tf.keras.backend.set_floatx('float16')  # Potentially causes problems with Tensortflow


def hparam_key(hparams):
    return "-".join( f"{key}={value}" for key,value in sorted(hparams.items()) ).replace(' ','')


def min_lr(hparams):
    # tensorboard --logdir logs/convergence_search/min_lr-optimized_scheduler-random-scheduler/ --reload_multifile=true
    # There is a high degree of randomness in this parameter, so it is hard to distinguish from statistical noise
    # Lower min_lr values for CycleCR tend to train slower
    hparams = { **settings['hparam_defaults'], **hparams }
    if 'min_lr'  in hparams:              return hparams['min_lr']
    if hparams["optimizer"] == "SGD":     return 1e-05  # preferred by SGD
    else:                                 return 1e-03  # fastest, least overfitting and most accidental high-scores


# DOCS: https://ruder.io/optimizing-gradient-descent/index.html
def scheduler(hparams: dict, dataset: DatasetDF, verbose=False):
    hparams = { **settings['hparam_defaults'], **hparams }
    if hparams['scheduler'] is 'constant':
        return LearningRateScheduler(lambda epocs: hparams['learning_rate'], verbose=False)

    if hparams['scheduler'] is 'linear_decay':
        return LearningRateScheduler(
            lambda epocs: max(
                hparams['learning_rate'] * (10. / (10. + epocs)),
                min_lr(hparams)
            ),
            verbose=verbose
        )

    if hparams['scheduler'].startswith('CyclicLR') \
            or hparams['scheduler'] in ["triangular", "triangular2", "exp_range"]:
        # DOCS: https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
        # CyclicLR_triangular, CyclicLR_triangular2, CyclicLR_exp_range
        mode = re.sub(r'^CyclicLR_', '', hparams['scheduler'])

        # step_size should be epoc multiple between 2 and 8, but multiple of 2 (= full up/down cycle)
        if   hparams['patience'] <=  6: whole_cycles = 1   #  1/2   = 0.5  | 6/2    = 3
        elif hparams['patience'] <= 12: whole_cycles = 2   #  8/4   = 2    | 12/4   = 3
        elif hparams['patience'] <= 24: whole_cycles = 3   # 14/6   = 2.3  | 24/6   = 4
        elif hparams['patience'] <= 36: whole_cycles = 4   # 26/8   = 3.25 | 36/8   = 4.5
        elif hparams['patience'] <= 48: whole_cycles = 5   # 28/10  = 2.8  | 48/10  = 4.8
        elif hparams['patience'] <= 72: whole_cycles = 6   # 50/12  = 4.2  | 72/12  = 6
        elif hparams['patience'] <= 96: whole_cycles = 8   # 74/16  = 4.6  | 96/16  = 6
        else:                           whole_cycles = 12  # 100/24 = 4.2  | 192/24 = 8

        return CyclicLR(
            mode      = mode,
            step_size =dataset.epoch_size() * (hparams['patience'] / (2.0 * whole_cycles)),
            base_lr   = min_lr(hparams),
            max_lr    = hparams['learning_rate']
        )

    if hparams['scheduler'].startswith('plateau'):
        factor = int(( re.findall(r'\d+', hparams['scheduler']) + [10] )[0])            # plateau2      || plateau10 (default)
        if 'sqrt' in hparams['scheduler']:  patience = math.sqrt(hparams['patience'])  # plateau2_sqrt || plateau10__sqrt
        else:                               patience = hparams['patience'] / 2.0

        return ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 1 / factor,
            patience = math.floor(patience),
            min_lr   = 0,   # min_lr(train_hparams),
            verbose  = verbose,
        )

    print("Unknown scheduler: ", hparams)


def losses(output_shape):
    if   isinstance(output_shape, list): losses = [ categorical_crossentropy      for n   in output_shape        ]
    elif isinstance(output_shape, dict): losses = { key: categorical_crossentropy for key in output_shape.keys() }
    else:                                losses = categorical_crossentropy
    return losses


def loss_weights(output_shape):
    # unique = dataset.apply(lambda col: col.nunique()); unique
    # grapheme_root           168   | sqrt = 12.9 / 54.9 = 0.24
    # vowel_diacritic          11   | sqrt =  3.3 / 54.9 = 0.06
    # consonant_diacritic       7   | sqrt =  2.6 / 54.9 = 0.05
    # grapheme               1295   | sqrt = 35.9 / 54.9 = 0.65
    if not isinstance(output_shape, dict): return None
    norm    = sum(map(math.sqrt, output_shape.values()))
    weights = {
        key: math.sqrt(value)/norm
        for key,value in output_shape.items()
    }
    return weights



def callbacks(hparams, dataset, model_file=None, log_dir=None, best_only=True, verbose=False ):
    schedule  = scheduler(hparams, dataset, verbose=verbose)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=verbose,
            patience=hparams.get('patience', 10),
            restore_best_weights=best_only
        ),
        schedule,
    ]
    if hparams.get("timeout"):
        callbacks += [
            KaggleTimeoutCallback( hparams.get("timeout"), verbose=False ),
        ]
    if model_file:
        callbacks += [
            ModelCheckpoint(
                model_file,
                monitor='val_loss',
                verbose=False,
                save_best_only=best_only,
                save_weights_only=False,
                mode='auto',
            )
        ]
    if log_dir and settings['verbose']['tensorboard'] and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        callbacks += [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),  # log metrics
            KerasCallback(log_dir, hparams)                                     # log train_hparams
        ]
    return callbacks



def model_compile(
        hparams:      Dict,
        model:        tf.keras.models.Model,
        output_shape: Union[None, int, Dict] = None,
    ):
    hparams   = { **settings['hparam_defaults'], **hparams }
    optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])
    loss      = losses(output_shape)
    weights   = loss_weights(output_shape) if hparams.get('loss_weights') else None

    model.compile(
        loss=loss,
        loss_weights=weights,
        optimizer=optimiser(learning_rate=hparams.get('learning_rate', 0.001)),
        metrics=['accuracy']
    )
    return model


def model_compile_fit(
        hparams:      Dict,
        model:        tf.keras.models.Model,
        dataset:      DatasetDF,
        epochs      = 999,
        output_shape: Union[None, int, Dict] = None,
        model_file:   AnyStr = None,
        log_dir:      AnyStr = None,
        best_only   = True,
        verbose     = settings['verbose']['fit'],
):
    timer_start = time.time()

    hparams = { **settings['hparam_defaults'], **hparams }
    model   = model_compile( hparams, model, output_shape )

    callback = callbacks(hparams, dataset, model_file, log_dir, best_only, verbose)
    history  = model.fit(
        dataset.X["train"], dataset.Y["train"],
        batch_size=hparams.get("batch_size", 128),
        epochs=epochs,
        verbose=verbose,
        validation_data=(dataset.X["valid"], dataset.Y["valid"]),
        callbacks=callback
    )
    timer_seconds = int(time.time() - timer_start)

    model_stats = model_stats_from_history(history, timer_seconds, best_only)
    return model_stats


#####
##### END   src/util/hparam.py
#####

#####
##### START src/pipelines/multi_output_df_cnn.py
#####

#!/usr/bin/env python

import os

import glob2

# from src.dataset.DatasetDF import DatasetDF
# from src.models.MultiOutputCNN import MultiOutputCNN
# from src.settings import settings
# from src.util.argparse import argparse_from_dicts
# from src.util.csv import df_to_submission_csv, submission_df_generator
# from src.util.hparam import hparam_key, model_compile_fit
# from src.util.logs import log_model_stats



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3 # Disable Tensortflow Logging

# NOTE: This line doesn't work on Kaggle
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
# [ tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU') ]


def multi_output_df_cnn(train_hparams, model_hparams, pipeline_name):
    print("pipeline_name", pipeline_name)
    print("train_hparams", train_hparams)
    print("model_hparams", model_hparams)

    model_hparams_key = hparam_key(model_hparams)
    train_hparams_key = hparam_key(train_hparams)

    # csv_data    = pd.read_csv(f"{settings['dir']['data']}/train.csv")
    model_file  = f"{settings['dir']['models']}/{pipeline_name}/{pipeline_name}-{model_hparams_key}.hdf5"
    log_dir     = f"{settings['dir']['logs']}/{pipeline_name}/{model_hparams_key}/{train_hparams_key}"

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(log_dir,                     exist_ok=True)

    # output_shape = csv_data.drop(columns='image_id').nunique().to_dict()
    input_shape  = DatasetDF(test_train='train', fraction=0.0001, data_id=0).input_shape()
    output_shape = DatasetDF.output_shape()
    model = MultiOutputCNN(
        input_shape=input_shape,
        output_shape=output_shape,
        **model_hparams,
    )

    # Load Pre-existing weights
    if os.path.exists( model_file ):
        try:
            model.load_weights( model_file )
            print('Loaded Weights: ', model_file)
        except Exception as exception: print('exception', exception)

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        load_models = glob2.glob(f'../input/**/{model_file}')
        for load_model in load_models:
            try:
                model.load_weights( load_model )
                print('Loaded Weights: ', load_model)
                break
            except Exception as exception: print('exception', exception)

    model.summary()

    model_stats = []
    for loop in range(train_hparams['loops']):
        for data_id in range(0,4):
            print("------------------------------")
            print(f"Training | data_id: {data_id}")
            print(f"model_hparams: {model_hparams}")
            print(f"train_hparams: {train_hparams}")
            print("------------------------------")

            dataset = DatasetDF(
                test_train='train',
                data_id=data_id,
                split=train_hparams['split'],
                fraction=train_hparams['fraction'],
            )

            stats = model_compile_fit(
                hparams      = {**train_hparams, **model_hparams},
                model        = model,
                dataset      = dataset,
                output_shape = output_shape,
                model_file   = model_file,
                log_dir      = log_dir,
                best_only    = True,
                verbose      = 2,
                epochs       = train_hparams.get('epochs',99),
            )
            if stats is None: break  # KaggleTimeoutCallback() triggered on_train_begin()
            model_stats.append(stats)
        else: continue
        break                        # KaggleTimeoutCallback() triggered on_train_begin()

    return model, model_stats, output_shape



if __name__ == '__main__':
    # Fastest ImageDataGenerator CNN with high score
    # - maxpool_layers=5 | cnns_per_maxpool=3 | dense_layers=1 | dense_units=256 | global_maxpool=False | regularization=False
    model_hparams = {
        "cnns_per_maxpool":   3,
        "maxpool_layers":     5,
        "dense_layers":       1,
        "dense_units":      256,
        "regularization": False,
        "global_maxpool": False,
    }
    train_hparams = {
        "optimizer":     "RMSprop",
        "scheduler":     "constant",
        "learning_rate": 0.001,

        # "optimizer":     "Adagrad",
        # "scheduler":     "plateau10",
        # "learning_rate": 1,
        "patience":      10,

        # "min_lr":        0.001,
        # "split":         0.2,
        # "batch_size":    128,
        "fraction":      1,   # Reduce memory overhead, but do 4 loops
        "loops":         3,
        "epochs":        99,
        "loss_weights":  False,
    }
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive':
        train_hparams['patience'] = 0
        train_hparams['loops']    = 1
    train_hparams = { **settings['hparam_defaults'], **train_hparams }

    argparse_from_dicts([train_hparams, model_hparams], inplace=True)


    pipeline_name     = "multi_output_df_cnn"
    model_hparams_key = hparam_key(model_hparams)
    train_hparams_key = hparam_key(train_hparams)
    logfilename       = f"{settings['dir']['submissions']}/{pipeline_name}/{model_hparams_key}-submission.log"
    csv_filename      = f"{settings['dir']['submissions']}/{pipeline_name}/{model_hparams_key}-submission.csv"

    model, model_stats, output_shape = multi_output_df_cnn(train_hparams, model_hparams, pipeline_name)

    log_model_stats(model_stats, logfilename, model_hparams, train_hparams)

    submission = submission_df_generator(model, output_shape)
    df_to_submission_csv( submission, csv_filename )



#####
##### END   src/pipelines/multi_output_df_cnn.py
#####

##### 
##### ./kaggle_compile.py src/pipelines/multi_output_df_cnn.py --commit
##### 
##### 2020-03-31 18:16:41+01:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master f0d1273 [ahead 3] multi_output_df_cnn | model_hparams = Fastest ImageDataGenerator CNN
##### 
##### f0d1273b1ca3b899345796760abd962c6d6b8e16
##### 