#!/usr/bin/env python3

##### 
##### ./kaggle_compile.py src/preprocessing/write_images_to_filesystem.py --commit
##### 
##### 2020-03-16 02:55:38+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master dd50187 [ahead 4] write_images_to_filesystem | fix verbose
##### 
##### dd501876d997cd481c49712229e7b1589f6b28e1
##### 

#####
##### START src/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os

settings = {}

settings['hparam_defaults'] = {
    "optimizer":     "RMSprop",
    "scheduler":     "constant",
    "learning_rate": 0.001,
    "min_lr":        0.001,
    "split":         0.2,
    "batch_size":    128,
    "fraction":      1.0,
    "patience": {
        'Localhost':    5,
        'Interactive':  0,
        'Batch':        5,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')],
    "loops": {
        'Localhost':   1,
        'Interactive': 1,
        'Batch':       1,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')],

    # Timeout = 120 minutes | allow 30 minutes for testing submit | TODO: unsure of KAGGLE_KERNEL_RUN_TYPE on Submit
    "timeout": "5m" if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == "Interactive" else "90m"
}

settings['verbose'] = {
    "tensorboard": {
        {
            'Localhost':   True,
            'Interactive': False,
            'Batch':       False,
        }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
    },
    "fit": {
        'Localhost':   1,
        'Interactive': 2,
        'Batch':       2,
    }[os.environ.get('KAGGLE_KERNEL_RUN_TYPE','Localhost')]
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
for dirname in settings['dir'].values(): os.makedirs(dirname, exist_ok=True)



#####
##### END   src/settings.py
#####

#####
##### START src/dataset/transforms.py
#####

import gc
import math
from time import sleep
from typing import AnyStr, Dict, Union, List

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
        gc.collect(); sleep(1)

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

        gc.collect(); sleep(1)
        return train


    @classmethod
    def invert(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 0.5% of DatasetDF() runtime
        return (255-train)


    @classmethod
    def normalize(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 15.4% of DatasetDF() runtime
        train = train.astype('float16') / 255.0   # prevent division cast: int -> float64
        return train


    @classmethod
    def denoise(cls, train: np.ndarray) -> np.ndarray:
        ### Profiler: 0.3% of DatasetDF() runtime
        train = train * (train >= 42)  # 42 is the maximum mean
        return train


    @classmethod
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
##### END   src/dataset/transforms.py
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
##### START src/preprocessing/write_images_to_filesystem.py
#####

#!/usr/bin/env python
import copy
import os
import time

import glob2
import matplotlib
import matplotlib.image
import pandas as pd
from pyarrow.parquet import ParquetFile

# from src.dataset.transforms import Transforms
# from src.settings import settings
# from src.util.argparse import argparse_from_dicts


# Entries into the Bengali AI Competition often suffer from out of memory errors when reading from a dataframe
# Quick and dirty solution is to write data as images to a directory and use ImageDataGenerator.flow_from_directory()
# noinspection PyDefaultArgument
def write_images_to_filesystem( data_dir, feature_dir, ext='png', only=None, verbose=False, force=False, transform_args={} ):
    transform_defaults = { 'resize': 2, 'denoise': True, 'center': True, 'invert': True, 'normalize': False }
    transform_args     = { **transform_defaults, **transform_args }

    time_start = time.time()
    filename_groups = {
        "test":  sorted(glob2.glob(f"{settings['dir']['data']}/test_image_data_*.parquet")),
        "train": sorted(glob2.glob(f"{settings['dir']['data']}/train_image_data_*.parquet")),
    }
    if only:
        for test_train_valid, parquet_filenames in list(filename_groups.items()):
            if only not in test_train_valid:
                del filename_groups[test_train_valid]

    image_count = 0
    for test_train_valid, parquet_filenames in filename_groups.items():
        image_dir = f'{feature_dir}/{test_train_valid}'
        os.makedirs(image_dir, exist_ok=True)

        # Skip image creation if all images have already been extracted
        if not force:
            expected_images = sum([ ParquetFile(file).metadata.num_rows for file in parquet_filenames ])
            existing_images = len(glob2.glob(f'{image_dir}/*.{ext}'))
            if existing_images == expected_images: continue

        for parquet_filename in parquet_filenames:
            if verbose:
                print(f'write_images_to_filesystem({only or ""}) - reading:  ', parquet_filename)

            dataframe  = pd.read_parquet(parquet_filename)
            image_ids  = dataframe['image_id'].tolist()
            image_data = Transforms.transform_X(dataframe, **transform_args )

            for index, image_id in enumerate(image_ids):
                image_filename = f'{image_dir}/{image_id}.{ext}'
                if not force and os.path.exists(image_filename):
                    print(f'write_images_to_filesystem({only or ""}) - skipping: ', image_filename)
                    continue

                matplotlib.image.imsave(image_filename, image_data[index].squeeze(), cmap='gray')
                image_count += 1
                if verbose == 1:
                    print(f'write_images_to_filesystem({only or ""}) - wrote:    ', image_filename)

    if verbose:
        print( f'write_images_to_filesystem({only or ""}) - wrote: {image_count} files in: {round(time.time() - time_start,2)}s')



if __name__ == '__main__':
    args = {
        'data_dir':    settings['dir']['data'],
        'feature_dir': settings['dir']['features'],
        'ext':         'png',
        'verbose':      2,
        'force':        False,
    }
    transform_args = {
        'resize':    2,
        'rescale':   1,
        'denoise':   1,
        'center':    1,
        'invert':    1,
        'normalize': 0
    }
    argparse_from_dicts([args, transform_args], inplace=True)

    test_args = copy.deepcopy(args)
    test_args['force']   = True
    test_args['verbose'] = True

    write_images_to_filesystem(only='test',  transform_args=transform_args, **test_args )
    write_images_to_filesystem(only='train', transform_args=transform_args, **args )


#####
##### END   src/preprocessing/write_images_to_filesystem.py
#####

##### 
##### ./kaggle_compile.py src/preprocessing/write_images_to_filesystem.py --commit
##### 
##### 2020-03-16 02:55:38+00:00
##### 
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (fetch)
##### origin	git@github.com:JamesMcGuigan/kaggle-bengali-ai.git (push)
##### 
##### * master dd50187 [ahead 4] write_images_to_filesystem | fix verbose
##### 
##### dd501876d997cd481c49712229e7b1589f6b28e1
##### 