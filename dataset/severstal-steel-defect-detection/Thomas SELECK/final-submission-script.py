# As the competition does not allow commit with the kernel that uses internet connection, we use offline installation
import subprocess
subprocess.run(["python", "../input/mlcomp/mlcomp/mlcomp/setup.py"])

import warnings
warnings.filterwarnings('ignore')
import os

import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm_notebook
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap

###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file is the main entry point of the solution.                          #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-10-13                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import sys

# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode = None):
        if mode is None: 
            mode = "w"

        self.file = open(file, mode)

    def write(self, message, is_terminal = 1, is_file = 1):
        if "\r" in message: 
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file contains the code needed to convert masks to RLE format and vice  #
# versa.                                                                      #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-06-26                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np

class RLEFormatConverter(object):
    """
    This static class contains the tools needed to convert images masks to RLE format
    and vice versa.
    """

    @staticmethod
    def mask2rle(mask):
        """
        This method converts a mask to its RLE representation.

        Parameters
        ----------
        mask: numpy array
                Mask that needs to be converted.
                                
        Returns
        -------
        rle_string: string
                Mask converted to RLE.
        """

        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        rle_string = " ".join(str(x) for x in runs)

        return rle_string

    @staticmethod
    def rle2mask(rle_string, width, height, fill_value = 1):
        """
        This method converts a RLE string to its mask representation.

        Parameters
        ----------
        rle_string: string
                Mask converted to RLE.

        width: integer
                Width of the image in pixels.

        height: integer
                Height of the image in pixels.

        fill_value: integer (default = 1)
                Value used to fill the mask
        
        Returns
        -------
        img: numpy array
                Generated mask.
        """

        s = rle_string.split()
        starts, lengths = [np.asarray(x, dtype = int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(width * height, dtype = np.uint8)
        
        for lo, hi in zip(starts, ends):
            img[lo:hi] = fill_value

        return img.reshape((width, height)).T
    
###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file contains the code that is needed to create a dataset for a PyTorch#
# Deep Learning model.                                                        #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-09-06                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import os
import cv2
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
import torch
from torch import nn
from torch.utils.data.dataset import Dataset

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD = [0.229, 0.224, 0.225]

def image_to_input(image, rbg_mean, rbg_std):
    input = image.astype(np.float32)
    input = input[..., ::-1] / 255
    input = input.transpose(0, 3, 1, 2)
    input[:, 0] = (input[:, 0] - rbg_mean[0]) / rbg_std[0]
    input[:, 1] = (input[:, 1] - rbg_mean[1]) / rbg_std[1]
    input[:, 2] = (input[:, 2] - rbg_mean[2]) / rbg_std[2]
    
    return input

class SteelSegmentationDataset(Dataset):
    """
    This class creates a dataset that loads steel images and RLE labels into a
    PyTorch Deep Learning model.
    """
    
    def __init__(self, labels_df, data_folder, mean, std, phase, augment = "default"):
        """
        This is the class' constructor.

        Parameters
        ----------
        labels_df: Pandas DataFrame
                Labels assoociated with corresponding ImageId.

        data_folder: string
                Path of the folder where images are stored.

        mean: tuple
                Mean value of the images for each component (R, G, B).
        
        std: tuple
                Standard deviation of the images for each component (R, G, B).

        phase: string
                Either "train" or "val". Indicates if the dataset is the training set
                or the validation set.

        augment: string (default = "default")
                Type of data augmentation to use for training set.
                Either "default" for default data augmentation or
                "256x400crop_da" for data augmentation with crop to 256 x 400 px.

        Returns
        -------
        None
        """

        self.labels_df = labels_df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.augment = augment
        self.transforms = self._get_transforms(phase, mean, std)
        self.fnames = self.labels_df["ImageId"].unique().tolist()
                
        labels_df["Label"] = (~labels_df["EncodedPixels"].isnull()).astype(np.int32)
        self.labels = labels_df["Label"].values
        self.labels = self.labels.reshape(-1, 4)
        self.labels = np.hstack([self.labels.sum(1, keepdims = True) == 0, self.labels]).T
        labels_df.drop("Label", axis = 1, inplace = True)

        # Group masks by ID
        labels_df = labels_df.groupby("ImageId")["EncodedPixels"].agg(lambda x: x.tolist())
        
        # Create dict with ImageId as key and masks as value
        self.labels_dict = labels_df.to_dict()

    def do_random_crop(self, image, mask, w, h):
        height, width = image.shape[:2]
        x = 0
        y = 0

        if width > w:
            x = np.random.choice(width - w)

        if height > h:
            y = np.random.choice(height - h)

        image = image[y:y + h, x:x + w]
        mask = mask[y:y + h, x:x + w]

        return image, mask

    def do_random_cutout(self, image, mask):
        height, width = image.shape[:2]

        u0 = [0, 1][np.random.choice(2)]
        u1 = np.random.choice(width)

        if u0 == 0:
            x0 = 0
            x1 = u1

        if u0 == 1:
            x0 = u1
            x1 = width

        image[:, x0:x1] = 0
        mask [:, x0:x1] = 0

        return image, mask

    def do_random_crop_rescale(self, image, mask, w, h):
        height, width = image.shape[:2]
        x, y = 0, 0
        if width > w:
            x = np.random.choice(width - w)
        
        if height > h:
            y = np.random.choice(height - h)

        image = image[y:y + h, x:x + w]
        if self.augment == "256x400crop_da": # For EfficientNetB5 model
            mask  = mask [y:y + h, x:x + w]

            if (w, h) != (width, height):
                image = cv2.resize(image, dsize = (width, height), interpolation = cv2.INTER_LINEAR)
                mask = cv2.resize(mask, dsize = (width, height), interpolation = cv2.INTER_NEAREST)
        else:
            mask = mask [:, y:y + h, x:x + w]

            if (w, h) != (width, height):
                image = cv2.resize(image, dsize = (width, height), interpolation = cv2.INTER_LINEAR)
                mask = mask.transpose(1, 2, 0)
                mask = cv2.resize(mask, dsize = (width, height), interpolation = cv2.INTER_NEAREST)
                mask = mask.transpose(2, 0, 1)

        return image, mask

    def do_random_crop_rotate_rescale(self, image, mask, w, h):
        H, W = image.shape[:2]

        #dangle = np.random.uniform(-2.5, 2.5)
        #dscale = np.random.uniform(-0.10, 0.10, 2)
        dangle = np.random.uniform(-8, 8)
        dshift = np.random.uniform(-0.1, 0.1, 2)

        dscale_x = np.random.uniform(-0.00075, 0.00075)
        dscale_y = np.random.uniform(-0.25, 0.25)

        cos = np.cos(dangle / 180 * np.pi)
        sin = np.sin(dangle / 180 * np.pi)
        sx, sy = 1 + dscale_x, 1 + dscale_y #1,1 #
        tx, ty = dshift * np.min([H, W])

        src = np.array([[-w / 2, -h / 2],[w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32)
        src = src * [sx, sy]
        x = (src * [cos, -sin]).sum(1) + W / 2
        y = (src * [sin, cos]).sum(1) + H / 2
               
        src = np.column_stack([x, y])
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        s = src.astype(np.float32)
        d = dst.astype(np.float32)
        transform = cv2.getPerspectiveTransform(s, d)

        image = cv2.warpPerspective(image, transform, (W, H), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = (0,0,0))

        if self.augment == "256x400crop_da": # For EfficientNetB5 model
            mask = cv2.warpPerspective(mask, transform, (W, H), flags = cv2.INTER_NEAREST, borderMode = cv2.BORDER_CONSTANT, borderValue = (0))
        else:
            mask = mask.transpose(1, 2, 0)
            mask = cv2.warpPerspective(mask, transform, (W, H), flags = cv2.INTER_NEAREST, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0, 0))
            mask = mask.transpose(2, 0, 1)
        
        return image, mask
    
    def do_random_log_contast(self, image, gain = [0.70, 1.30]):
        gain = np.random.uniform(gain[0], gain[1], 1)
        inverse = np.random.choice(2, 1)

        image = image.astype(np.float32) / 255 

        if inverse == 0:
            image = gain * np.log(image + 1)
        else:
            image = gain * (2 ** image - 1)

        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return image

    def do_flip_lr(self, image, mask):
        image = cv2.flip(image, 1)

        if self.augment == "256x400crop_da": # For EfficientNetB5 model
            mask = mask[:, ::-1]
        else:
            mask = mask[:, :, ::-1]

        return image, mask

    def do_flip_ud(self, image, mask):
        image = cv2.flip(image, 0)

        if self.augment == "256x400crop_da": # For EfficientNetB5 model
            mask = mask[::-1, :]
        else:
            mask = mask[:, ::-1, :]

        return image, mask

    def do_random_noise(self, image, noise = 8):
        H, W = image.shape[:2]
        image = image.astype(np.float32)
        image = image + np.random.uniform(-1, 1, (H, W, 1)) * noise
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image
    
    def do_noise(self, image, mask, noise = 8):
        H, W = image.shape[:2]
        image = image + np.random.uniform(-1, 1, (H, W, 1)) * noise
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image, mask
    
    def train_augment(self, image, mask):
        u = np.random.choice(3)

        if u == 0:
            pass
        elif u == 1:
            image, mask = self.do_random_crop_rescale(image, mask, 1600 - (256 - 224), 224)
        elif u == 2:
            image, mask = self.do_random_crop_rotate_rescale(image, mask, 1600 - (256 - 224), 224)

        if np.random.rand() > 0.5:
            image = self.do_random_log_contast(image)

        if np.random.rand() > 0.5:
            image, mask = self.do_flip_lr(image, mask)

        if np.random.rand() > 0.5:
            image, mask = self.do_flip_ud(image, mask)

        if np.random.rand() > 0.5:
            image, mask = self.do_noise(image, mask)
            
        return image, mask

    def train_augment_256x400crop_da(self, image, mask):
        u = np.random.choice(3)

        if u == 0:
            pass
        elif u == 1:
            image, mask = self.do_random_crop_rescale(image, mask, 1600 - (256 - 180), 180)
        elif u == 2:
            image, mask = self.do_random_crop_rotate_rescale(image, mask, 1600 - (256 - 200), 200)

        image, mask = self.do_random_crop(image, mask, 400, 256)

        if np.random.rand() > 0.25:
             image, mask = self.do_random_cutout(image, mask)

        if np.random.rand() > 0.5:
            image, mask = self.do_flip_lr(image, mask)

        if np.random.rand()>0.5:
            image, mask = self.do_flip_ud(image, mask)

        if np.random.rand() > 0.5:
            image = self.do_random_log_contast(image, gain = [0.50, 1.75])

        u = np.random.choice(2)

        if u == 0:
            pass
        if u == 1:
            image = self.do_random_noise(image, noise = 8)

        return image, mask
        
    def _get_transforms(self, phase, mean, std):
        """
        This method contains the data augmentation step.

        Parameters
        ----------
        phase: string
                Either "train" or "val". Indicates if the dataset is the training set
                or the validation set.

        mean: tuple
                Mean value of the images for each component (R, G, B).
        
        std: tuple
                Standard deviation of the images for each component (R, G, B).

        Returns
        -------
        list_trfms: albumentation Object
                Albumentation object defining the data augmentation.
        """

        list_transforms = []

        """
        if phase == "train":
            list_transforms.extend(
                [
                    HorizontalFlip(), # only horizontal flip as of now
                ]
            )

        """
            
        list_transforms.extend(
            [
                Normalize(mean = mean, std = std, p = 1),
                #ToTensor(num_classes = 4)
            ]
        )
        list_trfms = Compose(list_transforms)
        
        return list_trfms

    def __getitem__(self, idx):
        """
        This method returns the number of items contained in the dataset.

        Parameters
        ----------
        idx: integer
                Index of the image we want to get from the dataset.

        Returns
        -------
        img: MxNet ndarray
                Image from the dataset.

        mask: numpy array
                Corresponding mask to the image above.
        """
        
        image_id = self.fnames[idx]
        image_path = os.path.join(self.root, image_id)
        mask_rle_str_lst = self.labels_dict[image_id]
        
        # Read image from disk
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert mask from RLE to array
        masks_npa = np.zeros((256, 1600, 4), dtype = np.float32)

        if self.augment == "default":
            for idx, mask_rle_str in enumerate(mask_rle_str_lst):
                if type(mask_rle_str) == str:
                    masks_npa[:, :, idx] = RLEFormatConverter.rle2mask(mask_rle_str, 1600, 256)
                           
            masks_npa = masks_npa.transpose(2, 0, 1)
        elif self.augment == "256x400crop_da": # For EfficientNetB5 model
            labels_lst = []

            for idx, mask_rle_str in enumerate(mask_rle_str_lst):
                if type(mask_rle_str) == str:
                    masks_npa[:, :, idx] = RLEFormatConverter.rle2mask(mask_rle_str, 1600, 256, fill_value = idx + 1)

                if mask_rle_str == "":
                    labels_lst.append(0)
                else:
                    labels_lst.append(1)

            masks_npa = masks_npa.transpose(2, 0, 1)
            masks_npa = masks_npa.max(0, keepdims = 0)
            
        if self.phase == "train" and self.augment == "default":
            img, masks_npa = self.train_augment(img, masks_npa)
        elif self.phase == "train" and self.augment == "256x400crop_da":
            img, masks_npa = self.train_augment_256x400crop_da(img, masks_npa)
            
        """
        if self.transforms is not None:
            augmented = self.transforms(image = img, mask = masks_npa)
            img = augmented["image"]
            masks_npa = augmented["mask"]
        """
        
        if self.augment == "256x400crop_da":
            return img, labels_lst, masks_npa
        else:
            return img, masks_npa
    
    def __len__(self):
        """
        This method returns the number of items contained in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        : int
                Number of items contained in the dataset.
        """

        return len(self.fnames)
    
###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file contains the code that is needed to create a dataset for a Gluon  #
# Deep Learning model.                                                        #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-09-06                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import mxnet as mx
import os
import cv2
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)

class SteelClassificationDataset(mx.gluon.data.dataset.Dataset):
    """
    This class creates a dataset that loads steel images and RLE labels into a
    Gluon Deep Learning model.
    """

    def __init__(self, labels_df, data_folder, mean, std, phase):
        """
        This is the class' constructor.

        Parameters
        ----------
        labels_df: Pandas DataFrame
                Labels assoociated with corresponding ImageId.

        data_folder: string
                Path of the folder where images are stored.

        mean: tuple
                Mean value of the images for each component (R, G, B).
        
        std: tuple
                Standard deviation of the images for each component (R, G, B).

        phase: string
                Either "train" or "val". Indicates if the dataset is the training set
                or the validation set.

        Returns
        -------
        None
        """

        self.labels_df = labels_df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = self._get_transforms(phase, mean, std)
        self.fnames = self.labels_df["ImageId"].unique().tolist()
                
        # Group masks by ID
        labels_df = labels_df.groupby("ImageId")["EncodedPixels"].agg(lambda x: x.tolist())
        
        # Create dict with ImageId as key and masks as value
        self.labels_dict = labels_df.to_dict()
                
    def _get_transforms(self, phase, mean, std):
        """
        This method contains the data augmentation step.

        Parameters
        ----------
        phase: string
                Either "train" or "val". Indicates if the dataset is the training set
                or the validation set.

        mean: tuple
                Mean value of the images for each component (R, G, B).
        
        std: tuple
                Standard deviation of the images for each component (R, G, B).

        Returns
        -------
        list_trfms: albumentation Object
                Albumentation object defining the data augmentation.
        """

        list_transforms = []

        """
        if phase == "train":
            list_transforms.extend(
                [
                    HorizontalFlip(), # only horizontal flip as of now
                ]
            )

        """
            
        list_transforms.extend(
            [
                Normalize(mean = mean, std = std, p = 1)
            ]
        )
        list_trfms = Compose(list_transforms)
        
        return list_trfms

    def __getitem__(self, idx):
        """
        This method returns the number of items contained in the dataset.

        Parameters
        ----------
        idx: integer
                Index of the image we want to get from the dataset.

        Returns
        -------
        img: MxNet ndarray
                Image from the dataset.

        mask: numpy array
                Corresponding mask to the image above.
        """
        
        image_id = self.fnames[idx]
        image_path = os.path.join(self.root, image_id)
        mask_rle_str_lst = self.labels_dict[image_id]
        
        # Read image from disk
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert mask from RLE to array
        masks_npa = np.zeros((256, 1600, 4), dtype = np.float32)
        labels_lst = []

        for idx, mask_rle_str in enumerate(mask_rle_str_lst):
            if type(mask_rle_str) == str:
                masks_npa[:, :, idx] = RLEFormatConverter.rle2mask(mask_rle_str, 1600, 256)
                labels_lst.append(1)
            else:
                labels_lst.append(0)

        labels_npa = np.array(labels_lst, dtype = np.float32)
                        
        if self.transforms is not None:
            augmented = self.transforms(image = img, mask = masks_npa)
            img = augmented["image"]
            masks_npa = augmented["mask"] # 256x1600x4
            masks_npa = masks_npa.transpose(2, 0, 1) # 4x256x1600
                    
        # to tensor
        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)

        return img, masks_npa, labels_npa
    
    def __len__(self):
        """
        This method returns the number of items contained in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        : int
                Number of items contained in the dataset.
        """

        return len(self.fnames)
    
###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file contains the code that is needed to create a ResNet model using   #
# PyTorch and pretrained weights from ImageNet.                               #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-10-06                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import os
import time
import warnings
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel.data_parallel import data_parallel
import copy
from timeit import default_timer as timer
from torch.utils.data.sampler import Sampler, SequentialSampler

class NullScheduler(object):
    def __init__(self, lr = 0.01):
        super(NullScheduler, self).__init__()

        self.lr = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = "NullScheduler: lr = %0.5f " % (self.lr)
        return string

class ConvBn2d(nn.Module):
    """
    This class a ConvBn2d module for a ResNet34 model.
    """

    def __init__(self, in_channel, out_channel, kernel_size = 3, padding = 1, stride = 1):
        """
        This is the class' constructor.

        Parameters
        ----------
        in_channel: int
                Number of input channels of this module.

        out_channel: int
                Number of output channels of this module.

        kernel_size: int (default = 1)
                Kernel size for the Conv2d layer of the module.

        padding: int (default = 1)
                Padding size for the Conv2d layer of the module.

        stride: int (default = 1)
                Stride size for the Conv2d layer of the module.
        
        Returns
        -------
        None
        """

        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, padding = padding, stride = stride, bias = False)
        self.bn = nn.BatchNorm2d(out_channel, eps = 1e-5)

    def forward(self, x):
        """
        This method defines the behavior of the module when data goes through it.

        Parameters
        ----------
        x: Pytorch Tensor
                Input tensor of the layer.
        
        Returns
        -------
        x: Pytorch Tensor
                Output tensor of the layer.
        """

        x = self.conv(x)
        x = self.bn(x)

        return x
    
class BasicBlock(nn.Module):
    """
    This class a BasicBlock module for a ResNet34 model (bottleneck type C).
    """

    def __init__(self, in_channel, channel, out_channel, stride = 1, is_shortcut = False):
        """
        This is the class' constructor.

        Parameters
        ----------
        in_channel: int
                Number of input channels of this module.

        channel:


        out_channel: int
                Number of output channels of this module.

        stride: int (default = 1)
                Stride size for the block.

        is_shortcut: bool (default = False)
                Whether the block has a shortcut connection or not.
        
        Returns
        -------
        None
        """

        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel, channel, kernel_size = 3, padding = 1, stride = stride)
        self.conv_bn2 = ConvBn2d(channel, out_channel, kernel_size = 3, padding = 1, stride = 1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size = 1, padding = 0, stride = stride)

    def forward(self, x):
        """
        This method defines the behavior of the module when data goes through it.

        Parameters
        ----------
        x: Pytorch Tensor
                Input tensor of the layer.
        
        Returns
        -------
        z: Pytorch Tensor
                Output tensor of the layer.
        """

        z = F.relu(self.conv_bn1(x),inplace = True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z, inplace = True)

        return z
    
class ResNet34(nn.Module):
    """
    This class creates a ResNet34 model.
    """

    def __init__(self, num_class = 1000):
        """
        This is the class' constructor.

        Parameters
        ----------
        num_class: int (default = 1000)
                Number of classes for classification problem.
        
        Returns
        -------
        None
        """

        super(ResNet34, self).__init__()
        
        self.block0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, padding = 3, stride = 2, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )

        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 2),
            BasicBlock(64, 64, 64, stride = 1, is_shortcut = False),
            *[BasicBlock(64, 64, 64, stride = 1, is_shortcut = False) for i in range(1, 3)]
        )

        self.block2 = nn.Sequential(
            BasicBlock(64, 128, 128, stride = 2, is_shortcut = True),
            *[BasicBlock(128, 128, 128, stride = 1, is_shortcut = False) for i in range(1, 4)]
        )

        self.block3 = nn.Sequential(
            BasicBlock(128, 256, 256, stride = 2, is_shortcut = True),
            *[BasicBlock(256, 256, 256, stride = 1, is_shortcut = False) for i in range(1, 6)]
        )

        self.block4 = nn.Sequential(
            BasicBlock(256, 512, 512, stride = 2, is_shortcut = True),
            *[BasicBlock(512, 512, 512, stride = 1, is_shortcut = False) for i in range(1, 3)]
        )

        self.logit = nn.Linear(512, num_class)

    def forward(self, x):
        """
        This method defines the behavior of the model when data goes through it.

        Parameters
        ----------
        x: Pytorch Tensor
                Input tensor of the layer.
        
        Returns
        -------
        x: Pytorch Tensor
                Output tensor of the layer.
        """

        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        logit = self.logit(x)

        return logit

class Decode(nn.Module):
    """
    This class define the decoder part of a Unet model.
    """

    def __init__(self, in_channel, out_channel):
        """
        This is the class' constructor.

        Parameters
        ----------
        in_channel: int
                Number of input channels of this module.

        out_channel: int
                Number of output channels of this module.
        
        Returns
        -------
        None
        """

        super(Decode, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace = True),

            nn.Conv2d(out_channel // 2, out_channel, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )
        
    def forward(self, x):
        """
        This method defines the behavior of the layer when data goes through it.

        Parameters
        ----------
        x: Pytorch Tensor
                Input tensor of the layer.
        
        Returns
        -------
        x: Pytorch Tensor
                Output tensor of the layer.
        """

        x = self.top(torch.cat(x, 1))

        return x
    
class ResUnet34(nn.Module):
    """
    This class creates a ResNet34-based Unet model.
    """

    def __init__(self, num_class = 5):
        """
        This is the class' constructor.

        Parameters
        ----------
        num_class: int (default = 5)
                Number of classes for classification problem,
                increased of one for negative class.
        
        Returns
        -------
        None
        """

        super(ResUnet34, self).__init__()
        
        e = ResNet34()
        self.block0 = e.block0  # 64, 128, 128
        self.block1 = e.block1  # 64,  64,  64
        self.block2 = e.block2  #128,  32,  32
        self.block3 = e.block3  #256,  16,  16
        self.block4 = e.block4  #512,   8,   8
        e = None # Dropped

        self.decode1 = Decode(512, 256)       #  8,   8
        self.decode2 = Decode(256 + 256, 256) # 16,  16
        self.decode3 = Decode(256 + 128, 128)
        self.decode4 = Decode(128 + 64, 64)
        self.decode5 = Decode(64 + 64, 64)
        self.decode6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.logit = nn.Conv2d(64, num_class, kernel_size = 1)

    def load_pretrain(self, pretrain_file, skip = []):
        """
        This method loads the weights from a pretrained ResNet34.

        Parameters
        ----------
        pretrain_file: string
                Path where the weights of the pretrained model are stored.

        skip: list (default = [])
                Layers to skip when loading weights.
                                        
        Returns
        -------
        None
        """

        conversion = [
            "block0.0.weight", (64, 3, 7, 7), "conv1.weight", (64, 3, 7, 7),
            "block0.1.weight", (64,), "bn1.weight", (64,),
            "block0.1.bias", (64,), "bn1.bias", (64,),
            "block0.1.running_mean", (64,), "bn1.running_mean", (64,),
            "block0.1.running_var", (64,), "bn1.running_var", (64,),
            "block1.1.conv_bn1.conv.weight", (64, 64, 3, 3), "layer1.0.conv1.weight", (64, 64, 3, 3),
            "block1.1.conv_bn1.bn.weight", (64,), "layer1.0.bn1.weight", (64,),
            "block1.1.conv_bn1.bn.bias", (64,), "layer1.0.bn1.bias", (64,),
            "block1.1.conv_bn1.bn.running_mean", (64,), "layer1.0.bn1.running_mean", (64,),
            "block1.1.conv_bn1.bn.running_var", (64,), "layer1.0.bn1.running_var", (64,),
            "block1.1.conv_bn2.conv.weight", (64, 64, 3, 3), "layer1.0.conv2.weight", (64, 64, 3, 3),
            "block1.1.conv_bn2.bn.weight", (64,), "layer1.0.bn2.weight", (64,),
            "block1.1.conv_bn2.bn.bias", (64,), "layer1.0.bn2.bias", (64,),
            "block1.1.conv_bn2.bn.running_mean", (64,), "layer1.0.bn2.running_mean", (64,),
            "block1.1.conv_bn2.bn.running_var", (64,), "layer1.0.bn2.running_var", (64,),
            "block1.2.conv_bn1.conv.weight", (64, 64, 3, 3), "layer1.1.conv1.weight", (64, 64, 3, 3),
            "block1.2.conv_bn1.bn.weight", (64,), "layer1.1.bn1.weight", (64,),
            "block1.2.conv_bn1.bn.bias", (64,), "layer1.1.bn1.bias", (64,),
            "block1.2.conv_bn1.bn.running_mean", (64,), "layer1.1.bn1.running_mean", (64,),
            "block1.2.conv_bn1.bn.running_var", (64,), "layer1.1.bn1.running_var", (64,),
            "block1.2.conv_bn2.conv.weight", (64, 64, 3, 3), "layer1.1.conv2.weight", (64, 64, 3, 3),
            "block1.2.conv_bn2.bn.weight", (64,), "layer1.1.bn2.weight", (64,),
            "block1.2.conv_bn2.bn.bias", (64,), "layer1.1.bn2.bias", (64,),
            "block1.2.conv_bn2.bn.running_mean", (64,), "layer1.1.bn2.running_mean", (64,),
            "block1.2.conv_bn2.bn.running_var", (64,), "layer1.1.bn2.running_var", (64,),
            "block1.3.conv_bn1.conv.weight", (64, 64, 3, 3), "layer1.2.conv1.weight", (64, 64, 3, 3),
            "block1.3.conv_bn1.bn.weight", (64,), "layer1.2.bn1.weight", (64,),
            "block1.3.conv_bn1.bn.bias", (64,), "layer1.2.bn1.bias", (64,),
            "block1.3.conv_bn1.bn.running_mean", (64,), "layer1.2.bn1.running_mean", (64,),
            "block1.3.conv_bn1.bn.running_var", (64,), "layer1.2.bn1.running_var", (64,),
            "block1.3.conv_bn2.conv.weight", (64, 64, 3, 3), "layer1.2.conv2.weight", (64, 64, 3, 3),
            "block1.3.conv_bn2.bn.weight", (64,), "layer1.2.bn2.weight", (64,),
            "block1.3.conv_bn2.bn.bias", (64,), "layer1.2.bn2.bias", (64,),
            "block1.3.conv_bn2.bn.running_mean", (64,), "layer1.2.bn2.running_mean", (64,),
            "block1.3.conv_bn2.bn.running_var", (64,), "layer1.2.bn2.running_var", (64,),
            "block2.0.conv_bn1.conv.weight", (128, 64, 3, 3), "layer2.0.conv1.weight", (128, 64, 3, 3),
            "block2.0.conv_bn1.bn.weight", (128,), "layer2.0.bn1.weight", (128,),
            "block2.0.conv_bn1.bn.bias", (128,), "layer2.0.bn1.bias", (128,),
            "block2.0.conv_bn1.bn.running_mean", (128,), "layer2.0.bn1.running_mean", (128,),
            "block2.0.conv_bn1.bn.running_var", (128,), "layer2.0.bn1.running_var", (128,),
            "block2.0.conv_bn2.conv.weight", (128, 128, 3, 3), "layer2.0.conv2.weight", (128, 128, 3, 3),
            "block2.0.conv_bn2.bn.weight", (128,), "layer2.0.bn2.weight", (128,),
            "block2.0.conv_bn2.bn.bias", (128,), "layer2.0.bn2.bias", (128,),
            "block2.0.conv_bn2.bn.running_mean", (128,), "layer2.0.bn2.running_mean", (128,),
            "block2.0.conv_bn2.bn.running_var", (128,), "layer2.0.bn2.running_var", (128,),
            "block2.0.shortcut.conv.weight", (128, 64, 1, 1), "layer2.0.downsample.0.weight", (128, 64, 1, 1),
            "block2.0.shortcut.bn.weight", (128,), "layer2.0.downsample.1.weight", (128,),
            "block2.0.shortcut.bn.bias", (128,), "layer2.0.downsample.1.bias", (128,),
            "block2.0.shortcut.bn.running_mean", (128,), "layer2.0.downsample.1.running_mean", (128,),
            "block2.0.shortcut.bn.running_var", (128,), "layer2.0.downsample.1.running_var", (128,),
            "block2.1.conv_bn1.conv.weight", (128, 128, 3, 3), "layer2.1.conv1.weight", (128, 128, 3, 3),
            "block2.1.conv_bn1.bn.weight", (128,), "layer2.1.bn1.weight", (128,),
            "block2.1.conv_bn1.bn.bias", (128,), "layer2.1.bn1.bias", (128,),
            "block2.1.conv_bn1.bn.running_mean", (128,), "layer2.1.bn1.running_mean", (128,),
            "block2.1.conv_bn1.bn.running_var", (128,), "layer2.1.bn1.running_var", (128,),
            "block2.1.conv_bn2.conv.weight", (128, 128, 3, 3), "layer2.1.conv2.weight", (128, 128, 3, 3),
            "block2.1.conv_bn2.bn.weight", (128,), "layer2.1.bn2.weight", (128,),
            "block2.1.conv_bn2.bn.bias", (128,), "layer2.1.bn2.bias", (128,),
            "block2.1.conv_bn2.bn.running_mean", (128,), "layer2.1.bn2.running_mean", (128,),
            "block2.1.conv_bn2.bn.running_var", (128,), "layer2.1.bn2.running_var", (128,),
            "block2.2.conv_bn1.conv.weight", (128, 128, 3, 3), "layer2.2.conv1.weight", (128, 128, 3, 3),
            "block2.2.conv_bn1.bn.weight", (128,), "layer2.2.bn1.weight", (128,),
            "block2.2.conv_bn1.bn.bias", (128,), "layer2.2.bn1.bias", (128,),
            "block2.2.conv_bn1.bn.running_mean", (128,), "layer2.2.bn1.running_mean", (128,),
            "block2.2.conv_bn1.bn.running_var", (128,), "layer2.2.bn1.running_var", (128,),
            "block2.2.conv_bn2.conv.weight", (128, 128, 3, 3), "layer2.2.conv2.weight", (128, 128, 3, 3),
            "block2.2.conv_bn2.bn.weight", (128,), "layer2.2.bn2.weight", (128,),
            "block2.2.conv_bn2.bn.bias", (128,), "layer2.2.bn2.bias", (128,),
            "block2.2.conv_bn2.bn.running_mean", (128,), "layer2.2.bn2.running_mean", (128,),
            "block2.2.conv_bn2.bn.running_var", (128,), "layer2.2.bn2.running_var", (128,),
            "block2.3.conv_bn1.conv.weight", (128, 128, 3, 3), "layer2.3.conv1.weight", (128, 128, 3, 3),
            "block2.3.conv_bn1.bn.weight", (128,), "layer2.3.bn1.weight", (128,),
            "block2.3.conv_bn1.bn.bias", (128,), "layer2.3.bn1.bias", (128,),
            "block2.3.conv_bn1.bn.running_mean", (128,), "layer2.3.bn1.running_mean", (128,),
            "block2.3.conv_bn1.bn.running_var", (128,), "layer2.3.bn1.running_var", (128,),
            "block2.3.conv_bn2.conv.weight", (128, 128, 3, 3), "layer2.3.conv2.weight", (128, 128, 3, 3),
            "block2.3.conv_bn2.bn.weight", (128,), "layer2.3.bn2.weight", (128,),
            "block2.3.conv_bn2.bn.bias", (128,), "layer2.3.bn2.bias", (128,),
            "block2.3.conv_bn2.bn.running_mean", (128,), "layer2.3.bn2.running_mean", (128,),
            "block2.3.conv_bn2.bn.running_var", (128,), "layer2.3.bn2.running_var", (128,),
            "block3.0.conv_bn1.conv.weight", (256, 128, 3, 3), "layer3.0.conv1.weight", (256, 128, 3, 3),
            "block3.0.conv_bn1.bn.weight", (256,), "layer3.0.bn1.weight", (256,),
            "block3.0.conv_bn1.bn.bias", (256,), "layer3.0.bn1.bias", (256,),
            "block3.0.conv_bn1.bn.running_mean", (256,), "layer3.0.bn1.running_mean", (256,),
            "block3.0.conv_bn1.bn.running_var", (256,), "layer3.0.bn1.running_var", (256,),
            "block3.0.conv_bn2.conv.weight", (256, 256, 3, 3), "layer3.0.conv2.weight", (256, 256, 3, 3),
            "block3.0.conv_bn2.bn.weight", (256,), "layer3.0.bn2.weight", (256,),
            "block3.0.conv_bn2.bn.bias", (256,), "layer3.0.bn2.bias", (256,),
            "block3.0.conv_bn2.bn.running_mean", (256,), "layer3.0.bn2.running_mean", (256,),
            "block3.0.conv_bn2.bn.running_var", (256,), "layer3.0.bn2.running_var", (256,),
            "block3.0.shortcut.conv.weight", (256, 128, 1, 1), "layer3.0.downsample.0.weight", (256, 128, 1, 1),
            "block3.0.shortcut.bn.weight", (256,), "layer3.0.downsample.1.weight", (256,),
            "block3.0.shortcut.bn.bias", (256,), "layer3.0.downsample.1.bias", (256,),
            "block3.0.shortcut.bn.running_mean", (256,), "layer3.0.downsample.1.running_mean", (256,),
            "block3.0.shortcut.bn.running_var", (256,), "layer3.0.downsample.1.running_var", (256,),
            "block3.1.conv_bn1.conv.weight", (256, 256, 3, 3), "layer3.1.conv1.weight", (256, 256, 3, 3),
            "block3.1.conv_bn1.bn.weight", (256,), "layer3.1.bn1.weight", (256,),
            "block3.1.conv_bn1.bn.bias", (256,), "layer3.1.bn1.bias", (256,),
            "block3.1.conv_bn1.bn.running_mean", (256,), "layer3.1.bn1.running_mean", (256,),
            "block3.1.conv_bn1.bn.running_var", (256,), "layer3.1.bn1.running_var", (256,),
            "block3.1.conv_bn2.conv.weight", (256, 256, 3, 3), "layer3.1.conv2.weight", (256, 256, 3, 3),
            "block3.1.conv_bn2.bn.weight", (256,), "layer3.1.bn2.weight", (256,),
            "block3.1.conv_bn2.bn.bias", (256,), "layer3.1.bn2.bias", (256,),
            "block3.1.conv_bn2.bn.running_mean", (256,), "layer3.1.bn2.running_mean", (256,),
            "block3.1.conv_bn2.bn.running_var", (256,), "layer3.1.bn2.running_var", (256,),
            "block3.2.conv_bn1.conv.weight", (256, 256, 3, 3), "layer3.2.conv1.weight", (256, 256, 3, 3),
            "block3.2.conv_bn1.bn.weight", (256,), "layer3.2.bn1.weight", (256,),
            "block3.2.conv_bn1.bn.bias", (256,), "layer3.2.bn1.bias", (256,),
            "block3.2.conv_bn1.bn.running_mean", (256,), "layer3.2.bn1.running_mean", (256,),
            "block3.2.conv_bn1.bn.running_var", (256,), "layer3.2.bn1.running_var", (256,),
            "block3.2.conv_bn2.conv.weight", (256, 256, 3, 3), "layer3.2.conv2.weight", (256, 256, 3, 3),
            "block3.2.conv_bn2.bn.weight", (256,), "layer3.2.bn2.weight", (256,),
            "block3.2.conv_bn2.bn.bias", (256,), "layer3.2.bn2.bias", (256,),
            "block3.2.conv_bn2.bn.running_mean", (256,), "layer3.2.bn2.running_mean", (256,),
            "block3.2.conv_bn2.bn.running_var", (256,), "layer3.2.bn2.running_var", (256,),
            "block3.3.conv_bn1.conv.weight", (256, 256, 3, 3), "layer3.3.conv1.weight", (256, 256, 3, 3),
            "block3.3.conv_bn1.bn.weight", (256,), "layer3.3.bn1.weight", (256,),
            "block3.3.conv_bn1.bn.bias", (256,), "layer3.3.bn1.bias", (256,),
            "block3.3.conv_bn1.bn.running_mean", (256,), "layer3.3.bn1.running_mean", (256,),
            "block3.3.conv_bn1.bn.running_var", (256,), "layer3.3.bn1.running_var", (256,),
            "block3.3.conv_bn2.conv.weight", (256, 256, 3, 3), "layer3.3.conv2.weight", (256, 256, 3, 3),
            "block3.3.conv_bn2.bn.weight", (256,), "layer3.3.bn2.weight", (256,),
            "block3.3.conv_bn2.bn.bias", (256,), "layer3.3.bn2.bias", (256,),
            "block3.3.conv_bn2.bn.running_mean", (256,), "layer3.3.bn2.running_mean", (256,),
            "block3.3.conv_bn2.bn.running_var", (256,), "layer3.3.bn2.running_var", (256,),
            "block3.4.conv_bn1.conv.weight", (256, 256, 3, 3), "layer3.4.conv1.weight", (256, 256, 3, 3),
            "block3.4.conv_bn1.bn.weight", (256,), "layer3.4.bn1.weight", (256,),
            "block3.4.conv_bn1.bn.bias", (256,), "layer3.4.bn1.bias", (256,),
            "block3.4.conv_bn1.bn.running_mean", (256,), "layer3.4.bn1.running_mean", (256,),
            "block3.4.conv_bn1.bn.running_var", (256,), "layer3.4.bn1.running_var", (256,),
            "block3.4.conv_bn2.conv.weight", (256, 256, 3, 3), "layer3.4.conv2.weight", (256, 256, 3, 3),
            "block3.4.conv_bn2.bn.weight", (256,), "layer3.4.bn2.weight", (256,),
            "block3.4.conv_bn2.bn.bias", (256,), "layer3.4.bn2.bias", (256,),
            "block3.4.conv_bn2.bn.running_mean", (256,), "layer3.4.bn2.running_mean", (256,),
            "block3.4.conv_bn2.bn.running_var", (256,), "layer3.4.bn2.running_var", (256,),
            "block3.5.conv_bn1.conv.weight", (256, 256, 3, 3), "layer3.5.conv1.weight", (256, 256, 3, 3),
            "block3.5.conv_bn1.bn.weight", (256,), "layer3.5.bn1.weight", (256,),
            "block3.5.conv_bn1.bn.bias", (256,), "layer3.5.bn1.bias", (256,),
            "block3.5.conv_bn1.bn.running_mean", (256,), "layer3.5.bn1.running_mean", (256,),
            "block3.5.conv_bn1.bn.running_var", (256,), "layer3.5.bn1.running_var", (256,),
            "block3.5.conv_bn2.conv.weight", (256, 256, 3, 3), "layer3.5.conv2.weight", (256, 256, 3, 3),
            "block3.5.conv_bn2.bn.weight", (256,), "layer3.5.bn2.weight", (256,),
            "block3.5.conv_bn2.bn.bias", (256,), "layer3.5.bn2.bias", (256,),
            "block3.5.conv_bn2.bn.running_mean", (256,), "layer3.5.bn2.running_mean", (256,),
            "block3.5.conv_bn2.bn.running_var", (256,), "layer3.5.bn2.running_var", (256,),
            "block4.0.conv_bn1.conv.weight", (512, 256, 3, 3), "layer4.0.conv1.weight", (512, 256, 3, 3),
            "block4.0.conv_bn1.bn.weight", (512,), "layer4.0.bn1.weight", (512,),
            "block4.0.conv_bn1.bn.bias", (512,), "layer4.0.bn1.bias", (512,),
            "block4.0.conv_bn1.bn.running_mean", (512,), "layer4.0.bn1.running_mean", (512,),
            "block4.0.conv_bn1.bn.running_var", (512,), "layer4.0.bn1.running_var", (512,),
            "block4.0.conv_bn2.conv.weight", (512, 512, 3, 3), "layer4.0.conv2.weight", (512, 512, 3, 3),
            "block4.0.conv_bn2.bn.weight", (512,), "layer4.0.bn2.weight", (512,),
            "block4.0.conv_bn2.bn.bias", (512,), "layer4.0.bn2.bias", (512,),
            "block4.0.conv_bn2.bn.running_mean", (512,), "layer4.0.bn2.running_mean", (512,),
            "block4.0.conv_bn2.bn.running_var", (512,), "layer4.0.bn2.running_var", (512,),
            "block4.0.shortcut.conv.weight", (512, 256, 1, 1), "layer4.0.downsample.0.weight", (512, 256, 1, 1),
            "block4.0.shortcut.bn.weight", (512,), "layer4.0.downsample.1.weight", (512,),
            "block4.0.shortcut.bn.bias", (512,), "layer4.0.downsample.1.bias", (512,),
            "block4.0.shortcut.bn.running_mean", (512,), "layer4.0.downsample.1.running_mean", (512,),
            "block4.0.shortcut.bn.running_var", (512,), "layer4.0.downsample.1.running_var", (512,),
            "block4.1.conv_bn1.conv.weight", (512, 512, 3, 3), "layer4.1.conv1.weight", (512, 512, 3, 3),
            "block4.1.conv_bn1.bn.weight", (512,), "layer4.1.bn1.weight", (512,),
            "block4.1.conv_bn1.bn.bias", (512,), "layer4.1.bn1.bias", (512,),
            "block4.1.conv_bn1.bn.running_mean", (512,), "layer4.1.bn1.running_mean", (512,),
            "block4.1.conv_bn1.bn.running_var", (512,), "layer4.1.bn1.running_var", (512,),
            "block4.1.conv_bn2.conv.weight", (512, 512, 3, 3), "layer4.1.conv2.weight", (512, 512, 3, 3),
            "block4.1.conv_bn2.bn.weight", (512,), "layer4.1.bn2.weight", (512,),
            "block4.1.conv_bn2.bn.bias", (512,), "layer4.1.bn2.bias", (512,),
            "block4.1.conv_bn2.bn.running_mean", (512,), "layer4.1.bn2.running_mean", (512,),
            "block4.1.conv_bn2.bn.running_var", (512,), "layer4.1.bn2.running_var", (512,),
            "block4.2.conv_bn1.conv.weight", (512, 512, 3, 3), "layer4.2.conv1.weight", (512, 512, 3, 3),
            "block4.2.conv_bn1.bn.weight", (512,), "layer4.2.bn1.weight", (512,),
            "block4.2.conv_bn1.bn.bias", (512,), "layer4.2.bn1.bias", (512,),
            "block4.2.conv_bn1.bn.running_mean", (512,), "layer4.2.bn1.running_mean", (512,),
            "block4.2.conv_bn1.bn.running_var", (512,), "layer4.2.bn1.running_var", (512,),
            "block4.2.conv_bn2.conv.weight", (512, 512, 3, 3), "layer4.2.conv2.weight", (512, 512, 3, 3),
            "block4.2.conv_bn2.bn.weight", (512,), "layer4.2.bn2.weight", (512,),
            "block4.2.conv_bn2.bn.bias", (512,), "layer4.2.bn2.bias", (512,),
            "block4.2.conv_bn2.bn.running_mean", (512,), "layer4.2.bn2.running_mean", (512,),
            "block4.2.conv_bn2.bn.running_var", (512,), "layer4.2.bn2.running_var", (512,),
            "logit.weight", (1000, 512), "fc.weight", (1000, 512),
            "logit.bias", (1000,), "fc.bias", (1000,)
        ]

        pretrain_state_dict = torch.load(pretrain_file, map_location = lambda storage, loc: storage)
        state_dict = self.state_dict()

        i = 0
        conversion = np.array(conversion).reshape(-1, 4)
        for key, _, pretrain_key, _ in conversion:
            if any(s in key for s in [".num_batches_tracked", ] + skip):
                continue

            i = i + 1

            state_dict[key] = pretrain_state_dict[pretrain_key]

        self.load_state_dict(state_dict)
        
    def _upsize(self, x, scale_factor = 2):
        """
        This method upscales the tensor given as input by a factor of `scale_factor`.

        Parameters
        ----------
        x: Pytorch Tensor
                Input tensor of the layer.
        
        Returns
        -------
        x: Pytorch Tensor
                Output tensor of the layer.
        """

        x = F.interpolate(x, scale_factor = scale_factor, mode = "nearest")

        return x

    def forward(self, x):
        """
        This method defines the behavior of the model when data goes through it.

        Parameters
        ----------
        x: Pytorch Tensor
                Input tensor of the layer.
        
        Returns
        -------
        x: Pytorch Tensor
                Output tensor of the layer.
        """

        batch_size, C, H, W = x.shape

        backbone = []
        x = self.block0(x)
        backbone.append(x)
        x = self.block1(x)
        backbone.append(x)
        x = self.block2(x)
        backbone.append(x)
        x = self.block3(x)
        backbone.append(x)
        x = self.block4(x)
        backbone.append(x)

        x = self.decode1([backbone[-1], ])
        x = self.decode2([backbone[-2], self._upsize(x)])
        x = self.decode3([backbone[-3], self._upsize(x)])
        x = self.decode4([backbone[-4], self._upsize(x)])
        x = self.decode5([backbone[-5], self._upsize(x)])
        x = self.decode6(self._upsize(x))
        logit = self.logit(x)

        return logit

class PyTorchResUnet34Segmenter(object):
    """
    This class creates a ResNet-based Unet model with training and validation from a pretrained model.
    """

    def __init__(self, model_weights_dir, batch_size, classes = 4, model_name = "ResUnet34", learning_rate = 0.01, l2_reg = 0.0001, num_epochs = 50, augment = [], logger = None):
        """
        This is the class' constructor.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        self.model_weights_dir = model_weights_dir
        self.batch_size = batch_size
        self.classes = classes
        self.model_name = model_name
        self.accumulation_steps = 32 // batch_size
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.num_epochs = num_epochs
        self.augment = augment
        if logger is None:
            self.logger = Logger()
        else:
            self.logger = logger
                
        # Create the neural net
        self.net = ResUnet34().cuda()

        self.schduler = NullScheduler(lr = learning_rate)

        #self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),lr=schduler(0))
        #self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr =0.0005, alpha = 0.95)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr = self.schduler(0), momentum = 0.9, weight_decay = 0.0001)
        
    def time_to_str(self, t, mode = "min"):
        """
        This method formats a time given in argument into
        Hours:minutes or minutes:seconds.

        Parameters
        ----------
        t: time_it.timer
                Time to be converted.

        mode: string, either "min" or "sec" (default = "min")
                Time format to use.
        
        Returns
        -------
        : string
                Formatted time.
        """

        if mode == "min":
            t = int(t) / 60
            hr = t // 60
            min = t % 60
            return "%2d hr %02d min" % (hr, min)

        elif mode == "sec":
            t = int(t)
            min = t // 60
            sec = t % 60
            return "%2d min %02d sec" % (min, sec)

        else:
            raise NotImplementedError

    def metric_hit(self, classifier_output, truth, threshold = 0.5):
        """
        This method computes the number of true positives and
        true negatives for the classification model.

        Parameters
        ----------
        classifier_output: Torch Tensor
                Predictions made by the classifier without the sigmoid
                applied on it.

        truth: Torch Tensor
                Ground truth for samples in `classifier_output`.

        threshold: float (default = 0.5)
                Threshold used to convert predictions probabilities into
                labels.
        
        Returns
        -------
        tn: float
                Ratio of correct true negatives.

        tp: list
                Ratio of correct true positives for each class.

        num_neg: float
                Number of positive labels in truth set.

        num_pos: list
                Number of positive labels in truth set for each class.
        """

        batch_size, num_class, H, W = classifier_output.shape

        with torch.no_grad():
            classifier_output = classifier_output.view(batch_size, num_class, -1)
            truth = truth.view(batch_size, -1)

            probability = torch.softmax(classifier_output, 1)
            p = torch.max(probability, 1)[1]
            t = truth
            correct = (p == t)

            index0 = t == 0
            index1 = t == 1
            index2 = t == 2
            index3 = t == 3
            index4 = t == 4

            num_neg = index0.sum().item()
            num_pos1 = index1.sum().item()
            num_pos2 = index2.sum().item()
            num_pos3 = index3.sum().item()
            num_pos4 = index4.sum().item()

            neg  = correct[index0].sum().item() / (num_neg + 1e-12)
            pos1 = correct[index1].sum().item() / (num_pos1 + 1e-12)
            pos2 = correct[index2].sum().item() / (num_pos2 + 1e-12)
            pos3 = correct[index3].sum().item() / (num_pos3 + 1e-12)
            pos4 = correct[index4].sum().item() / (num_pos4 + 1e-12)

            num_pos = [num_pos1, num_pos2, num_pos3, num_pos4]
            tn = neg
            tp = [pos1, pos2, pos3, pos4]
            
        return tn, tp, num_neg, num_pos

    def one_hot_encode_truth(self, truth, num_class = 4):
        """
        This method one-hot encodes the classification labels
        given in argument.

        Parameters
        ----------
        truth: Torch Tensor
                Ground truth.

        num_class: int (default = 4)
                Number of classes that will be encoded.
        
        Returns
        -------
        one_hot: Torch Tensor
                Ground truth one-hot encoded.
        """

        one_hot = truth.repeat(1, num_class, 1,1)
        arange  = torch.arange(1,num_class+1).view(1,num_class,1,1).to(truth.device)
        one_hot = (one_hot == arange).float()
        return one_hot

    def one_hot_encode_predict(self, predict, num_class = 4):
        """
        This method one-hot encodes the predictions
        given in argument.

        Parameters
        ----------
        predict: Torch Tensor
                Predictions

        num_class: int (default = 4)
                Number of classes that will be encoded.
        
        Returns
        -------
        value: Torch Tensor
                One-hot encoded predictions.
        """

        value, index = torch.max(predict, 1, keepdim = True)
        value = value.repeat(1, num_class, 1, 1)
        index = index.repeat(1, num_class, 1, 1)
        arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(predict.device)

        one_hot = (index == arange).float()
        value = value * one_hot

        return value

    def metric_dice(self, logit, truth, threshold = 0.1, sum_threshold = 1):
        """
        This method computes the Dice score of the predictions.

        Parameters
        ----------
        logit: Torch Tensor
                Predictions made by the model.

        truth: Torch Tensor
                Ground truth.

        threshold: float (default = 0.1)
                Threshold for predictions to be considered as positive.

        sum_threshold: int (default = 1)
        
        Returns
        -------
        dn: float
                Dice score for negative class.

        dp: list
                Dice score for each positive class.

        num_neg: float
                Number of positive labels in truth set.

        num_pos: list
                Number of positive labels in truth set for each class.
        """

        with torch.no_grad():
            probability = torch.softmax(logit, 1)
            probability = self.one_hot_encode_predict(probability)
            truth = self.one_hot_encode_truth(truth)

            batch_size, num_class, H, W = truth.shape
            probability = probability.view(batch_size, num_class, -1)
            truth = truth.view(batch_size, num_class, -1)
            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)

            d_neg = (p_sum < sum_threshold).float()
            d_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-12)

            neg_index = (t_sum == 0).float()
            pos_index = 1 - neg_index

            num_neg = neg_index.sum()
            num_pos = pos_index.sum(0)
            dn = (neg_index * d_neg).sum() / (num_neg + 1e-12)
            dp = (pos_index * d_pos).sum(0) / (num_pos + 1e-12)

            dn = dn.item()
            dp = list(dp.data.cpu().numpy())
            num_neg = num_neg.item()
            num_pos = list(num_pos.data.cpu().numpy())

        return dn, dp, num_neg, num_pos
    
    def criterion(self, logit, truth, weight = None):
        """
        This method computes the model's loss.

        Parameters
        ----------
        logit: Torch Tensor
                Predictions made by the model.

        truth: Torch Tensor
                Ground truth.

        weight: Torch Tensor (default = None)
                Manual rescaling weight given to each class.
        
        Returns
        -------
        loss: Torch Tensor
                Computed loss for the current batch.
        """
        
        logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
        truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)

        if weight is not None: 
            weight = torch.FloatTensor([1] + weight).to(truth.device)
            
        loss = F.cross_entropy(logit, truth, weight = weight, reduction = "none")
        loss = loss.mean()

        return loss
        
    def do_valid(self, val_dataloader):
        """
        This method computes the model's loss.

        Parameters
        ----------
        val_dataloader: Torch DataLoader
                Data that will be used for the validation step.
                        
        Returns
        -------
        valid_loss: numpy array
                Computed loss for the validation set.
        """

        valid_num  = np.zeros(11, np.float32)
        valid_loss = np.zeros(11, np.float32)

        self.logger.write("\n")

        for t, (input, truth_mask) in enumerate(val_dataloader):
            self.net.eval()
            input = input.cuda()
            truth_mask = truth_mask.cuda()
            
            with torch.no_grad():
                logit = data_parallel(self.net, input)
                loss = self.criterion(logit, truth_mask)

                tn, tp, num_neg, num_pos = self.metric_hit(logit, truth_mask)
                dn, dp, num_neg, num_pos = self.metric_dice(logit, truth_mask, threshold = 0.5, sum_threshold = 100)

            batch_size = val_dataloader.batch_size
            l = np.array([loss.item(), tn, *tp, dn, *dp])
            n = np.array([batch_size, num_neg, *num_pos, num_neg, *num_pos])
            valid_loss += l * n
            valid_num += n
            
            print("\r %8d /%8d" % (valid_num[0], len(val_dataloader.dataset)), end = "", flush = True)

        print("\r", end = "", flush = True)
        assert(valid_num[0] == len(val_dataloader.dataset))
        valid_loss = valid_loss / valid_num

        return valid_loss
    
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def get_learning_rate(self, optimizer):
        lr = []
        for param_group in optimizer.param_groups:
           lr += [param_group["lr"]]

        assert(len(lr) == 1) # We support only one param_group
        lr = lr[0]

        return lr
                
    def fit(self, train_dataloader, val_dataloader):
        """
        This method fits the classifier.

        Parameters
        ----------
        train_dataloader: Torch DataLoader
                Data loader used to load training data.
                
        val_dataloader: Torch DataLoader
                Data loader used to load validation data.

        Returns
        -------
        None
        """

        self.logger.write("Fitting " + self.model_name + " model...\n")

        initial_checkpoint = None

        #schduler = NullScheduler(lr = self.lr)
        iter_accum = 1 # 4
        loss_weight = [5, 10, 2, 5] # [5, 5, 2, 5]
        iter_smooth = 50
        iter_log = 500
        iter_valid = 1500
        start_iter = 0

        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),lr = self.lr)
        #optimizer = torch.optim.RMSprop(self.net.parameters(), lr =0.0005, alpha = 0.95)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr = self.lr, momentum = 0.9, weight_decay = self.l2_reg)

        if initial_checkpoint is not None:
            state_dict = torch.load(initial_checkpoint, map_location = lambda storage, loc: storage)
            self.net.load_state_dict(state_dict, strict = False)

            initial_optimizer = initial_checkpoint.replace("_weights.pth", "_optimizer.pth")
            if os.path.exists(initial_optimizer):
                checkpoint = torch.load(initial_optimizer)
                start_iter = checkpoint["iter"]

        else:
            self.net.load_pretrain(skip = ["logit"], pretrain_file = self.model_weights_dir + "resnet34-333f7ec4.pth")

        self.logger.write("optimizer\n  %s\n" % (optimizer))
        self.logger.write("\n")

        self.logger.write("** start training here! **\n")
        self.logger.write("                      |-------------------------------- VALID-----------------------------|---------- TRAIN/BATCH ------------------------------\n")
        self.logger.write("rate     iter   epoch |  loss    hit_neg,pos1,2,3,4           dice_neg,pos1,2,3,4         |  loss    hit_neg,pos1,2,3,4          | time         \n")
        self.logger.write("------------------------------------------------------------------------------------------------------------------------------------------------\n")
                          #0.00000    0.0*   0.0 |  0.690   0.50 [0.00,1.00,0.00,1.00]   0.44 [0.00,0.02,0.00,0.15]  |  0.000   0.00 [0.00,0.00,0.00,0.00]  |  0 hr 00 min

        train_loss = np.zeros(20, np.float32)
        valid_loss = np.zeros(20, np.float32)
        batch_loss = np.zeros(20, np.float32)
        iter = 0
        i = 0
        non_improving_epochs = 0
        best_valid_loss = copy.copy(valid_loss)

        start = timer()
        for epoch in range(self.num_epochs):
            sum_train_loss = np.zeros(20, np.float32)
            sum = np.zeros(20, np.float32)

            optimizer.zero_grad()
            for t, (input, truth_mask) in enumerate(train_dataloader):
                iter = i + start_iter

                # Change learning rate if needed
                if non_improving_epochs > 9:
                    self.logger.write("Learning rate reduced from " + str(self.lr) + " to " + str(self.lr / 2) + "...")
                    self.adjust_learning_rate(optimizer, self.lr / 2)
                    self.lr = self.get_learning_rate(optimizer)
                    non_improving_epochs = 0
            
                self.net.train()
                input = input.cuda()
                truth_mask = truth_mask.cuda()
                
                logit = data_parallel(self.net, input)
                loss = self.criterion(logit, truth_mask, loss_weight)                
                tn, tp, num_neg, num_pos = self.metric_hit(logit, truth_mask)
                
                (loss / iter_accum).backward()
                if (iter % iter_accum) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # print statistics  ------------
                l = np.array([loss.item(), tn, *tp])
                n = np.array([train_dataloader.batch_size, num_neg, *num_pos])

                batch_loss[:6] = l
                sum_train_loss[:6] += l * n
                sum[:6] += n

                if iter % iter_smooth == 0:
                    train_loss = sum_train_loss / (sum + 1e-12)
                    sum_train_loss[...] = 0
                    sum[...] = 0

                print("\r", end = "", flush = True)
                print("%0.5f  %5.1f  %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s" % (\
                             self.lr, iter / 1000, epoch,
                             *valid_loss[:11],
                             *batch_loss[:6],
                             self.time_to_str((timer() - start), "min")), end = "", flush = True)

                i = i + 1

            # Validation for the current epoch
            valid_loss = self.do_valid(val_dataloader)

            self.logger.write("%0.5f  %5.1f  %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s" % (\
                             self.lr, iter / 1000, epoch,
                             *valid_loss[:11],
                             *batch_loss[:6],
                             self.time_to_str((timer() - start), "min")))
            self.logger.write("\n")
            self.logger.write("Saving weights ...")

            # Save weights of the model
            torch.save(self.net.state_dict(), self.model_weights_dir + "resunet34_epoch_%03d_weights.pth" % (epoch))
            torch.save({"iter": iter, "epoch": epoch}, self.model_weights_dir + "resunet34_epoch_%03d_optimizer.pth" % (epoch))
            
            if np.sum(valid_loss[2:6]) > np.sum(best_valid_loss[2:6]):
                best_valid_loss = copy.copy(valid_loss)
                non_improving_epochs = 0
            else:
                non_improving_epochs += 1
                self.logger.write("\n")
            
    def load_trained_model(self, best_weights_path):
        """
        This method makes predictions from the trained model.

        Parameters
        ----------
        best_weights_path: string
                Best weights we want to load.

        Returns
        -------
        None
        """

        self.logger.write("Loading weights from file: " + best_weights_path + "...\n")
        self.net.load_state_dict(torch.load(best_weights_path, map_location = lambda storage, loc: storage), strict = False)

    def _sharpen(self, p, t = 0.5):
        """
        This method makes predictions shapes sharper,
        depending on the `t` argument.

        Parameters
        ----------
        p: MxNet NDArray
                Predictions that need to be sharpened.

        t: float
                Sharpening value.

        Returns
        -------
        : MxNet NDArray
                Sharpened predictions.
        """

        if t != 0:
            return p ** t
        else:
            return p

    def predict(self, test_dataloader):
        """
        This method makes predictions from the trained model.

        Parameters
        ----------
        test_dataloader: Gluon DataLoader
                Data loader used to load testing data.

        Returns
        -------
        None
        """

        test_num = 0
        test_id = []
        test_probability_label = []
        test_probability_mask = []
        test_probability = []
        test_truth_label = []
        test_truth_mask = []

        start = timer()
        for t, (input, truth_mask) in enumerate(test_dataloader):
            batch_size, C, H, W = input.shape
            input = input.cuda()

            with torch.no_grad():
                self.net.eval()

                num_augment = 1
                logit = data_parallel(self.net, input)
                probability = torch.softmax(logit, 1)
                probability_mask = self._sharpen(probability, 0)

                if "flip_lr" in self.augment:
                    logit = data_parallel(self.net, torch.flip(input, dims = [3]))
                    probability  = torch.softmax(torch.flip(logit, dims = [3]), 1)

                    probability_mask += self._sharpen(probability)
                    num_augment += 1

                if "flip_ud" in self.augment:
                    logit = data_parallel(self.net,torch.flip(input, dims = [2]))
                    probability = torch.softmax(torch.flip(logit, dims = [2]), 1)

                    probability_mask += self._sharpen(probability)
                    num_augment += 1

                probability_mask = probability_mask / num_augment
                probability = probability_mask.clone()

                probability_mask = self.one_hot_encode_predict(probability_mask)
                truth_mask = self.one_hot_encode_truth(truth_mask)

            batch_size = probability.shape[0]
            truth_mask = truth_mask.data.cpu().numpy()
            probability_mask = probability_mask.data.cpu().numpy()
            probability = probability.data.cpu().numpy()

            test_probability.append(probability)
            test_probability_mask.append(probability_mask)
            test_truth_mask.append(truth_mask)
            test_num += batch_size
            
            print("\r %4d / %4d  %s" % (test_num, len(test_dataloader.dataset), self.time_to_str((timer() - start), "min")), end = "", flush = True)

        assert(test_num == len(test_dataloader.dataset))
        self.logger.write("")
        
        test_probability = np.concatenate(test_probability)
        test_probability_mask = np.concatenate(test_probability_mask)
        test_truth_mask = np.concatenate(test_truth_mask)

        return test_probability, test_probability_mask, test_truth_mask

    def predict_from_batch(self, input, truth_mask):
        """
        This method makes predictions from the trained model.

        Parameters
        ----------
        input: Torch tensor
                Batch data

        Returns
        -------
        None
        """

        batch_size, C, H, W = input.shape
        input = input.cuda()

        with torch.no_grad():
            self.net.eval()

            num_augment = 1
            logit = data_parallel(self.net, input)
            probability = torch.softmax(logit, 1)
            probability_mask = self._sharpen(probability, 0)

            if "flip_lr" in self.augment:
                logit = data_parallel(self.net, torch.flip(input, dims = [3]))
                probability  = torch.softmax(torch.flip(logit, dims = [3]), 1)

                probability_mask += self._sharpen(probability)
                num_augment += 1

            if "flip_ud" in self.augment:
                logit = data_parallel(self.net,torch.flip(input, dims = [2]))
                probability = torch.softmax(torch.flip(logit, dims = [2]), 1)

                probability_mask += self._sharpen(probability)
                num_augment += 1

            probability_mask = probability_mask / num_augment
            probability = probability_mask.clone()

            probability_mask = self.one_hot_encode_predict(probability_mask)
            truth_mask = self.one_hot_encode_truth(truth_mask)

        batch_size = probability.shape[0]
        truth_mask = truth_mask.data.cpu().numpy()
        probability_mask = probability_mask.data.cpu().numpy()
        probability = probability.data.cpu().numpy()

        return probability, probability_mask, truth_mask

###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file contains the code that is needed to create a ResNet model using   #
# Gluon and pretrained weights from ImageNet.                                 #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-09-06                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import mxnet as mx
import os
import time
import warnings
import random
from sklearn.model_selection import train_test_split

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

from timeit import default_timer as timer

class GluonResNetClassifier(object):
    """
    This class creates a ResNet model with training and validation from a pretrained model.
    """

    def __init__(self, model_weights_dir, batch_size = 4, classes = 4, model_name = "ResNet34_v1", learning_rate = 0.00025, l2_reg = 0.0001, num_epochs = 50, ctx = None, logger = None):
        """
        This is the class' constructor.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        self.model_weights_dir = model_weights_dir
        self.batch_size = batch_size
        self.classes = classes
        self.model_name = model_name
        self.accumulation_steps = 32 // batch_size
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.num_epochs = num_epochs
        if logger is None:
            self.logger = Logger()
        else:
            self.logger = logger

        if ctx is None:
            self.ctx = [mx.gpu(i) for i in mx.test_utils.list_gpus()] if mx.test_utils.list_gpus() else mx.cpu()
        else:
            self.ctx = ctx

        if type(self.ctx) == list and len(self.ctx) > 0:
            self.ctx = self.ctx[0]
        
        self.prefix = model_weights_dir + model_name

        self.net = get_model(self.model_name, pretrained = False)
        with self.net.name_scope():
            self.net.output = gluon.nn.Dense(self.classes)

        self.net.initialize(init.Xavier(), ctx = self.ctx)
        self.net.output.initialize(init.Xavier(), ctx = self.ctx)
        self.net.collect_params().reset_ctx(self.ctx)
        self.net.hybridize()

        self.trainer = gluon.Trainer(self.net.collect_params(), "sgd", {"learning_rate": self.lr, "momentum": 0.9, "wd": self.l2_reg})
        self.criterion = mx.gluon.loss.SigmoidBCELoss(from_sigmoid = False)
                
    def time_to_str(self, t, mode = "min"):
        """
        This methods formats a time given in argument into
        Hours:minutes or minutes:seconds.

        Parameters
        ----------
        t: time_it.timer
                Time to be converted.

        mode: string, either "min" or "sec" (default = "min")
                Time format to use.
        
        Returns
        -------
        : string
                Formatted time.
        """

        if mode == "min":
            t = int(t) / 60
            hr = t // 60
            min = t % 60
            return "%2d hr %02d min" % (hr, min)

        elif mode == "sec":
            t = int(t)
            min = t // 60
            sec = t % 60
            return "%2d min %02d sec" % (min, sec)

        else:
            raise NotImplementedError

    def metric_hit(self, classifier_output, truth, threshold = 0.5):
        """
        This methods computes the number of true positives and
        true negatives for the classification model.

        Parameters
        ----------
        classifier_output: MxNet NDArray
                Predictions made by the classifier without the sigmoid
                applied on it.

        truth: MxNet NDArray
                Ground truth for samples in `classifier_output`.

        threshold: float (default = 0.5)
                Threshold used to convert predictions probabilities into
                labels.
        
        Returns
        -------
        tn: float
                Ratio of correct true negatives.

        tp: list
                Ratio of correct true positives for each class.

        num_neg: float
                Number of positive labels in truth set.

        num_pos: list
                Number of positive labels in truth set for each class.
        """

        batch_size, num_class = classifier_output.shape
        W = 1
        H = 1

        classifier_output = classifier_output.reshape((batch_size, num_class, -1))
        truth = truth.reshape((batch_size, num_class, -1))

        probability = mx.nd.sigmoid(classifier_output)
        p = (probability > threshold)
        t = (truth > 0.5)

        tp = ((p + t) == 2) # True positives
        tn = ((p + t) == 0) # True negatives

        tp = tp.sum(axis = [0, 2])
        tn = tn.sum(axis = [0, 2])
        num_pos = t.sum(axis = [0, 2])
        num_neg = batch_size * H * W - num_pos

        tp = tp.asnumpy()
        tn = tn.asnumpy().sum()
        num_pos = num_pos.asnumpy()
        num_neg = num_neg.asnumpy().sum()

        tp = np.nan_to_num(tp / (num_pos + 1e-12), 0)
        tn = np.nan_to_num(tn / (num_neg + 1e-12), 0)

        tp = list(tp)
        num_pos = list(num_pos)

        return tn, tp, num_neg, num_pos

    def do_valid(self, valid_loader):
        """
        This methods performs the validation step of the training at the end of each epoch.

        Parameters
        ----------                
        valid_loader: Gluon DataLoader
                Data loader used to load validation data.

        Returns
        -------
        valid_loss: float
                Computed loss from validation step.
        """

        valid_num  = np.zeros(6, np.float32)
        valid_loss = np.zeros(6, np.float32)

        self.logger.write("\n")

        for t, (input_tensor, truth_mask, truth_label) in enumerate(valid_loader):
            input_tensor = input_tensor.as_in_context(self.ctx)
            truth_mask = truth_mask.as_in_context(self.ctx)
            truth_label = truth_label.as_in_context(self.ctx)

            logit = self.net(input_tensor)
            loss = self.criterion(logit, truth_label)
            tn, tp, num_neg, num_pos = self.metric_hit(logit, truth_label)

            l = np.array([np.asscalar(loss.mean().asnumpy()), tn, *tp])
            n = np.array([self.batch_size, num_neg, *num_pos])
            valid_loss += l * n
            valid_num += n
        
            print("\r %8d /%8d" % (valid_num[0], len(valid_loader._dataset)), end = "", flush = True)
                        
        print("\r", end = "", flush = True)
        assert(valid_num[0] == len(valid_loader._dataset))
        valid_loss = valid_loss / valid_num

        return valid_loss
                
    def fit(self, train_dataloader, val_dataloader):
        """
        This methods fits the classifier.

        Parameters
        ----------
        train_dataloader: Gluon DataLoader
                Data loader used to load training data.
                
        val_dataloader: Gluon DataLoader
                Data loader used to load validation data.

        Returns
        -------
        None
        """

        self.logger.write("Fitting " + self.model_name + " model...\n")

        iter_smooth = 50
        iter_log = 1000
        iter_accum = 1
        start_iter = 0

        ## start training here! ##############################################
        self.logger.write("** start training here! **\n")
        self.logger.write("   batch_size=%d,  iter_accum=%d\n"%(self.batch_size, iter_accum))
        self.logger.write("                      |--------------- VALID-----------------|---------------------- TRAIN/BATCH ------------------\n")
        self.logger.write("lr       iter   epoch |  loss    tn, [tp1,tp2,tp3,tp4]       |  loss    tn, [tp1,tp2,tp3,tp4]       | time        \n")
        self.logger.write("--------------------------------------------------------------------------------------------------------------------\n")
                          #0.00000   40.5*  26.8 |  0.124   0.98 [0.77,0.40,0.93,0.91]  |  0.000   0.00 [0.00,0.00,0.00,0.00]  |  0 hr 00 min
              
        train_loss = np.zeros(20, np.float32)
        valid_loss = np.zeros(20, np.float32)
        batch_loss = np.zeros(20, np.float32)
        iter = 0
        i = 0
        
        # For gradient accumulation
        for p in self.net.collect_params().values():
            p.grad_req = "add"

        start = timer()
        for epoch in range(self.num_epochs):
            sum_train_loss = np.zeros(20, np.float32)
            sum = np.zeros(20, np.float32)

            for p in self.net.collect_params().values():
                p.zero_grad()

            for t, (input_tensor, truth_mask, truth_label) in enumerate(train_dataloader):
                iter = i + start_iter

                if (iter % iter_log == 0):
                    print("\r", end = "", flush = True)
                    print("%0.5f  %5.1f  %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s" % (\
                             self.lr, iter / 1000, epoch,
                             *valid_loss[:6],
                             *train_loss[:6],
                             self.time_to_str((timer() - start), "min"))
                    )
                    print("\n")
                
                input_tensor = input_tensor.as_in_context(self.ctx)
                truth_label = truth_label.as_in_context(self.ctx)
                truth_mask = truth_mask.as_in_context(self.ctx)

                with mx.autograd.record():
                    logit = self.net(input_tensor)
                    loss = self.criterion(logit, truth_label)
                    mx.autograd.backward(loss / iter_accum)

                tn, tp, num_neg, num_pos = self.metric_hit(logit, truth_label)

                if (iter % iter_accum) == 0:
                    self.trainer.step(self.batch_size * iter_accum, ignore_stale_grad = True)
                    for p in self.net.collect_params().values():
                        p.zero_grad()

                # print statistics  ------------
                l = np.array([np.asscalar(loss.mean().asnumpy()), tn, *tp])
                n = np.array([self.batch_size, num_neg, *num_pos])

                batch_loss[:6] = l
                sum_train_loss[:6] += l * n
                sum[:6] += n

                if iter % iter_smooth == 0:
                    train_loss = sum_train_loss / (sum + 1e-12)
                    sum_train_loss[...] = 0
                    sum[...] = 0
                
                print("\r", end = "", flush = True)
                print("%0.5f  %5.1f  %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s" % (\
                             self.lr, iter / 1000, epoch,
                             *valid_loss[:6],
                             *batch_loss[:6],
                             self.time_to_str((timer() - start), "min")), end = "", flush = True)
                i = i + 1

            valid_loss = self.do_valid(val_dataloader)

            self.logger.write("%0.5f  %5.1f  %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s" % (\
                             self.lr, iter / 1000, epoch,
                             *valid_loss[:6],
                             *batch_loss[:6],
                             self.time_to_str((timer() - start), "min")))

            #self.net.save_parameters("{:s}_best_weights.params".format(self.prefix))
            self.net.save_parameters("{:s}_weights_epoch_{:s}.params".format(self.prefix, str(epoch)))

    def load_trained_model(self, epoch):
        """
        This methods makes predictions from the trained model.

        Parameters
        ----------
        epoch: int
                Epoch we want to load weights from.

        Returns
        -------
        None
        """

        self.logger.write("Loading weights of epoch " + str(epoch) + "from file:" + "{:s}_weights_epoch_{:s}.params".format(self.prefix, str(epoch)) + "...\n")

        self.net.load_parameters("{:s}_weights_epoch_{:s}.params".format(self.prefix, str(epoch)))
        self.net.collect_params().reset_ctx(self.ctx)
        self.net.hybridize()

    def _sharpen(self, p, t = 0.5):
        """
        This methods makes predictions shapes sharper,
        depending on the `t` argument.

        Parameters
        ----------
        p: MxNet NDArray
                Predictions that need to be sharpened.

        t: float
                Sharpening value.

        Returns
        -------
        : MxNet NDArray
                Sharpened predictions.
        """

        if t != 0:
            return p ** t
        else:
            return p

    def predict(self, test_dataloader, test_df, threshold_label):
        """
        This methods makes predictions from the trained model.

        Parameters
        ----------
        test_dataloader: Gluon DataLoader
                Data loader used to load testing data.

        Returns
        -------
        None
        """

        self.logger.write("Making predictions from " + self.model_name + " model...\n")

        test_num = 0
        test_probability_label = []
        test_probability = []
        #test_truth_label = []
        #test_truth_mask = []
        encoded_pixel = []

        start = timer()
    
        for t, (input_tensor, truth_mask, truth_label) in enumerate(test_dataloader):
            batch_size, C, H, W = input_tensor.shape
            input_tensor = input_tensor.as_in_context(self.ctx)
            num_augment = 1
            logit = self.net(input_tensor)
            probability = mx.nd.sigmoid(logit)
        
            probability_label = self._sharpen(probability, 0)

            """
            if "flip_lr" in augment:
                logit = self.net(mx.nd.flip(input_tensor, axis = 3))
                probability = mx.nd.sigmoid(logit)

                probability_label += self._sharpen(probability)
                num_augment += 1

            if "flip_ud" in augment:
                logit = self.net(mx.nd.flip(input_tensor, axis = 2))
                probability = mx.nd.sigmoid(logit)

                probability_label += self._sharpen(probability)
                num_augment += 1
            """

            probability_label = probability_label / num_augment
            #truth_mask  = truth_mask.asnumpy()
            #truth_label = truth_label.asnumpy()
            probability_label = probability_label.asnumpy()

            # For Severstal competition
            predict_label = probability_label > np.array(threshold_label)
        
            for b in range(batch_size):
                for c in range(4):
                    if not predict_label[b, c]:
                        rle = ""
                    else:
                        rle = "1 1"
                    encoded_pixel.append(rle)

            #test_probability_label.append(probability_label)
            #test_truth_mask.append(truth_mask)
            #test_truth_label.append(truth_label)
            test_num += batch_size

            print("\r %4d / %4d  %s" % (test_num, len(test_dataloader._dataset), self.time_to_str((timer() - start), "min")), end = "", flush = True)

        assert(test_num == len(test_dataloader._dataset))
        self.logger.write("")

        """
        #test_image = np.concatenate(test_image)
        #test_probability = np.concatenate(test_probability)
        test_probability_label = np.concatenate(test_probability_label)
        #test_truth_mask  = np.concatenate(test_truth_mask)
        #test_truth_label = np.concatenate(test_truth_label)
        """

        df = pd.DataFrame(zip(test_df["ImageId_ClassId"].tolist(), encoded_pixel), columns=["ImageId_ClassId", "EncodedPixels"])

        return df
    
###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file is the main entry point of the solution.                          #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-10-06                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import os
import time
import warnings
import random
import cv2
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
import pickle
import torch
from datetime import date
import datetime
import gc
from tqdm import tqdm

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD = [0.229, 0.224, 0.225]

def image_to_input(image, rbg_mean, rbg_std):
    input = image.astype(np.float32)
    input = input[..., ::-1] / 255
    input = input.transpose(0, 3, 1, 2)
    input[:, 0] = (input[:, 0] - rbg_mean[0]) / rbg_std[0]
    input[:, 1] = (input[:, 1] - rbg_mean[1]) / rbg_std[1]
    input[:, 2] = (input[:, 2] - rbg_mean[2]) / rbg_std[2]
    
    return input

def onehot_to_index(onehot, num_class = 4):
    index = torch.arange(0, num_class) + 1
    index = index.to(onehot.device).view(1, num_class, 1, 1).float()
    index = index * onehot
    index = index.max(1, keepdim = True)[0]
    index = index.long()

    return index

def mask_to_label(mask, threshold=0):
    label = (mask.sum(-1).sum(-1) > threshold).float()

    return label

def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_mask = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_mask.append(batch[b][1])

    input = np.stack(input)
    input = image_to_input(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)
    input = torch.from_numpy(input).float()

    truth_mask = np.stack(truth_mask)
    truth_mask = torch.from_numpy(truth_mask).float()

    with torch.no_grad():
        truth_label = mask_to_label(truth_mask).float()
        truth_mask = onehot_to_index(truth_mask)

    return input, truth_mask

def remove_small_one(predict, min_size):
    H, W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H, W), np.bool)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = True

    return predict

def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b, c] = remove_small_one(predict[b, c], min_size[c])
    return predict

def run_length_encode(mask):
    #possible bug for here

    m = mask.T.flatten()

    if m.sum() == 0:
        rle = ""
    else:
        m = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = " ".join(str(r) for r in run)

    return rle

def print_submission_csv(df):

    text = ""
    df["Class"] = df["ImageId_ClassId"].str[-1].astype(np.int32)
    df["Label"] = (df["EncodedPixels"] != "").astype(np.int32)
    pos1 = ((df["Class"] == 1) & (df["Label"] == 1)).sum()
    pos2 = ((df["Class"] == 2) & (df["Label"] == 1)).sum()
    pos3 = ((df["Class"] == 3) & (df["Label"] == 1)).sum()
    pos4 = ((df["Class"] == 4) & (df["Label"] == 1)).sum()

    num_image = len(df) // 4
    num = len(df)
    pos = (df["Label"] == 1).sum()
    neg = num - pos

    text += "compare with LB probing ... \n"
    text += "\t\tnum_image = %5d(1801) \n" % num_image
    text += "\t\tnum  = %5d(7204) \n" % num
    text += "\t\tneg  = %5d(6172)  %0.3f \n" % (neg, neg / num)
    text += "\t\tpos  = %5d(1032)  %0.3f \n" % (pos, pos / num)
    text += "\t\tpos1 = %5d( 128)  %0.3f  %0.3f \n" % (pos1, pos1 / num_image, pos1 / pos)
    text += "\t\tpos2 = %5d(  43)  %0.3f  %0.3f \n" % (pos2, pos2 / num_image, pos2 / pos)
    text += "\t\tpos3 = %5d( 741)  %0.3f  %0.3f \n" % (pos3, pos3 / num_image, pos3 / pos)
    text += "\t\tpos4 = %5d( 120)  %0.3f  %0.3f \n" % (pos4, pos4 / num_image, pos4 / pos)
    text += " \n"

    return text

# Call to main
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

# Start the timer
start_time = time.time()
current_date = date.today().strftime("%d%m%Y")

sample_submission_path = "../input/severstal-steel-defect-detection/sample_submission.csv"
train_df_path = "../input/severstal-steel-defect-detection/train.csv"
data_folder = "../input/severstal-steel-defect-detection/train_images/"
test_data_folder = "../input/severstal-steel-defect-detection/test_images/"
log_file_path_str = "PyTorchResUnet34Segmentation_submit_" + current_date + ".txt"

out_dir = "" #"E:/Kaggle_kernel_code/result/resnet34-cls-full-foldb0-0/"
augment = ["null", "flip_lr", "flip_ud"] #["null, "flip_lr","flip_ud","5crop"]

# Create a logger
log = Logger()
log.open(log_file_path_str, mode = "a")
log.write("*** PyTorchResUnet34Segmentation Test started... ***\n")
log.write("Date: " + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "\n")
log.write("\n")

log.write("** dataset setting **\n")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 1
num_workers = 4
classes = 4
threshold_pixel = [0.65, 0.65, 0.6, 0.4]
threshold_size  = [600, 600, 1000, 2500]

# Create datasets
labels_df = pd.read_csv(sample_submission_path)
labels_df[["ImageId", "ClassId"]] = labels_df["ImageId_ClassId"].str.split("_", expand = True)
test_items_lst = list(np.load("../input/hengs-data-split-files/test_1801.npy", allow_pickle = True))
test_items_lst = [item.split("/")[1] for item in test_items_lst]
test_df = labels_df.loc[labels_df["ImageId"].isin(test_items_lst)]

test_dataset = SteelSegmentationDataset(labels_df.loc[labels_df["ImageId"].isin(test_df["ImageId"])], test_data_folder, mean, std, "val")

log.write("len test_dataset: " + str(len(test_dataset)) + "\n")

# Create dataloaders
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size, drop_last = False, num_workers = 0, pin_memory = True, collate_fn = null_collate)

## net ----------------------------------------
log.write("** net setting **\n")

weights_paths_lst = ["../input/severstal-comp-final-weights/resunet34_epoch_021_fold0_weights.pth",
                     "../input/severstal-comp-final-weights/resunet34_epoch_026_fold1_weights.pth"]

resunet34_models_lst = []

for weights_path_str in weights_paths_lst:
    net = PyTorchResUnet34Segmenter(model_weights_dir = "../input/downloadresnetpretrained/", batch_size = batch_size, augment = augment, logger = log)
    net.load_trained_model(weights_path_str)
    resunet34_models_lst.append(net)
    
unet_se_resnext50_32x4d = load('/kaggle/input/severstalmodels/unet_se_resnext50_32x4d.pth').cuda()
unet_mobilenet2 = load('/kaggle/input/severstalmodels/unet_mobilenet2.pth').cuda()
unet_resnet34 = load('/kaggle/input/severstalmodels/unet_resnet34.pth').cuda()

class Model:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)

model = Model([unet_se_resnext50_32x4d, unet_mobilenet2, unet_resnet34])

def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.230, 0.225, 0.223)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

# Tried ([0.485, 0.492], [0.456, 0.460], [0.406, 400]), std=([0.230, 229, 232], [0.225, 230, 224], [0.223, 224, 220])
# Best Result with mean=(0.485, 0.456, 0.406), std=(0.230, 0.225, 0.223)

img_folder = '/kaggle/input/severstal-steel-defect-detection/test_images'
num_workers = 0

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)]
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [test_dataloader] + [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]

log.write("Making predictions...\n")
test_num = 0
test_id = []
test_probability_label = []
test_probability_mask = []
test_probability = []
test_truth_label = []
test_truth_mask = []
encoded_pixel = []

#thresholds = [0.5, 0.5, 0.5, 0.5]
#min_area = [600, 600, 1000, 2000]

start = timer()
for t, item in enumerate(tqdm(zip(*loaders))):
    input = item[0][0]
    truth_mask = item[0][1]
    loaders_batch = item[1:3]

    batch_size, C, H, W = input.shape
    current_probability_mask = None

    for net in resunet34_models_lst:
        probability, probability_mask, test_truth_mask = net.predict_from_batch(input, truth_mask)

        if current_probability_mask is None:
            current_probability_mask = probability_mask.copy()
        else:
            current_probability_mask += probability_mask
                        
    
    current_probability_mask /= len(weights_paths_lst)

    predict_mask = current_probability_mask > np.array(threshold_pixel).reshape(1, 4, 1, 1)
    predict_mask = remove_small(predict_mask, threshold_size)
    predict_label = ((predict_mask.sum(-1).sum(-1)) > 0).astype(np.int32)

    for b in range(batch_size):
        for c in range(4):
            if predict_label[b, c] == 0:
                rle = ""
            else:
                rle = run_length_encode(predict_mask[b, c])

            encoded_pixel.append(rle)

    test_num += batch_size

assert(test_num == len(test_dataloader.dataset))

log.write("")
log.write("submitting .... @ %s\n" % str(augment))
log.write("threshold_pixel = %s\n" % str(threshold_pixel))
log.write("threshold_size  = %s\n" % str(threshold_size))
log.write("\n")

log.write("test submission .... @ %s\n" % str(augment))
csv_file = out_dir + "resnet34-softmax-tta-0.50_" + current_date + ".csv"

df = pd.DataFrame(zip(test_df["ImageId_ClassId"].tolist(), encoded_pixel), columns = ["ImageId_ClassId", "EncodedPixels"])
df.to_csv(csv_file, index = False)

## print statistics ----
text = print_submission_csv(df)
log.write("\n")
log.write("%s" % (text))

del resunet34_models_lst
gc.collect()
    
###############################################################################
# First solution for the Severstal Steel Defect Detection competition         #
#                                                                             #
# This file is the main entry point of the solution.                          #
# Developped using Python 3.7.                                                #
#                                                                             #
# Author: Thomas SELECK                                                       #
# e-mail: thomas.seleck@outlook.fr                                            #
# Date: 2019-09-06                                                            #
# Version: 1.0.0                                                              #
###############################################################################

import numpy as np
import pandas as pd
import mxnet as mx
import os
import time
import warnings
import random
from sklearn.model_selection import train_test_split
import cv2
from gluoncv.model_zoo import get_model
from timeit import default_timer as timer
from mxnet.gluon import nn
from mxnet import gluon, image, init, nd
from gluoncv.model_zoo import get_model
import pickle
from datetime import date
import datetime

def remove_small_one(predict, min_size):
    H, W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H, W), np.bool)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = True

    return predict

def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b,c] = remove_small_one(predict[b, c], min_size[c])
    return predict

def print_submission_csv(df):

    text = ""
    df["Class"] = df["ImageId_ClassId"].str[-1].astype(np.int32)
    df["Label"] = (df["EncodedPixels"] != "").astype(np.int32)
    pos1 = ((df["Class"] == 1) & (df["Label"] == 1)).sum()
    pos2 = ((df["Class"] == 2) & (df["Label"] == 1)).sum()
    pos3 = ((df["Class"] == 3) & (df["Label"] == 1)).sum()
    pos4 = ((df["Class"] == 4) & (df["Label"] == 1)).sum()

    num_image = len(df) // 4
    num = len(df)
    pos = (df["Label"] == 1).sum()
    neg = num - pos

    text += "compare with LB probing ... \n"
    text += "\t\tnum_image = %5d(1801) \n" % num_image
    text += "\t\tnum  = %5d(7204) \n" % num
    text += "\t\tneg  = %5d(6172)  %0.3f \n" % (neg, neg / num)
    text += "\t\tpos  = %5d(1032)  %0.3f \n" % (pos, pos / num)
    text += "\t\tpos1 = %5d( 128)  %0.3f  %0.3f \n" % (pos1, pos1 / num_image, pos1 / pos)
    text += "\t\tpos2 = %5d(  43)  %0.3f  %0.3f \n" % (pos2, pos2 / num_image, pos2 / pos)
    text += "\t\tpos3 = %5d( 741)  %0.3f  %0.3f \n" % (pos3, pos3 / num_image, pos3 / pos)
    text += "\t\tpos4 = %5d( 120)  %0.3f  %0.3f \n" % (pos4, pos4 / num_image, pos4 / pos)
    text += " \n"

    return text


# Call to main
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

# Start the timer
start_time = time.time()
current_date = date.today().strftime("%d%m%Y")

sample_submission_path = "../input/severstal-steel-defect-detection/sample_submission.csv"
train_df_path = "../input/severstal-steel-defect-detection/train.csv"
data_folder = "../input/severstal-steel-defect-detection/train_images/"
test_data_folder = "../input/severstal-steel-defect-detection/test_images/"
log_file_path_str = "GluonResNet34Classifier_submit_" + current_date + ".txt"

out_dir = "" #"E:/Kaggle_kernel_code/result/resnet34-cls-full-foldb0-0/"
best_checkpoint = "../input/severstal-comp-final-weights/ResNet34_v1_weights_epoch_21.params"
augment = ["null"] #["null", "flip_lr","flip_ud"] #["null, "flip_lr","flip_ud","5crop"]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 20
num_workers = 4
classes = 4
model_name = "ResNet34_v1"
ctx = mx.gpu(0)

# Create a logger
log = Logger()
log.open(log_file_path_str, mode = "a")
log.write("*** Gluon ResNet34_v1 Test started... ***\n")
log.write("Date: " + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") + "\n")
log.write("\n")
log.write("Model's parameters:\n")
log.write("    Batch size: " + str(batch_size) + "\n")
log.write("    Number of workers: " + str(num_workers) + "\n")
log.write("    Encoder type: " + model_name + "\n")    

# Create datasets
labels_df = pd.read_csv(sample_submission_path)
labels_df[["ImageId", "ClassId"]] = labels_df["ImageId_ClassId"].str.split("_", expand = True)
test_items_lst = list(np.load("../input/hengs-data-split-files/test_1801.npy", allow_pickle = True))
test_items_lst = [item.split("/")[1] for item in test_items_lst]
test_df = labels_df.loc[labels_df["ImageId"].isin(test_items_lst)]

test_dataset = SteelClassificationDataset(labels_df.loc[labels_df["ImageId"].isin(test_df["ImageId"])], test_data_folder, mean, std, "val")

log.write("len(test_dataset): " + str(len(test_dataset)) + "\n")
log.write("\n")

test_loader = mx.gluon.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, last_batch = "keep", num_workers = num_workers)

## net ----------------------------------------
log.write("** net setting **\n")

net = GluonResNetClassifier(model_weights_dir = "../input/severstal-comp-final-weights/", batch_size = batch_size, model_name = model_name, logger = log)
net.load_trained_model(21)
threshold_label = [0.50, 0.50, 0.50, 0.50]
df = net.predict(test_loader, test_df, threshold_label)

# inspect here !!!  ###################
log.write("")
log.write("submitting .... @ %s\n" % str(augment))
log.write("threshold_label = %s\n" % str(threshold_label))
log.write("\n")

log.write("test submission .... @ %s\n" % str(augment))
csv_file = out_dir + "resnet34-cls-tta-0.50_" + current_date + ".csv"

df.to_csv(csv_file, index = False)

## print statistics ----
text = print_submission_csv(df)
log.write("\n")
log.write("%s" % (text))

del net
gc.collect()

import pandas as pd
import numpy as np

df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
df.set_index('ImageId_ClassId', inplace = True)

mask_csv = "resnet34-softmax-tta-0.50_" + current_date + ".csv"
label_csv = "resnet34-cls-tta-0.50_" + current_date + ".csv"

df_mask = pd.read_csv(mask_csv).fillna('')
df_label = pd.read_csv(label_csv).fillna('')

assert(np.all(df_mask['ImageId_ClassId'].values == df_label['ImageId_ClassId'].values))
df_mask.loc[df_label['EncodedPixels']=='','EncodedPixels'] = ''

df_mask.set_index('ImageId_ClassId', inplace=True)

for name, row in df_mask.iterrows():
    df.loc[name] = row

df.reset_index(inplace = True)
df.to_csv('submission.csv', index = False)