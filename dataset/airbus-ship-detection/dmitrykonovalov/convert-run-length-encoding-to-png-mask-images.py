# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
print('np.__version__ = ', np.__version__)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print('pd.__version__', pd.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# NOTE!!! 
# STARTED FROM https://www.kaggle.com/inversion/run-length-decoding-quick-start (many thanks ;)

df = pd.read_csv('../input/train_ship_segmentations.csv')
print(df.head())

img_names = df['ImageId'].unique()
print('img_names[:10] = ', img_names[:10])

import keras
print('keras.__version__', keras.__version__)
from keras.preprocessing.image import save_img 

IMG_SHAPE = (768, 768)
OUTPUT_DIR = 'train_y'
# OUTPUT_DIR = '.'
print('OUTPUT_DIR = ', OUTPUT_DIR)

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    print('rle_decode(mask_rle = ', mask_rle)
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


count = 0
for ImageId in img_names:
    print('ImageId', ImageId)
    
    count+= 1
    if count > 10:  # TODO: remove this to run all
        break
    else:
        print('TODO: remove this to run all cases !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    fname = ImageId.replace('.jpg', '.png')
    print('fname', fname)
    
    out_path = os.path.join(OUTPUT_DIR, fname)
    print('out_path = ', out_path)
    
    all_masks = np.zeros(IMG_SHAPE)
    # NOTE: multiple masks for the same image
    img_masks = df.loc[df['ImageId'] == ImageId, 'EncodedPixels'].tolist()
    for mask_rle in img_masks:
        print('mask_rle = ', mask_rle)
        if not pd.isnull(mask_rle):
            all_masks += rle_decode(mask_rle, shape=IMG_SHAPE)
    
    print('np.min(all_masks), np.max(all_masks) = ', np.min(all_masks), np.max(all_masks))
    
    # from keras.preprocessing.image import save_img 
    # Is there a way to save to kaggle?
    # save_img(out_path, all_masks[..., np.newaxis])    # TODO: comment out to save to disk
    print(' TODO save_img(out_path, all_masks[..., np.newaxis])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    