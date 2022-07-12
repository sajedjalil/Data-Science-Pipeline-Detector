
# Fast inplementation of Run-Length Encoding algorithm
# Ref.: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

import numpy as np # linear algebra
from PIL import Image

def rle (img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    bytes = np.where(img.flatten()==1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b>prev+1): runs.extend((b+1, 0))
        runs[-1] += 1
        prev = b
    
    return ' '.join([str(i) for i in runs])

mask = np.array(Image.open('../input/train_masks/00087a6bd4dc_01_mask.gif'), dtype=np.uint8)
mask_rle = rle(mask)
print(mask_rle)