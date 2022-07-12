# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def RLenc(img,order='F',format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs

mask = cv2.imread('../input/train/1_2_mask.tif',cv2.IMREAD_GRAYSCALE)
mask_rle = RLenc(mask)

#check output
print(mask_rle[:100])
