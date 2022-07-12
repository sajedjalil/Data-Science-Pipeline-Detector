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

from itertools import chain;
def run_length(label):
    x = label.transpose().flatten();
    y = np.where(x>0.5)[0];
    if len(y)<10:# consider as empty
        return [];
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    leng = end - start;
    res = [[s+1,l+1] for s,l in zip(list(start),list(leng))]
    res = list(chain.from_iterable(res));
    return res;
    
mask = cv2.imread('../input/train/1_1_mask.tif',cv2.IMREAD_GRAYSCALE)
mask_rle = run_length(mask);
print(mask_rle[:200])
