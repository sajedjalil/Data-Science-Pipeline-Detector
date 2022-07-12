from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python import SKCompat
from sklearn.preprocessing import label_binarize
from tensorflow.contrib import learn
from subprocess import check_output
import tensorflow as tf
import skimage.feature
import numpy as np 
import pandas as pd 
import cv2
import os
# if anyone knows me, they know my imports
# have to be ascending order

print(check_output(["ls", "../input"]).decode("utf8"))
print('# File sizes')
for f in os.listdir('../input'):
    if not os.path.isdir('../input/' + f):
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
    else:
        sizes = [os.path.getsize('../input/'+f+'/'+x)/1000000 for x in os.listdir('../input/' + f)]
        print(f.ljust(30) + str(round(sum(sizes), 2)) + 'MB' + ' ({} files)'.format(len(sizes)))