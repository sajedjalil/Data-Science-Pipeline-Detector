# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
import os, glob
os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True" 
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
#from sklearn import GridSearchCV, KFold
import pandas as pd
import numpy as np
np.random.seed(2017)
import cv2
from IPython.core.display import display, HTML
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.DataFrame([[i.split('/')[3],i.split('/')[4],i] for i in glob.glob('../input/train/*/*.jpg')])
train.columns = ['type','image','path']

train_data = []
train_target = []
folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

print (train)
