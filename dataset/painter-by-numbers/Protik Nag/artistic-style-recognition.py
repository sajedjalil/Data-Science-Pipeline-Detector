import numpy as np
import pandas as pd
import scipy as sp
import PIL.Image
import os
from os import listdir
from os.path import isfile, join
from os import walk
from random import sample
from scipy.misc import imresize
import pickle, cv2
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

data_root = '../input/'
csv_file = pd.read_csv('{}{}'.format(data_root,'train_info.csv'))
mask = csv_file['filename'].str.startswith('2')
print(mask)
df = csv_file[mask]
print(df)

train_data_root = data_root + 'train_2/'
x = os.listdir(train_data_root)
print(x)

base_image_path = '{}{}'.format(train_data_root,'20675.jpg')
img = PIL.Image.open(base_image_path)
img