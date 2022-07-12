# Import
import sys
import os
import subprocess

from six import string_types

# Make sure you have all of these packages installed, e.g. via pip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from skimage import io
from scipy import ndimage
from IPython.display import display


# training files and labels
train_tags = pd.read_csv('../input/train_v2.csv')

print(type(train_tags)) # pandas.cor.frame.DataFrame
print(train_tags.shape) # 40479 rows, 2 colums
print(train_tags.head()) # image_name, tags

tags_lst = train_tags.tags.unique()
print(len(tags_lst))
#print(tags_lst)

#print(tags_lst[0].split(' ') + tags_lst[1].split(' '))

words = []
for tag in tags_lst:
    tmp = tag.split(' ')
    words = words + tmp

#print(tags_uniq_lst[:20])
word_lst = list(set(words))
print(word_lst)

# Words matrix
for wrd in word_lst:
    train_tags[wrd] = train_tags['tags'].apply(lambda x: 1 if wrd in x.split(' ') else 0)
    
# Histogram
train_tags[word_lst].sum().sort_values(ascending=False).plot.bar()

hist = train_tags[word_lst].sum().sort_values(ascending=False)
print(hist)

# co-occurence matrix
m1 = train_tags[word_lst].T
m2 = train_tags[word_lst]
#See diff: 
#co_matrix = m1.dot(m2)
#co_matrix = np.dot(m1, m2)

co_matrix = m1.dot(m2)
print(co_matrix[co_matrix==0])

sns.heatmap(co_matrix)

    
