# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!
'''

########################################
## import packages
########################################
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import re
import csv
import codecs
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imp import reload
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate


import sys
reload(sys)
#sys.setdefaultencoding('utf-8')

#####