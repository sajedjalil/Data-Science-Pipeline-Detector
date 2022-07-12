

# Any results you write to the current directory are saved as output.
import pandas as pd
from bs4 import BeautifulSoup
# import spacy
import string
import re
import glob
import numpy as np
import os
import numpy as np
np.random.seed(1337)
# from __future__ import print_function
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Embedding, LSTM, GRU, Activation
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss, f1_score
import sys
from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score, coverage_error, label_ranking_loss
from joblib import Parallel, delayed
import pickle
import csv
#nlp = spacy.load('en')
#table = string.maketrans("","")

