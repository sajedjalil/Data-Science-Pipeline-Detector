# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import time, pickle, os
from tqdm import tqdm
import string

OUT_FILE = r'../output/FeatureEngineering/ListofUniqueTest.pkl'
TEST_FILE = r'../input/test.csv'



def get_words(f, c2r):
    with tqdm(total=2500000) as pbar: # actually something like 2,345,678 or something
        for line in f:
            #b_string = line.replace(',', ' ')
            c_string = line.translate(str.maketrans({key: ' ' for key in c2r}))
            pbar.update()
            for word in c_string.split():
                yield word
            

start_time = time.time()
chars_to_replace = string.punctuation + string.digits


with open(TEST_FILE, encoding="utf8") as infile:
    unique_words = sorted(set(get_words(infile, chars_to_replace)))
### F U KAGGLE NO DUMPING OF PICKLES
#pickle.dump(unique_words, open(OUT_FILE, 'wb'), -1)
elapsed_time = time.time() - start_time
print(elapsed_time)