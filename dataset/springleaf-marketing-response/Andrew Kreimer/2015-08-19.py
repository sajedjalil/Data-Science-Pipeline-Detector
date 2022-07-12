# 1. based on original script by: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#    http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html
# 2. based on original script by: https://www.kaggle.com/jiweiliu, FTRL starter code
# 3. based on original script by: Lars Buitinck <L.J.Buitinck@uva.nl>
#    http://scikit-learn.org/stable/auto_examples/text/hashing_vs_dict_vectorizer.html?wb48617274=67251A88#example-text-hashing-vs-dict-vectorizer-py
# 4. based on original script by Harsh, Random Forest
#    https://www.kaggle.com/harshsinha/springleaf-marketing-response/rnadom-forest/run/51121

import pandas as pd
import numpy as np

#########################################################################################################
# feature selection: black list 
#mixed_types = (8,9,10,11,12,43,196,214,225,228,229,231,235,238)
#black_list = (404, 305, 283, 222, 365, 217, 216, 204, 202, 75, 44, 73, 157, 158, 176, 156, 159, 166, 167, 168, 169, 177, 178, 179, 207, 213, 214, 840, 8, 9, 10, 11, 12, 43, 196, 229, 239)
mixed_types = (81, 82, 74, 65, 64, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 32, 33, 34, 35, 36, 37, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57) # white list
black_list = (214, 83, 84, 79, 80, 77, 78, 76, 69, 70, 71, 72, 68, 67, 66, 63, 62, 61, 60, 59, 58, 114, 73, 40, 41, 42, 43, 39, 38, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 21, 20, 19, 18, 17, 8, 9, 10, 11, 12, 75, 44, 404, 305, 283, 222, 202, 204, 216, 217, 466, 467, 493)
black_list += mixed_types

columns = []
black_list_columns = []
for i in range(1,1935):
    if i not in black_list:
        columns.append(i)
    else:
        black_list_columns.append(i)

columns.sort()
black_list_columns.sort()
columns = [str(n).zfill(4) for n in columns]
columns = ['VAR_' + n for n in columns] 
columns.remove('VAR_0240')
columns.remove('VAR_0218')
columns.append('target')
columns.insert(0,'ID')

black_list_columns = [str(n).zfill(4) for n in black_list_columns]
black_list_columns = ['VAR_' + n for n in black_list_columns] 

#########################################################################################################
# get train data
n = 100000 # read from train
#train = pd.read_csv("../input/train.csv", nrows=n)# , usecols=columns
train = pd.read_csv("../input/train.csv")

train.drop(black_list_columns, axis=1, inplace=True)
columns.remove('target')
#print("train: %s" % (str(train.shape)))
#submission_test = pd.read_csv("../input/test.csv", nrows=t)#, usecols=columns,
#submission_test.drop(black_list_columns, axis=1, inplace=True)
submission = 'submission.csv'
#print("test: %s" % (str(submission_test.shape)))

# fill NaN
#train = train.fillna(0)
#submission_test = submission_test.fillna(0)

# feature selection: use only numeric features
#numeric_include = ['int16', 'int32', 'float16', 'int64']
#numeric_exclude = ['float32', 'float64']
#train.select_dtypes(exclude=numeric_exclude)
#submission_test.select_dtypes(exclude=numeric_exclude)
#train = train.select_dtypes(include=numeric_include)
#submission_test = submission_test.select_dtypes(include=numeric_include)

# save IDs for submission
#ids = submission_test.values[:,0]

# remove IDs from train and test
train.drop('ID', axis=1, inplace=True)
#submission_test.drop('ID', axis=1, inplace=True)
print("train: %s" % (str(train.shape)))
#print("test: %s" % (str(submission_test.shape)))

for i, c in enumerate(train):
    #print(i, ":", c, train[c].unique())
    #print(train[c].describe())
    if not np.isreal(train[c][0]):
        print(i,c, train[c].unique())
        #print(train[c].describe())
    else:
        print(i, c, "Numeric")
        print(train[c].describe())