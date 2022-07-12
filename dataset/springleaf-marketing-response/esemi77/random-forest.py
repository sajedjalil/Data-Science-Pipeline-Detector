# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd

# The competition datafiles are in the directory ../input
# List the files we have available to work with
#print("> ls ../input")
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Read train data file:
train = pd.read_csv("../input/train.csv")

# Write summaries of the train and test sets to the log
#print('\nSummary of train dataset:\n')
#print(train.describe())

# coding: utf-8


# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

# Columns with almost same value
mixCol = [8,9,10,11,12,18,19,20,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45, 
          73, 74, 98, 99, 100, 106, 107, 108, 156, 157, 158, 159, 166, 167, 168, 169, 176, 177, 178, 179, 180, 
          181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 202, 205, 206, 207, 
          208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 240, 371, 372, 373, 374,
          375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 
          396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 
          437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
          458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
          479, 480, 481, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
          510, 511, 512, 513, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 840]

#Columns with logical datatype
alphaCol = [283, 305, 325, 352, 353, 354, 1934]

#Columns with Places as entries
placeCol = [200, 274, 342]

#Columns with timestamps
dtCol = [75, 204, 217]

selectColumns = []
rmCol = mixCol+alphaCol+placeCol+dtCol
for i in range(1,1935):
    if i not in rmCol:
        selectColumns.append(i)

cols = [str(n).zfill(4) for n in selectColumns]
strColName = ['VAR_' + strNum for strNum in cols] 


# In[27]:

# Use only required columns
nrows = 500
trainData = pd.read_csv("../input/train.csv", skiprows=[107], usecols=strColName, nrows=nrows)
label = pd.read_csv("../input/train.csv", skiprows=[107], usecols=['target'], nrows=nrows)



# In[28]:

from IPython.display import display
# display(trainData.head())
# display(label.head())



# In[31]:

numericFeatures = trainData._get_numeric_data()

# filling na values
removeNA = numericFeatures.fillna(0)

# remove all features that are either one or zero (on or off) in more than 80% of the samples
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(removeNA)


#display(pd.DataFrame(features).head())


# In[32]:

# Tree-based estimators (see the sklearn.tree module and forest of trees in the sklearn.ensemble module)
# can be used to compute feature importances, which in turn can be used to discard irrelevant features:
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
y = np.array(label).ravel()

clf = ExtraTreesClassifier()
X_new = clf.fit(features, y).transform(features)



# In[33]:

from sklearn import preprocessing
X_scaled = preprocessing.scale(X_new)
#display(pd.DataFrame(X_scaled).head())

normalizer = preprocessing.Normalizer().fit(X_scaled)
X_norm = normalizer.transform(X_scaled)  
#display(pd.DataFrame(X_norm).head())


# In[34]:

# Dividing Data into training and crossvalidation sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)


# In[35]:


y_train = np.array(y_train)
#print zip(range(0, len(X_train), 128), range(128, len(X_train), 128))


# In[36]:

from sklearn import svm
clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, 
              tol=0.001, cache_size=1000, class_weight=None, verbose=False, max_iter=-1, random_state=None)
clf.fit(X_train, y_train)  
predictions = clf.predict(X_test)


# In[37]:

from sklearn.metrics import roc_auc_score
print ('roc_auc_score', roc_auc_score(y_test, predictions))


# In[40]:

from sklearn.metrics import mean_squared_error
print ('RMSE', mean_squared_error(y_test, predictions))
