# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS,TfidfVectorizer
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
# %% [code]
def stumbleUpon():
    # Loading the training file
    df_train=pd.read_csv('../input/stumbleupon/train.tsv',delimiter='\t')
    print('Training File Loaded')

    # Loading test file
    df_test=pd.read_csv('../input/stumbleupon/test.tsv',delimiter='\t')
    print('Test File Loaded')

    # Separating labels
    y_train = df_train.label.values
    print('Labels Seperated')

    # Extracting urlid for creating submission file
    urlid_test = df_test.urlid
    print('Extracted urlid from test to make submission file')

    # Getting length of training data
    lentraindata = df_train.shape[0]

    # Seperating usable data from all the rest
    text_train = df_train.boilerplate 
    text_test = df_test.boilerplate

    # Cleaining data
    text_train_clean = [i[9:-3] for i in text_train]
    text_test_clean = [i[9:-3] for i in text_test]

    # Combining text_train and text_test_clean
    text = text_train_clean + text_test_clean
    print('Both are combined')


    # Building count vectorizer 
    vect = TfidfVectorizer(stop_words='english',min_df=3,strip_accents='unicode',token_pattern=r'\w{1,}',ngram_range=(1,2),use_idf=True, smooth_idf=True, sublinear_tf=True)
    vect.fit(text)
    X= vect.transform(text)

    # Seperating test and train file
    X_train= X[:lentraindata]
    X_test = X[lentraindata:]

    # Building the classifier
    
    lr = LogisticRegression(penalty='l2',C=1,max_iter=10000,dual=False)
    print('Cross Val Score: {:.2f}'.format(np.mean(cross_val_score(lr,X_train,y_train,cv=5,scoring='roc_auc'))))
    lr.fit(X_train,y_train)
    # Predicting 
    y_pred = lr.predict(X_test)
    print('Y predicted')
    data = {'urlid':urlid_test,'label':y_pred}
    
    submission_df =pd.DataFrame(data)
    print('Submission DataFrame build')
    submission_df.to_csv('Submission.csv',index=False)
    print('File Submitted')
    
    
if __name__=="__main__":
    stumbleUpon()
    
    
    
