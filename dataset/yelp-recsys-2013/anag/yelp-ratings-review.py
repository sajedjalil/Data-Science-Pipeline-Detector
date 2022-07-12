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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
yelp = pd.read_csv('../input/yelpcsv/yelp.csv')
yelp.head()
yelp.info()
yelp['Text Length'] = yelp['text'].apply(len)
yelp.corr()
import string
from nltk.corpus import stopwords
def text_preprocess(st):
    no_punc = [i for i in st if i not in string.punctuation]
    no_punc = ''.join(no_punc)
    for i in no_punc:
        new_st = no_punc.rstrip('\n\n')
    return [i for i in new_st.split() if i not in stopwords.words('english')]
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars']==5)]
yelp_class['stars'].value_counts()
yelp_class.info()
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_preprocess)),
    ('Classifier', MultinomialNB())
])
from sklearn.model_selection import train_test_split
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
df = pd.DataFrame(data=X_test)
df['Predictions'] = predictions
df.to_csv('Output')