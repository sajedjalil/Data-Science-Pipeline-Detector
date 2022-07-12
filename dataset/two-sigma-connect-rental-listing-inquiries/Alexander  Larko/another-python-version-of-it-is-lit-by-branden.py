# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
# We need listing_id, description and interest_level for this notebook
train = train[['listing_id','description','interest_level']]
test = test[['listing_id','description']]

train['flag'] = 'train'
test['flag'] = 'test'
full_data = pd.concat([train,test])

from nltk.stem import PorterStemmer
import re
# Removes symbols, numbers and stem the words to reduce dimentional space
stemmer = PorterStemmer()

def clean(x):
    regex = re.compile('[^a-zA-Z ]')
    # For user clarity, broken it into three steps
    i = regex.sub(' ', x).lower()
    i = i.split(" ") 
    i= [stemmer.stem(l) for l in i]
    i= " ".join([l.strip() for l in i if (len(l)>2) ]) # Keeping words that have length greater than 2
    return i
# This takes some time to run. It would be helpful if someone can help me optimize clean() function.
full_data['description_new'] = full_data.description.apply(lambda x: clean(x))
from sklearn.feature_extraction.text import CountVectorizer #Can use tfidffvectorizer as well

cvect_desc = CountVectorizer(stop_words='english', max_features=200)
full_sparse = cvect_desc.fit_transform(full_data.description_new)
 # Renaming words to avoid collisions with other feature names in the model
col_desc = ['desc_'+ i for i in cvect_desc.get_feature_names()] 
count_vect_df = pd.DataFrame(full_sparse.todense(), columns=col_desc)
full_data = pd.concat([full_data.reset_index(),count_vect_df],axis=1)

train =(full_data[full_data.flag=='train'])
test =(full_data[full_data.flag=='test'])

train_f = train.drop(['interest_level','flag','description','index','description_new'],axis=1).columns.values
test_f = test.drop(['interest_level','flag','description','index','description_new'],axis=1).columns.values

train = train[train_f]
test = test[test_f]
#train = pd.DataFrame(train)
#test = pd.DataFrame(test)
train.to_csv('train_desc.csv', index=False)
test.to_csv('test_desc.csv', index=False)
