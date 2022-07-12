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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


reviews_train=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
reviews_test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
reviews_train.dropna(axis=0,inplace=True)
text_train,y_table=reviews_train.selected_text,reviews_train.sentiment
text_test,test_table=reviews_test.text,reviews_test.sentiment
text_train = [str.replace("****", "bitch") for str in text_train]
text_train = [str.replace("****", "bitch") for str in text_train]
vect=CountVectorizer(min_df=2).fit(text_train)
X_train=vect.transform(text_train)
X_test=vect.transform(text_test)

# scores=cross_val_score(LogisticRegression(),X_train,y_table,cv=5)
# print("{}".format(np.mean(scores)))
param_grid={'C':[0.2,0.3,0.5,0.6,0.65,0.7,0.75,0.8,0.9,1]}
gird=GridSearchCV(LogisticRegression(),param_grid,cv=5)
gird.fit(X_train,y_table)
y_tabtest=gird.predict(X_test)
result = pd.DataFrame({'textID':reviews_test['textID'].values, 'Survived':y_tabtest.astype(np.str)})
result.to_csv("train_test_table.csv", index=False)