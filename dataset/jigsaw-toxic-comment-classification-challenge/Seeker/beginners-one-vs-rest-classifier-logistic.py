# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


#import pandas as pd
#import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('../input/train.csv')


#train.info()

#train.describe()
#train.comment_text.head()

#creating x and y
x=train.loc[:,'comment_text']

y = train.drop(['id','comment_text'],axis=1)

#tokens on alphanumeric
tks = '[A-Za-z0-9]+(?=\\s+)'



# creating pipe line to fit 
#Pipelines help a lot when trying different cominations
pl = Pipeline([
        ('vec', CountVectorizer(token_pattern = tks)),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(x,y)

test = pd.read_csv('../input/test.csv')
test.info()
#1 missing value


test = test.fillna("")
#predicting
predictions = pl.predict_proba(test.comment_text)

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=y.columns,
                             index=test.id,
                             data=predictions)


# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')