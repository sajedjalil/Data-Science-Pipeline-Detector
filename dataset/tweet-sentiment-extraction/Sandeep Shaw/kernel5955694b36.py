# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
import matplotlib.pyplot as plt

a=pd.read_csv("D:/Kaggle/Tweet/tweet-sentiment-extraction/train.csv")
b=pd.read_csv("D:/Kaggle/Tweet/tweet-sentiment-extraction/test.csv")

a.columns.values

b.columns.values

a[a.columns.values[3]].value_counts(normalize=True)*100
df1=pd.DataFrame(a[a.columns.values[3]].value_counts(normalize=True)*100)
df1['Index']=df1.index
df1.plot.bar(x='Index', y="sentiment", title="Train_Sentiment")

b[b.columns.values[2]].value_counts()
df2=pd.DataFrame(b[b.columns.values[2]].value_counts())
df2['Index']=df2.index
df2.plot.bar(x='Index', y="sentiment", title="Test_Sentiment")

#pie_chart
df1.plot.pie(y="sentiment", title="Train_Sentiment")
df2.plot.pie(y="sentiment", title="Test_Sentiment")

fig, axes = plt.subplots(figsize= (10,8),nrows=2, ncols=2)
df1.plot.bar(x='Index', y="sentiment", title="Train_Sentiment", ax=axes[0,0])
df2.plot.bar(x='Index', y="sentiment", title="Test_Sentiment", ax=axes[0,1])
df1.plot.pie(y="sentiment", title="Train_Sentiment", ax=axes[1,0])
df2.plot.pie(y="sentiment", title="Test_Sentiment", ax=axes[1,1])

#percentage
df1=pd.DataFrame(a[a.columns.values[3]].value_counts(normalize=True)*100)
df1['Index']=df1.index
df2=pd.DataFrame(b[b.columns.values[2]].value_counts(normalize=True)*100)
df2['Index']=df2.index

fig, axes = plt.subplots(figsize= (14,10),nrows=2, ncols=2)
df1.plot.bar(x='Index', y="sentiment", title="Train_Sentiment", ax=axes[0,0])
df2.plot.bar(x='Index', y="sentiment", title="Test_Sentiment", ax=axes[0,1])
df1.plot.pie(y="sentiment", title="Train_Sentiment", ax=axes[1,0])
df2.plot.pie(y="sentiment", title="Test_Sentiment", ax=axes[1,1])

from wordcloud import WordCloud, STOPWORDS

text=str(a[a[a.columns.values[3]]=="positive"][a[a[a.columns.values[3]]=="positive"].columns.values[2]])
type(text)
plt.figure(figsize = (8, 8), facecolor=None) 
wc=WordCloud(width=800, height=800, stopwords=stopwords).generate(text)
plt.imshow(wc)

text=str(a[a[a.columns.values[3]]=="neutral"][a[a[a.columns.values[3]]=="neutral"].columns.values[2]])

text=str(a[a[a.columns.values[3]]=="negative"][a[a[a.columns.values[3]]=="negative"].columns.values[2]])
type(text)
plt.figure(figsize = (8, 8), facecolor=None) 
wc=WordCloud(width=800, height=800, stopwords=stopwords, background_color="white").generate(text)
plt.imshow(wc)

from sklearn.feature_extraction.text import TfidfVectorizer

a[a[a.columns.values[3]]=="negative"][a[a[a.columns.values[3]]=="negative"].columns.values[2]].head()
t=a[a[a.columns.values[3]]=="negative"][a[a[a.columns.values[3]]=="negative"].columns.values[2]]

vect = TfidfVectorizer()
tfidf_matrix = vect.fit_transform(t)
df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
print(df)


